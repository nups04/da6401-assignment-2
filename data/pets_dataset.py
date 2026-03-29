
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet multi-task dataset loader.
    Provides:
    - Image
    - Breed label (0-36)
    - Bounding box [x_center, y_center, width, height] (normalized)
    - Segmentation mask (trimap: 0=Pet, 1=Background, 2=Border)
    """

    def __init__(self, root_dir, split="trainval", transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'data').
            split (str): 'trainval' or 'test'.
            transform: Albumentations transformation pipeline.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")
        self.trimaps_dir = os.path.join(self.annotations_dir, "trimaps")

        split_file = os.path.join(self.annotations_dir, f"{split}.txt")
        self.image_names = []
        self.labels = []
        
        with open(split_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                self.image_names.append(parts[0])
                # Class IDs are 1-37 in the file, convert to 0-36
                self.labels.append(int(parts[1]) - 1)

    def __len__(self):
        return len(self.image_names)

    def _parse_xml(self, xml_path):
        """Parse PASCAL VOC XML for head bounding box."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        # Dataset XMLs have a single 'object' for the head ROI
        obj = root.find("object")
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        return xmin, ymin, xmax, ymax, width, height

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.labels[idx]

        # 1. Load Image
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # 2. Load Mask (Trimap)
        mask_path = os.path.join(self.trimaps_dir, f"{img_name}.png")
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        # Original trimap values: 1 (foreground), 2 (background), 3 (border)
        # Convert to 0, 1, 2 for training
        mask_np = (mask_np - 1).astype(np.int64)

        # 3. Load Bounding Box
        xml_path = os.path.join(self.xmls_dir, f"{img_name}.xml")
        if os.path.exists(xml_path):
            xmin, ymin, xmax, ymax, w, h = self._parse_xml(xml_path)
            # Coordinates in [xmin, ymin, xmax, ymax] format
            bboxes = [[xmin, ymin, xmax, ymax]]
        else:
            # For missing XMLs, use the whole image as a dummy box or handle as needed
            # Since the task requires head ROI, we should ideally handle this.
            # Oxford-IIIT Pet has most head ROIs.
            bboxes = [[0, 0, image_np.shape[1], image_np.shape[0]]]

        # 4. Apply Transforms (Albumentations)
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np, bboxes=bboxes, category_ids=[label])
            image_np = augmented['image']
            mask_np = augmented['mask']
            bboxes = augmented['bboxes']

        # 5. Prepare Output
        # Convert image to Tensor (B, C, H, W) and scale [0, 1]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).long()

        # Convert bbox to [x_center, y_center, width, height] and normalize by image dimensions
        # Albumentations 'pascal_voc' format is [xmin, ymin, xmax, ymax]
        if len(bboxes) > 0:
            xmin, ymin, xmax, ymax = bboxes[0]
            # Get current height and width from transformed image
            h_new, w_new = image_np.shape[:2]
            
            x_center = (xmin + xmax) / 2.0 / w_new
            y_center = (ymin + ymax) / 2.0 / h_new
            bbox_width = (xmax - xmin) / w_new
            bbox_height = (ymax - ymin) / h_new
            bbox_tensor = torch.tensor([x_center, y_center, bbox_width, bbox_height], dtype=torch.float32)
        else:
            bbox_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "bbox": bbox_tensor,
            "mask": mask_tensor
        }
