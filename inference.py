"""
Inference and evaluation script for Multi-Task Perception Pipeline.
Handles visualizations and TTA (Test-Time Augmentation).
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from models.multitask import MultiTaskPerceptionModel

def load_model(checkpoint_path, device):
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_inference_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def run_tta_prediction(model, image_np, device):
    """
    Runs Test-Time Augmentation (TTA) using:
    - Horizontal Flip
    - Multi-scale (0.9x, 1.0x, 1.1x)
    """
    scales = [0.9, 1.0, 1.1]
    base_size = 224
    
    # Aggregators
    all_cls_logits = []
    all_loc_boxes = []
    all_seg_logits = []
    
    with torch.no_grad():
        for scale in scales:
            size = int(base_size * scale)
            transform = get_inference_transform(size)
            
            # Versions: Original and Flipped
            for flip in [False, True]:
                img = image_np.copy()
                if flip:
                    img = np.ascontiguousarray(img[:, ::-1, :])
                
                # Preprocess
                aug = transform(image=img)
                img_tensor = torch.from_numpy(aug['image']).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Predict
                out = model(img_tensor)
                
                # --- 1. Classification ---
                all_cls_logits.append(out['classification'])
                
                # --- 2. Localization (Bbox) ---
                box = out['localization'].cpu() # [1, 4] -> [xc, yc, w, h]
                if flip:
                    # If flipped, xc becomes (1 - xc)
                    box[0, 0] = 1.0 - box[0, 0]
                all_loc_boxes.append(box)
                
                # --- 3. Segmentation ---
                seg = out['segmentation'] # [1, 3, size, size]
                # Resize back to base_size (224x224)
                seg = F.interpolate(seg, size=(base_size, base_size), mode='bilinear', align_corners=False)
                if flip:
                    seg = torch.flip(seg, dims=[3])
                all_seg_logits.append(seg)

    # Aggregate Results
    avg_cls = torch.stack(all_cls_logits).mean(dim=0)
    avg_loc = torch.stack(all_loc_boxes).mean(dim=0)
    avg_seg = torch.stack(all_seg_logits).mean(dim=0)
    
    return {
        'classification': avg_cls,
        'localization': avg_loc,
        'segmentation': avg_seg
    }

def visualize_feature_maps(model, image_tensor, layer_idx=0):
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    if layer_idx == 0:
        handle = model.encoder.block1[0].register_forward_hook(hook_fn)
    else:
        handle = model.encoder.block5[3].register_forward_hook(hook_fn)
        
    with torch.no_grad():
        model.encoder(image_tensor)
        
    handle.remove()
    return feature_maps[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inference script with TTA (Flip + Scale) ready.")

if __name__ == "__main__":
    main()
