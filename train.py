"""
Training entrypoint for the Multi-Task Perception Pipeline.
Handles Classification, Localization, and Segmentation tasks.
Optimized for high-performance hardware (RTX 4090 + i9-13900K).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def get_transforms(img_size=224, split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def calculate_metrics(outputs, targets, task_type="classification"):
    if task_type == "classification":
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)
    
    elif task_type == "segmentation":
        preds = torch.argmax(outputs, dim=1)
        dice_scores = []
        for cls in range(outputs.shape[1]):
            p = (preds == cls).float()
            t = (targets == cls).float()
            intersection = (p * t).sum()
            dice = (2. * intersection) / (p.sum() + t.sum() + 1e-6)
            dice_scores.append(dice.item())
        return np.mean(dice_scores)

def train_one_epoch(model, loader, optimizer, criterion_dict, device, epoch, scaler):
    model.train()
    total_loss = 0
    num_batches = len(loader)
    
    print(f"Epoch {epoch} [Train] starting...")
    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        bboxes = batch['bbox'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 1. Forward Pass with AMP
        with autocast():
            outputs = model(images)
            loss_cls = criterion_dict['cls'](outputs['classification'], labels)
            loss_loc = criterion_dict['loc'](outputs['localization'], bboxes)
            loss_seg = criterion_dict['seg'](outputs['segmentation'], masks)
            loss = loss_cls + loss_loc + loss_seg

        # 2. Scaled Backward Pass
        scaler.scale(loss).backward()
        
        # 3. Unscale and Step
        scaler.step(optimizer)
        scaler.update()
        
        # Safety: Check for NaN
        if torch.isnan(loss):
            print(f"CRITICAL: NaN loss detected at batch {i}. Skipping update.")
            optimizer.zero_grad()
            continue

        total_loss += loss.item()
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"Batch [{i+1}/{num_batches}], Loss: {loss.item():.4f}")
        
    return total_loss / num_batches

def validate(model, loader, criterion_dict, device):
    model.eval()
    val_metrics = {'loss': 0, 'cls_acc': 0, 'dice_score': 0}
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            bboxes = batch['bbox'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss_cls = criterion_dict['cls'](outputs['classification'], labels)
                loss_loc = criterion_dict['loc'](outputs['localization'], bboxes)
                loss_seg = criterion_dict['seg'](outputs['segmentation'], masks)
                loss = loss_cls + loss_loc + loss_seg
            
            val_metrics['loss'] += loss.item()
            val_metrics['cls_acc'] += calculate_metrics(outputs['classification'], labels, "classification")
            val_metrics['dice_score'] += calculate_metrics(outputs['segmentation'], masks, "segmentation")
            
    for k in val_metrics:
        val_metrics[k] /= len(loader)
    return val_metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    
    # Task-specific setup
    task_mapping = {
        "classification": "classifier.pth",
        "localization": "localizer.pth",
        "segmentation": "unet.pth",
        "multitask": "multitask.pth"
    }
    checkpoint_name = task_mapping.get(args.task, "model.pth")
    
    wandb.init(project="da6401-perception", name=f"{args.task}_{args.run_name}", config=args)
    
    # Dataset & High-Performance Loaders
    train_ds = OxfordIIITPetDataset(root_dir="data", split="trainval", transform=get_transforms(split="train"))
    val_ds = OxfordIIITPetDataset(root_dir="data", split="test", transform=get_transforms(split="val"))
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=16, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=16, pin_memory=True
    )
    
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3).to(device)
    
    if args.freeze_backbone:
        for param in model.encoder.parameters(): param.requires_grad = False
    elif args.partial_freeze:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = True if ("block4" in name or "block5" in name) else False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler() 
    
    criterion_dict = {
        'cls': nn.CrossEntropyLoss(),
        'loc': IoULoss(),
        'seg': nn.CrossEntropyLoss()
    }
    
    # Determine which metric to track for "best" model saving
    best_val_metric = 0
    metric_key = "dice_score" if args.task in ["segmentation", "multitask"] else "cls_acc"
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_dict, device, epoch, scaler)
        val_res = validate(model, val_loader, criterion_dict, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_res['loss']:.4f} | {metric_key}={val_res[metric_key]:.4f} | LR={current_lr:.2e}")
        
        wandb.log({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_res['loss'],
            "val_cls_acc": val_res['cls_acc'], "val_dice": val_res['dice_score'], "lr": current_lr
        })
        
        # Save Best Model using mandatory filenames
        if val_res[metric_key] > best_val_metric:
            best_val_metric = val_res[metric_key]
            checkpoint_payload = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_val_metric,
                "task": args.task
            }
            torch.save(checkpoint_payload, f"checkpoints/{checkpoint_name}")
            print(f"--> Saved best {args.task} model to checkpoints/{checkpoint_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--partial_freeze", action="store_true")
    args = parser.parse_args()
    main(args)
