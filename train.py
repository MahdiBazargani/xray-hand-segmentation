import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from dataset import HandXrayDataset
from augmentations import get_train_augmentation, get_val_augmentation
from models.unet import get_model
from utils.losses import get_loss
from utils.metrics import get_metrics
import segmentation_models_pytorch as smp

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Backward pass
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, metrics_tracker):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    metrics_tracker.reset()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Mixed precision inference
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Update metrics
            metrics_tracker.update(outputs, masks)
            
            # Display current loss
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    metrics = metrics_tracker.compute()
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, best_dice, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Get augmentations
    train_transform = get_train_augmentation(
        image_size=args.image_size,
        patch_size=args.patch_size
    )
    val_transform = get_val_augmentation(image_size=args.image_size)
    
    # Create datasets
    train_dataset = HandXrayDataset(
        root_dir=args.data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = HandXrayDataset(
        root_dir=args.data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = get_model(
        model_name=args.model,
        in_channels=1,
        n_classes=1,
        base_channels=args.base_channels,
        dropout=args.dropout
    )
    # model = smp.Unet(
    #     encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=1,                      # model output channels (number of classes in your dataset)
    # )

    model = model.to(device)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if args.deep_supervision:
        base_loss = get_loss(args.loss)
        criterion = get_loss('deep_supervision', base_loss=base_loss)
    else:
        criterion = get_loss(args.loss)
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Metrics
    metrics_tracker = get_metrics()
    
    # Training loop
    best_dice = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
        
        # Validate
        if epoch % args.val_interval == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics_tracker)
            
            # Log to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | Pixel Acc: {val_metrics['pixel_acc']:.4f}")
            
            # Save best model
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                save_checkpoint(model, optimizer, epoch, best_dice, output_dir / 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint every N epochs
        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, best_dice, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, best_dice, output_dir / 'final_model.pth')
    writer.close()
    
    print(f"\nTraining completed! Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hand X-ray Segmentation Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--image_size', type=int, default=512, help='Image size for training')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for training')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='unet_attention', choices=['unet', 'unet_attention'], help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=32, help='Base number of channels')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'poly', 'none'], help='LR scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--loss', type=str, default='dice', choices=['dice', 'bce', 'bce_dice', 'focal', 'tversky'], help='Loss function')
    parser.add_argument('--deep_supervision', action='store_true', help='Use deep supervision')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval (epochs)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    main(args)