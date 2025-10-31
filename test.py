import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import TestDataset
from augmentations import get_test_augmentation
from models.unet import get_model


def sliding_window_inference(model, image, patch_size=512, stride=256, device='cuda'):
    """
    Perform sliding window inference on large images
    
    Args:
        model: Trained model
        image: Input image tensor (1, 1, H, W)
        patch_size: Size of patches
        stride: Stride for sliding window
        device: Device to run inference on
    
    Returns:
        prediction: Predicted mask (H, W)
    """
    model.eval()
    
    _, _, h, w = image.shape
    
    # If image is smaller than patch size, process directly
    if h <= patch_size and w <= patch_size:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output = model(image)
                pred = torch.sigmoid(output)
        return pred.squeeze().cpu().numpy()
    
    # Initialize prediction map and count map
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # Calculate number of patches
    n_patches_h = (h - patch_size) // stride + 1
    n_patches_w = (w - patch_size) // stride + 1
    
    with torch.no_grad():
        for i in range(n_patches_h + 1):
            for j in range(n_patches_w + 1):
                # Calculate patch coordinates
                start_h = min(i * stride, h - patch_size)
                start_w = min(j * stride, w - patch_size)
                end_h = start_h + patch_size
                end_w = start_w + patch_size
                
                # Extract patch
                patch = image[:, :, start_h:end_h, start_w:end_w]
                
                # Inference
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    output = model(patch)
                    pred = torch.sigmoid(output)
                
                # Add to prediction map
                pred_np = pred.squeeze().cpu().numpy()
                prediction[start_h:end_h, start_w:end_w] += pred_np
                count_map[start_h:end_h, start_w:end_w] += 1
    
    # Average overlapping predictions
    prediction = prediction / np.maximum(count_map, 1)
    
    return prediction


def test_time_augmentation(model, image, device, n_tta=3):
    """
    Apply test-time augmentation for robust predictions
    
    Args:
        model: Trained model
        image: Input image tensor (1, 1, H, W)
        device: Device to run inference on
        n_tta: Number of TTA iterations (1=no TTA, 5=full TTA)
    
    Returns:
        prediction: Averaged prediction mask
    """
    predictions = []
    
    with torch.no_grad():
        # Original
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            output = model(image)
            pred = torch.sigmoid(output)
        predictions.append(pred)
        
        if n_tta > 1:
            # Horizontal flip
            image_hflip = torch.flip(image, dims=[3])
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output_hflip = model(image_hflip)
                pred_hflip = torch.sigmoid(output_hflip)
            pred_hflip = torch.flip(pred_hflip, dims=[3])
            predictions.append(pred_hflip)
        
        if n_tta > 2:
            # Vertical flip
            image_vflip = torch.flip(image, dims=[2])
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output_vflip = model(image_vflip)
                pred_vflip = torch.sigmoid(output_vflip)
            pred_vflip = torch.flip(pred_vflip, dims=[2])
            predictions.append(pred_vflip)
        
        if n_tta > 3:
            # Both flips
            image_hvflip = torch.flip(image, dims=[2, 3])
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output_hvflip = model(image_hvflip)
                pred_hvflip = torch.sigmoid(output_hvflip)
            pred_hvflip = torch.flip(pred_hvflip, dims=[2, 3])
            predictions.append(pred_hvflip)
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    
    return avg_pred.squeeze().cpu().numpy()


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = get_model(
        model_name=args.model,
        in_channels=1,
        n_classes=1,
        base_channels=args.base_channels
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (Epoch: {checkpoint.get('epoch', 'N/A')}, Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f})")
    
    # Get test images
    test_images = sorted(list(Path(args.input_dir).glob('*.png')))
    print(f"Found {len(test_images)} test images")
    
    # Process each image
    for img_path in tqdm(test_images, desc='Processing'):
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        original_h, original_w = image.shape
        
        # Normalize
        image_norm = image.astype(np.float32) / 255.0
        
        # Resize if needed (keep aspect ratio)
        if args.resize_to is not None:
            h, w = image_norm.shape
            scale = args.resize_to / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image_norm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to square
            pad_h = args.resize_to - new_h
            pad_w = args.resize_to - new_w
            image_padded = np.pad(image_resized, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        else:
            image_padded = image_norm
            new_h, new_w = original_h, original_w
            pad_h, pad_w = 0, 0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_padded).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        
        # Inference
        if args.use_sliding_window:
            prediction = sliding_window_inference(
                model, 
                image_tensor, 
                patch_size=args.patch_size,
                stride=args.stride,
                device=device
            )
        elif args.use_tta:
            prediction = test_time_augmentation(
                model,
                image_tensor,
                device=device,
                n_tta=args.n_tta
            )
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    output = model(image_tensor)
                    prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Remove padding
        if args.resize_to is not None:
            prediction = prediction[:new_h, :new_w]
        
        # Resize back to original size
        if args.resize_to is not None:
            prediction = cv2.resize(prediction, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Threshold
        mask = (prediction > args.threshold).astype(np.uint8) * 255
        
        # Apply morphological operations (post-processing)
        if args.post_process:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Save mask
        output_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(output_path), mask)
        
        # Optionally save overlay
        if args.save_overlay:
            # Create colored overlay
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay[mask > 0] = [0, 255, 0]  # Green for hand
            
            # Blend with original
            blended = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)
            
            overlay_path = output_dir / f"{img_path.stem}_overlay.png"
            cv2.imwrite(str(overlay_path), blended)
    
    print(f"\nProcessing completed! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Hand X-ray Segmentation Model')
    
    # Required parameters
    parser.add_argument('--input_dir', type=str, required=True, help='Path to directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory for masks')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='unet_attention', choices=['unet', 'unet_attention'], help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels')
    
    # Inference parameters
    parser.add_argument('--resize_to', type=int, default=None, help='Resize images to this size (None = keep original)')
    parser.add_argument('--use_sliding_window', action='store_true', help='Use sliding window inference for large images')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for sliding window')
    parser.add_argument('--stride', type=int, default=64, help='Stride for sliding window')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--n_tta', type=int, default=3, help='Number of TTA iterations (1-5)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary mask')
    parser.add_argument('--post_process', action='store_true', help='Apply morphological post-processing')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlay visualization')
    
    # Other parameters
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    main(args)