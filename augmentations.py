import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation(image_size=512, patch_size=512):
    """
    Training augmentation pipeline with patch-based cropping
    """
    return A.Compose([
        # Resize to standard size first
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                      border_mode=0, value=0, mask_value=0),
        
        # Geometric transformations (preserve anatomical structure)
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=25, 
            border_mode=0,
            p=0.7
        ),
        
        # Elastic deformation (simulate hand pose variations)
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=0,
            p=0.3
        ),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            p=0.3
        ),
        
        # Random crop for patch-based training
        A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
        
        # Flip augmentation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        
        # Intensity transformations (X-ray specific)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Noise augmentation
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        
        # Simulate different X-ray qualities
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        
        # Coarse dropout (simulate occlusions)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            mask_fill_value=0,
            p=0.3
        ),
    ], additional_targets={})


def get_val_augmentation(image_size=512):
    """
    Validation augmentation pipeline (minimal, only resize/pad)
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                      border_mode=0, value=0, mask_value=0),
    ])


def get_test_augmentation(image_size=512):
    """
    Test augmentation pipeline (resize/pad only, no random ops)
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                      border_mode=0, value=0, mask_value=0),
    ])


def get_tta_augmentation():
    """
    Test-Time Augmentation (TTA) pipeline
    Returns list of transforms for ensemble prediction
    """
    return [
        # Original
        A.Compose([]),
        
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)]),
        
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0)]),
        
        # Rotate 90
        A.Compose([A.Rotate(limit=90, p=1.0, border_mode=0)]),
        
        # Brightness adjustment
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0)]),
    ]