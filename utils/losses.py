import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss"""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class DeepSupervisionLoss(nn.Module):
    """Deep Supervision Loss for multi-scale outputs"""
    def __init__(self, base_loss=None, weights=None):
        super(DeepSupervisionLoss, self).__init__()
        
        if base_loss is None:
            self.base_loss = BCEDiceLoss()
        else:
            self.base_loss = base_loss
        
        # Default weights: give more weight to final output
        if weights is None:
            self.weights = [1.0, 0.5, 0.3, 0.2]
        else:
            self.weights = weights
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: List of outputs from different decoder levels
            target: Ground truth mask
        """
        if not isinstance(outputs, list):
            return self.base_loss(outputs, target)
        
        total_loss = 0
        for i, output in enumerate(outputs):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            total_loss += weight * self.base_loss(output, target)
        
        return total_loss


def get_loss(loss_name='bce_dice', **kwargs):
    """
    Factory function to create loss functions
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss
    """
    if loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'bce_dice':
        return BCEDiceLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_name == 'deep_supervision':
        return DeepSupervisionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")