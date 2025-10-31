import torch
import numpy as np


class DiceScore:
    """Dice coefficient metric"""
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        self.dice_sum = 0
        self.count = 0
    
    def update(self, pred, target):
        pred = torch.sigmoid(pred) if pred.dtype == torch.float32 else pred
        pred = (pred > self.threshold).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        self.dice_sum += dice.item()
        self.count += 1
    
    def compute(self):
        return self.dice_sum / max(self.count, 1)
    
    def __call__(self, pred, target):
        self.update(pred, target)
        return self.compute()


class IoUScore:
    """Intersection over Union (Jaccard Index) metric"""
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        self.iou_sum = 0
        self.count = 0
    
    def update(self, pred, target):
        pred = torch.sigmoid(pred) if pred.dtype == torch.float32 else pred
        pred = (pred > self.threshold).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        self.iou_sum += iou.item()
        self.count += 1
    
    def compute(self):
        return self.iou_sum / max(self.count, 1)
    
    def __call__(self, pred, target):
        self.update(pred, target)
        return self.compute()


class PixelAccuracy:
    """Pixel-wise accuracy metric"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, pred, target):
        pred = torch.sigmoid(pred) if pred.dtype == torch.float32 else pred
        pred = (pred > self.threshold).float()
        
        correct = (pred == target).sum().item()
        total = target.numel()
        
        self.correct += correct
        self.total += total
    
    def compute(self):
        return self.correct / max(self.total, 1)
    
    def __call__(self, pred, target):
        self.update(pred, target)
        return self.compute()


class Precision:
    """Precision metric (positive predictive value)"""
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        self.precision_sum = 0
        self.count = 0
    
    def update(self, pred, target):
        pred = torch.sigmoid(pred) if pred.dtype == torch.float32 else pred
        pred = (pred > self.threshold).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        
        self.precision_sum += precision.item()
        self.count += 1
    
    def compute(self):
        return self.precision_sum / max(self.count, 1)


class Recall:
    """Recall metric (sensitivity, true positive rate)"""
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        self.recall_sum = 0
        self.count = 0
    
    def update(self, pred, target):
        pred = torch.sigmoid(pred) if pred.dtype == torch.float32 else pred
        pred = (pred > self.threshold).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        
        self.recall_sum += recall.item()
        self.count += 1
    
    def compute(self):
        return self.recall_sum / max(self.count, 1)


class MetricTracker:
    """Track multiple metrics"""
    def __init__(self, metrics_dict):
        self.metrics = metrics_dict
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, pred, target):
        for metric in self.metrics.values():
            metric.update(pred, target)
    
    def compute(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}


def get_metrics():
    """Get default metrics for evaluation"""
    return MetricTracker({
        'dice': DiceScore(),
        'iou': IoUScore(),
        'pixel_acc': PixelAccuracy(),
        'precision': Precision(),
        'recall': Recall()
    })