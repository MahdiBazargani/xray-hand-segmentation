import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention gate for U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with convolution and downsampling"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and attention"""
    def __init__(self, in_channels, out_channels, skip_channels, use_attention=True, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.use_attention = use_attention
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        if use_attention:
            self.attention = AttentionBlock(F_g=out_channels, F_l=skip_channels, F_int=out_channels // 2)
        
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        if self.use_attention:
            skip = self.attention(g=x, x=skip)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net with Attention Gates for Hand X-ray Segmentation
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        n_classes: Number of output classes (1 for binary segmentation)
        base_channels: Base number of channels (default: 64)
        use_attention: Whether to use attention gates
        dropout: Dropout rate
    """
    def __init__(self, in_channels=1, n_classes=1, base_channels=64, use_attention=True, dropout=0.1):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = EncoderBlock(in_channels, base_channels, dropout)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2, dropout)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4, dropout)
        self.encoder4 = EncoderBlock(base_channels * 4, base_channels * 8, dropout)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, dropout)
        
        # Decoder
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, use_attention, dropout)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, use_attention, dropout)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, use_attention, dropout)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels, use_attention, dropout)
        
        # Final output
        self.outconv = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
        # Deep supervision outputs (optional)
        self.deep_sup4 = nn.Conv2d(base_channels * 8, n_classes, kernel_size=1)
        self.deep_sup3 = nn.Conv2d(base_channels * 4, n_classes, kernel_size=1)
        self.deep_sup2 = nn.Conv2d(base_channels * 2, n_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, deep_supervision=False):
        # Encoder
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder
        x = self.decoder4(x, skip4)
        d4 = x
        
        x = self.decoder3(x, skip3)
        d3 = x
        
        x = self.decoder2(x, skip2)
        d2 = x
        
        x = self.decoder1(x, skip1)
        
        # Final output
        out = self.outconv(x)
        
        if deep_supervision and self.training:
            # Return multiple outputs for deep supervision
            ds4 = self.deep_sup4(d4)
            ds3 = self.deep_sup3(d3)
            ds2 = self.deep_sup2(d2)
            
            # Upsample to original size
            original_size = out.shape[2:]
            ds4 = F.interpolate(ds4, size=original_size, mode='bilinear', align_corners=False)
            ds3 = F.interpolate(ds3, size=original_size, mode='bilinear', align_corners=False)
            ds2 = F.interpolate(ds2, size=original_size, mode='bilinear', align_corners=False)
            
            return [out, ds2, ds3, ds4]
        
        return out


def get_model(model_name='unet', in_channels=1, n_classes=1, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model ('unet', 'unet_attention')
        in_channels: Number of input channels
        n_classes: Number of output classes
        **kwargs: Additional model parameters
    """
    if model_name == 'unet':
        return UNet(in_channels, n_classes, use_attention=False, **kwargs)
    elif model_name == 'unet_attention':
        return UNet(in_channels, n_classes, use_attention=True, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")