import torch
import torch.nn as nn
import torchvision

from torchvision.models import ConvNeXt_Base_Weights

class Decoder(nn.Module):
    """
    Target decoder using frozen ConvNeXt backbone.
    Extracts features and applies lightweight normalization for Stage-1 training.
    """
    
    def __init__(self):
        super().__init__()
        self.convnext = torchvision.models.convnext_base(
            weights=ConvNeXt_Base_Weights.DEFAULT
        ).features
        for p in self.convnext.parameters():
            p.requires_grad_(False)

        # Lightweight normalization on target side, matching online branch's LayerNorm for stability
        # Using affine-free LayerNorm (equivalent to "frozen"), avoids introducing trainable params
        self.norm = nn.LayerNorm(1024, elementwise_affine=False)

    def train(self, mode: bool = True):
        """Keep ConvNeXt in eval mode even when Decoder is in train mode."""
        super().train(mode)
        self.convnext.eval()
        return self

    @torch.no_grad()
    def forward(self, inputs):
        """
        Extract and normalize features.
        Args:
            inputs: [B, 3, H, W] images
        Returns:
            pooled: [B, 1024] normalized features
        """
        feat = self.convnext(inputs) # [B, 1024, H', W']
        pooled = feat.mean(dim=[2, 3]) # [B, 1024]
        pooled = self.norm(pooled)     # Target feature normalization, suppress variance collapse and covariance redundancy
        return pooled