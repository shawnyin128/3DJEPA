import torch
import torch.nn as nn
import torchvision

from torchvision.models import ConvNeXt_Base_Weights


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnext = torchvision.models.convnext_base(
            weights=ConvNeXt_Base_Weights.DEFAULT
        ).features
        for p in self.convnext.parameters():
            p.requires_grad_(False)

        self.norm = nn.LayerNorm(1024, elementwise_affine=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        return self

    @torch.no_grad()
    def forward(self, inputs):
        feat = self.convnext(inputs) # [B, 1024, H', W']
        pooled = feat.mean(dim=[2, 3]) # [B, 1024]
        pooled = self.norm(pooled)
        return pooled
