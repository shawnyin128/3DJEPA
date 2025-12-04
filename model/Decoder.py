import torch
import torch.nn as nn
import torchvision

from torchvision.models import ConvNeXt_Base_Weights


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.convnext = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        for p in self.convnext.parameters():
            p.requires_grad_(False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        return self

    def forward(self, inputs):
        with torch.no_grad():
            inputs = self.convnext(inputs)
        pooled = inputs.mean(dim=[2, 3])
        outs = self.mlp(pooled)
        return outs
