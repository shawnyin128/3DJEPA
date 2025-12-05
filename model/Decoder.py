import math
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

        self._init_mlp_weights()

    def _init_mlp_weights(self):
        for i, module in enumerate(self.mlp):
            if isinstance(module, nn.Linear):
                if i == 0:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode='fan_in')
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
