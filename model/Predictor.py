import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class EncoderPredictor(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim

        # 最小门控：图像均值 + 动作门控（简洁稳定，易于报告说明）
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.gate_drop = nn.Dropout(p=0.3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for i, module in enumerate(self.mlp):
            if isinstance(module, nn.Linear):
                if i == 0:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode='fan_in')
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, 1+L, D]
        act = tokens[:, 0, :]              # [B, D]
        img_mean = tokens[:, 1:, :].mean(dim=1)  # [B, D]
        pooled = img_mean + 0.5 * self.gate_drop(F.silu(self.gate(act)))
        return self.mlp(pooled)
