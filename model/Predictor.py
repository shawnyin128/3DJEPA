import math
import torch.nn as nn


class EncoderPredictor(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim

        self.mlp=nn.Sequential(
            nn.Linear(hidden_size ,proj_dim),
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

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)
        out = self.mlp(pooled)
        
        return out


class TDGSPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int, proj_dim: int):
        super().__init__()
        pass
