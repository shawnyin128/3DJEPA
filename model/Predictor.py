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

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)
        out = self.mlp(pooled)
        
        return out


class TDGSPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int, proj_dim: int):
        super().__init__()
        pass
