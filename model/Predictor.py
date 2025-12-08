import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from utils.model_utils import repeat_kv


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


class TDGSAttention(nn.Module):
    def __init__(self, hidden_size, head_num, kv_head_num, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, head_num * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_head_num * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_head_num * head_dim, bias=False)

        self.o_proj = nn.Linear(head_num * head_dim, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

        nn.init.xavier_uniform_(self.o_proj.weight, gain=1 / math.sqrt(2))

    def forward(self, inputs: torch.Tensor, latent: torch.Tensor):
        input_shape = inputs.shape[:-1] # [B, 192]
        hidden_shape = (*input_shape, -1, self.head_dim)
        latent_shape = latent.shape[:-1] # [B, N]
        latent_hidden_shape = (*latent_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(latent).view(latent_hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(inputs).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(inputs).view(hidden_shape).transpose(1, 2)

        key_states = repeat_kv(key_states, self.head_num // self.kv_head_num)
        value_states = repeat_kv(value_states, self.head_num // self.kv_head_num)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.1, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*latent_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class TDGSFFN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.act = nn.SiLU()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))

        nn.init.xavier_uniform_(self.down_proj.weight, gain=1 / math.sqrt(2))

    def forward(self, inputs):
        down_proj = self.down_proj(self.act(self.gate_proj(inputs)) * self.up_proj(inputs))
        return down_proj


class TDGSBlock(nn.Module):
    def __init__(self, hidden_size, head_num, kv_head_num, head_dim):
        super().__init__()
        self.attn = TDGSAttention(hidden_size, head_num, kv_head_num, head_dim)
        self.ffn = TDGSFFN(hidden_size)

        self.pre_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, tokens, latent):
        x = latent
        x = self.pre_norm(x)
        x = self.attn(tokens, x)
        latent = latent + x

        x = latent
        x = self.post_norm(x)
        x = self.ffn(x)
        latent = latent + x
        return latent


class TDGSParamHead(nn.Module):
    def __init__(self, hidden_size: int, mean_radius: float = 2.0,
                 base_scale: float = 1.0, color_dim: int = 3):  # base_scale改大
        super().__init__()
        self.hidden_size = hidden_size
        self.mean_radius = mean_radius
        self.base_scale = base_scale
        self.color_dim = color_dim

        out_dim = 3 + 3 + 4 + 1 + color_dim
        self.out_proj = nn.Linear(hidden_size, out_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
            with torch.no_grad():
                b = self.out_proj.bias
                scales_start = 3
                b[scales_start:scales_start + 3] = 1.0

                quat_start = 6
                b[quat_start:quat_start + 4] = torch.tensor([0.0, 0.0, 0.0, 1.0])

                opac_start = 10
                b[opac_start] = 2.0

    def forward(self, latent: torch.Tensor):
        x = self.out_proj(latent)
        m, s, q, o_logit, c_logit = torch.split(
            x, [3, 3, 4, 1, self.color_dim], dim=-1
        )

        means = torch.tanh(m) * self.mean_radius

        scales = F.softplus(s + 1.0) * self.base_scale
        scales = scales.clamp(min=0.01, max=5.0)

        quats = F.normalize(q, dim=-1)
        opacity_logits = o_logit.squeeze(-1)
        color_logits = c_logit

        return means, quats, scales, opacity_logits, color_logits


class TDGSPredictor(nn.Module):
    def __init__(self, hidden_size: int,
                 head_num: int,
                 kv_head_num: int,
                 head_dim: int,
                 num_layers: int = 2,
                 num_latent: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_latent = num_latent

        self.latent = nn.Parameter(
            torch.randn(1, num_latent, hidden_size) * 0.1
        )

        self.blocks = nn.ModuleList([
            TDGSBlock(hidden_size, head_num, kv_head_num, head_dim)
            for _ in range(num_layers)
        ])
        self.head = TDGSParamHead(hidden_size)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor]:
        B = tokens.size(0)
        latent = self.latent.expand(B, -1, -1)  # [B, N, D]

        for blk in self.blocks:
            latent = blk(tokens, latent)

        means, quats, scales, opacities, colors = self.head(latent)

        return means, quats, scales, opacities, colors
