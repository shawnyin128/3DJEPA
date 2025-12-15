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
                 base_scale: float = 0.1, color_dim: int = 3):  # 降低初始尺度，便于稳定收敛
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
                b[opac_start] = 4.0  # 提高初始不透明度偏置，避免早期几何梯度消失

    def forward(self, latent: torch.Tensor):
        x = self.out_proj(latent)
        m, s, q, o_logit, c_logit = torch.split(
            x, [3, 3, 4, 1, self.color_dim], dim=-1
        )

        means = torch.tanh(m) * self.mean_radius

        scales = F.softplus(s + 1.0) * self.base_scale
        scales = scales.clamp(min=0.01, max=0.5)  # 收紧上限，防止“大面片”饱和

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


class SimpleSlotPredictor(nn.Module):
    """
    极简 Stage‑2 预测器（无自注意/交叉注意）：
    - 输入：multi‑view prior tokens P∈[B,K,D]（例如每步的 encoder 图像 token 均值）
    - 聚合：g = mean(P) ∈ [B,D]
    - 槽：N 个可学习 slot embedding，与 g 线性融合得到 N×D 潜表示
    - 头：分别预测 (u,v)、depth、scales、quats、opacity、colors
    说明：本类只输出像素平面参数和几何/外观 logits；将 (u,v,depth) 反投影到世界坐标的步骤在 SVJ.forward 中完成。
    """
    def __init__(self,
                 hidden_size: int,
                 num_latent: int = 2048,
                 d_min: float = 1.0,
                 d_max: float = 4.0,
                 base_scale: float = 0.05,
                 color_dim: int = 3,
                 opacity_bias: float = 3.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_latent = num_latent
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.base_scale = float(base_scale)
        self.color_dim = color_dim

        # N 个槽 + 全局先验线性映射
        self.slot_embed = nn.Parameter(torch.randn(1, num_latent, hidden_size) * 0.1)
        self.global_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

        # 头部（各自单层线性，保持极简）
        self.uv_head = nn.Linear(hidden_size, 2)
        self.d_head = nn.Linear(hidden_size, 1)
        self.s_head = nn.Linear(hidden_size, 3)
        self.q_head = nn.Linear(hidden_size, 4)
        self.o_head = nn.Linear(hidden_size, 1)
        self.c_head = nn.Linear(hidden_size, color_dim)

        # 初始化
        nn.init.xavier_uniform_(self.global_linear.weight)
        nn.init.zeros_(self.global_linear.bias)
        for head in [self.uv_head, self.d_head, self.s_head, self.q_head, self.o_head, self.c_head]:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        # 提高初始不透明度偏置，保证早期可见性
        with torch.no_grad():
            self.o_head.bias.add_(opacity_bias)

    def forward(self, tokens: torch.Tensor):
        """
        tokens: [B, K, D]
        返回：
          uv:  [B, N, 2]   （归一化到 0..1）
          d:   [B, N, 1]   （映射到 [d_min, d_max]）
          s:   [B, N, 3]
          q:   [B, N, 4]   （未归一化，调用方可再规范化或直接使用此处的 normalize）
          o:   [B, N]
          c:   [B, N, 3]
        """
        B, K, D = tokens.shape
        g = tokens.mean(dim=1)                      # [B, D]
        g_proj = self.global_linear(g).unsqueeze(1) # [B, 1, D]

        latents = self.slot_embed.expand(B, -1, -1) + g_proj
        latents = self.norm(latents)

        uv = torch.sigmoid(self.uv_head(latents))                     # [B,N,2]
        d = torch.sigmoid(self.d_head(latents)) * (self.d_max - self.d_min) + self.d_min  # [B,N,1]
        s = F.softplus(self.s_head(latents)) * self.base_scale        # [B,N,3]
        # 更严格的上限，避免“大面片”全屏覆盖
        s = s.clamp(min=0.01, max=0.15)
        q = F.normalize(self.q_head(latents), dim=-1)                 # [B,N,4]
        o = self.o_head(latents).squeeze(-1)                          # [B,N]
        c = self.c_head(latents)                                      # [B,N,3]

        return uv, d, s, q, o, c