import math
import torch
import torch.nn as nn

from utils.model_utils import build_rope_cache, apply_rotary_pos_emb_image_only, repeat_kv


class ActionEmbedding(nn.Module):
    def __init__(self, num_yaw: int, num_pitch: int, hidden_size: int):
        super().__init__()
        self.yaw_embed = nn.Embedding(num_yaw, hidden_size)
        self.pitch_embed = nn.Embedding(num_pitch, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.yaw_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pitch_embed.weight, mean=0.0, std=0.02)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, yaw_ids: torch.Tensor, pitch_ids: torch.Tensor):
        yaw_tok = self.yaw_embed(yaw_ids)      # [B, D]
        pitch_tok = self.pitch_embed(pitch_ids)  # [B, D]
        concat = torch.cat([yaw_tok, pitch_tok], dim=-1)
        action_vec = self.mlp(concat)          # 融合向量（用于单一动作 token 兼容现有流程）

        # 返回分轴向量与融合向量
        return yaw_tok, pitch_tok, action_vec


class SplitActionRMSNorm(nn.Module):
    """
    分轴动作调制版 RMSNorm (FiLM)：
    - yaw_vec、pitch_vec 分别产生 (gamma, beta)，再做简单平均聚合；
    - 可保持与单一动作向量相同的接口效果，但更强调轴向信息。
    x: [B, L, D]; yaw_vec/pitch_vec: [B, D]
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.affine_y = nn.Linear(hidden_size, hidden_size * 2, bias=True)
        self.affine_p = nn.Linear(hidden_size, hidden_size * 2, bias=True)

        nn.init.xavier_uniform_(self.affine_y.weight)
        nn.init.zeros_(self.affine_y.bias)
        nn.init.xavier_uniform_(self.affine_p.weight)
        nn.init.zeros_(self.affine_p.bias)

    def forward(self, x: torch.Tensor, yaw_vec: torch.Tensor, pitch_vec: torch.Tensor) -> torch.Tensor:
        x_n = self.norm(x)
        gy, by = self.affine_y(yaw_vec).chunk(2, dim=-1)  # [B, D]
        gp, bp = self.affine_p(pitch_vec).chunk(2, dim=-1)
        gamma = 0.5 * (gy + gp)
        beta = 0.5 * (by + bp)
        return x_n * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class EncoderAttention(nn.Module):
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

    def forward(self, inputs):
        input_shape = inputs.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(inputs).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(inputs).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(inputs).view(hidden_shape).transpose(1, 2)

        B, H_q, L, D = query_states.shape
        L_img = L - 1
        if L_img > 0:
            cos, sin = build_rope_cache(
                seq_len=L_img,
                dim=D,
                device=query_states.device
            )
            query_states, key_states = apply_rotary_pos_emb_image_only(
                query_states, key_states, cos, sin,
                num_action_tokens=1,
            )

        key_states = repeat_kv(key_states, self.head_num // self.kv_head_num)
        value_states = repeat_kv(value_states, self.head_num // self.kv_head_num)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.1, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class ActionCrossAttention(nn.Module):
    """
    轻量动作→图像 Cross-Attention（残差用）：
    - q: 图像 tokens [B, L_img, D]
    - k,v: 动作 tokens（建议用 [yaw_vec, pitch_vec] 堆叠成 2 个 token）[B, 2, D]
    heads=2, dropout=0.1
    """
    def __init__(self, hidden_size: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, img_tokens: torch.Tensor, act_tokens: torch.Tensor) -> torch.Tensor:
        # img_tokens: [B, L, D], act_tokens: [B, 2, D]
        B, L, D = img_tokens.shape
        _, A, _ = act_tokens.shape  # A=2
        q = self.q_proj(img_tokens).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, L, d]
        k = self.k_proj(act_tokens).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, A, d]
        v = self.v_proj(act_tokens).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, A, d]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, A]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)  # [B, H, L, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        return self.o_proj(attn_out)


class EncoderFFN(nn.Module):
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


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, head_num, kv_head_num, head_dim):
        super().__init__()
        self.attn = EncoderAttention(hidden_size, head_num, kv_head_num, head_dim)
        self.ffn = EncoderFFN(hidden_size)
        # 分轴动作调制归一化 + 残差门控（由动作控制）
        self.pre_norm = SplitActionRMSNorm(hidden_size)
        self.post_norm = SplitActionRMSNorm(hidden_size)
        # 轻量 Cross-Attention（动作→图像）与其归一化
        self.cross_norm = SplitActionRMSNorm(hidden_size)
        self.act_cross_attn = ActionCrossAttention(hidden_size, num_heads=2, dropout=0.1)
        self.gate_attn = nn.Linear(hidden_size, hidden_size, bias=True)
        self.gate_ffn = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.gate_attn.weight)
        nn.init.zeros_(self.gate_attn.bias)
        nn.init.xavier_uniform_(self.gate_ffn.weight)
        nn.init.zeros_(self.gate_ffn.bias)

    def forward(self, inputs, yaw_vec, pitch_vec):
        # 先做动作→图像的 Cross-Attention（残差）
        a_tok = torch.stack([yaw_vec, pitch_vec], dim=1)  # [B, 2, D]
        img = inputs[:, 1:, :]  # [B, L_img, D]
        x_img = img
        h_img = self.cross_norm(img, yaw_vec, pitch_vec)
        h_img = self.act_cross_attn(h_img, a_tok)
        img = x_img + h_img
        inputs = torch.cat([inputs[:, :1, :], img], dim=1)

        # 注意力子层
        x = inputs
        h = self.pre_norm(inputs, yaw_vec, pitch_vec)
        h = self.attn(h)
        # 残差门控：使用分轴向量的平均作为门控输入
        a_avg = 0.5 * (yaw_vec + pitch_vec)
        g = torch.sigmoid(self.gate_attn(a_avg)).unsqueeze(1)  # [B,1,D]
        inputs = x + g * h

        # FFN 子层
        x = inputs
        h = self.post_norm(inputs, yaw_vec, pitch_vec)
        h = self.ffn(h)
        g = torch.sigmoid(self.gate_ffn(a_avg)).unsqueeze(1)
        inputs = x + g * h
        return inputs


class Encoder(nn.Module):
    def __init__(self, head_num, kv_head_num, head_dim, hidden_size, num_yaw, num_pitch, num_layers):
        super().__init__()
        self.action_embed = ActionEmbedding(num_yaw=num_yaw, num_pitch=num_pitch, hidden_size=hidden_size)
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_size=hidden_size, head_num=head_num, kv_head_num=kv_head_num, head_dim=head_dim) for _ in range(num_layers)
        ])
        self.post_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, actions):
        # encode action tokens
        yaw_ids, pitch_ids = actions[:, 0].long(), actions[:, 1].long()
        yaw_vec, pitch_vec, action_vec = self.action_embed(yaw_ids, pitch_ids)   # [B, D] x3
        action_tokens = action_vec.unsqueeze(1)

        # concat action + image: [B, 2 + H*W, hidden_size]
        # tokens = action_tokens + inputs
        tokens = torch.cat([action_tokens, inputs], dim=1)

        # encode
        for blk in self.blocks:
            tokens = blk(tokens, yaw_vec, pitch_vec)

        # normalize
        tokens = self.post_norm(tokens)

        return tokens
