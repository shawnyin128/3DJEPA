import torch
import torch.nn as nn
import torchvision

from torchvision.models import ConvNeXt_Base_Weights

from utils.model_utils import build_rope_cache, apply_rotary_pos_emb_image_only, repeat_kv


class ActionEmbedding(nn.Module):
    def __init__(self, num_yaw: int, num_pitch: int, hidden_size: int):
        super().__init__()
        self.yaw_embed = nn.Embedding(num_yaw, hidden_size)
        self.pitch_embed = nn.Embedding(num_pitch, hidden_size)
        self.type_embed = nn.Embedding(2, hidden_size)

    def forward(self, yaw_ids: torch.Tensor, pitch_ids: torch.Tensor):
        yaw_tok = self.yaw_embed(yaw_ids)
        pitch_tok = self.pitch_embed(pitch_ids)

        yaw_tok = yaw_tok + self.type_embed(torch.zeros_like(yaw_ids))
        pitch_tok = pitch_tok + self.type_embed(torch.ones_like(pitch_ids))

        action_tokens = torch.stack([yaw_tok, pitch_tok], dim=1)
        return action_tokens


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

    def forward(self, inputs):
        input_shape = inputs.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(inputs).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(inputs).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(inputs).view(hidden_shape).transpose(1, 2)

        B, H_q, L, D = query_states.shape
        L_img = L - 2
        if L_img > 0:
            cos, sin = build_rope_cache(
                seq_len=L_img,
                dim=D,
                device=query_states.device
            )
            query_states, key_states = apply_rotary_pos_emb_image_only(
                query_states, key_states, cos, sin,
                num_action_tokens=2,
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


class EncoderFFN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, inputs):
        down_proj = self.down_proj(self.act(self.gate_proj(inputs)) * self.up_proj(inputs))
        return down_proj


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, head_num, kv_head_num, head_dim):
        super().__init__()
        self.attn = EncoderAttention(hidden_size, head_num, kv_head_num, head_dim)
        self.ffn = EncoderFFN(hidden_size)
        self.pre_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, inputs):
        x = inputs
        inputs = self.pre_norm(inputs)
        inputs = self.attn(inputs)
        inputs = inputs + x

        x = inputs
        inputs = self.post_norm(inputs)
        inputs = self.ffn(inputs)
        inputs = inputs + x
        return inputs


class Encoder(nn.Module):
    def __init__(self, head_num, kv_head_num, head_dim, hidden_size, num_yaw, num_pitch, num_layers):
        super().__init__()
        self.convnext = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        for p in self.convnext.parameters():
            p.requires_grad_(False)

        self.action_embed = ActionEmbedding(num_yaw=num_yaw, num_pitch=num_pitch, hidden_size=hidden_size)
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_size=hidden_size, head_num=head_num, kv_head_num=kv_head_num, head_dim=head_dim) for _ in range(num_layers)
        ])
        self.post_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        return self

    def forward(self, inputs, actions):
        # extract feature map from convnext
        with torch.no_grad():
            img = self.convnext(inputs)
        B, C, H, W = img.shape

        # reshape tokens
        image_tokens = img.view(B, C, H * W).transpose(1, 2)

        # encode action tokens
        yaw_ids, pitch_ids = actions[:, 0].long(), actions[:, 1].long()
        action_tokens = self.action_embed(yaw_ids, pitch_ids)

        # concat action + image: [B, 2 + H*W, hidden_size]
        tokens = torch.cat([action_tokens, image_tokens], dim=1)

        # encode
        for blk in self.blocks:
            tokens = blk(tokens)

        # normalize
        tokens = self.post_norm(tokens)

        return tokens
