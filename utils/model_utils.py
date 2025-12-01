import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_rope_cache(seq_len: int, dim: int, device, base: float = 10000.0):
    assert dim % 2 == 0, "head_dim for RoPE must be even"
    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_pos_emb_image_only(q, k, cos, sin, num_action_tokens: int):
    B, H, L, D = q.shape
    half = D // 2

    img_start = num_action_tokens
    L_img = L - img_start
    if L_img <= 0:
        return q, k

    q_img = q[:, :, img_start:, :]
    k_img = k[:, :, img_start:, :]

    q1, q2 = q_img[..., :half], q_img[..., half:]
    k1, k2 = k_img[..., :half], k_img[..., half:]

    cos = cos[:L_img].unsqueeze(0).unsqueeze(0)
    sin = sin[:L_img].unsqueeze(0).unsqueeze(0)

    q1_rot = q1 * cos - q2 * sin
    q2_rot = q1 * sin + q2 * cos
    k1_rot = k1 * cos - k2 * sin
    k2_rot = k1 * sin + k2 * cos

    q_img_rot = torch.cat([q1_rot, q2_rot], dim=-1)
    k_img_rot = torch.cat([k1_rot, k2_rot], dim=-1)

    q = q.clone()
    k = k.clone()
    q[:, :, img_start:, :] = q_img_rot
    k[:, :, img_start:, :] = k_img_rot
    return q, k