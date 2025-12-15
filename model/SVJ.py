import torch
import torch.nn as nn

from model.Predictor import SimpleSlotPredictor


class SVJ(nn.Module):
    def __init__(self, convnext, encoder, hidden_size, head_num, kv_head_num, head_dim, num_layer=2, num_latent=128, prior_length = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.prior_length = prior_length

        self.convnext = convnext
        self.encoder = encoder
        self.view_embed = nn.Embedding(prior_length, hidden_size)
        nn.init.normal_(self.view_embed.weight, mean=0.0, std=0.02)

        # 极简预测器（无注意力），输出 (u,v,depth,scales,quats,opacity,colors)
        self.predictor = SimpleSlotPredictor(
            hidden_size=hidden_size,
            num_latent=num_latent,
            d_min=1.0,
            d_max=4.0,
            base_scale=0.05,
            color_dim=3,
            opacity_bias=3.0,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        self.encoder.eval()
        return self

    def forward(self, images, actions, K_ref: torch.Tensor, cam_to_world_ref: torch.Tensor, H: int, W: int):
        B, T, C, H, W = images.shape

        x0 = images[:, 0]
        with torch.no_grad():
            feat0 = self.convnext(x0)
        B_, C_, H_, W_ = feat0.shape
        x_prev = feat0.view(B_, C_, H_ * W_).transpose(1, 2)

        # 对齐 Stage-1 的 action 使用与 token 传递语义：
        # - 每步仅将“图像 tokens”传给下一步（丢弃动作 token）
        # - 使用图像 tokens 的均值作为该视角的先验向量
        assert self.prior_length <= actions.size(1), "prior_length 不应超过动作序列长度"
        view_vecs = []  # prior P: [B, K, D]
        for t in range(self.prior_length):
            action_t = actions[:, t, :]                 # [B, 2] (long ids)
            tokens_all = self.encoder(x_prev, action_t) # [B, 1+L, D]
            img_tok = tokens_all[:, 1:, :]              # 去掉动作 token，只保留图像 tokens
            p_t = img_tok.mean(dim=1)                   # [B, D]
            view_vecs.append(p_t)
            x_prev = img_tok.detach()                   # 与 Stage-1 对齐，避免污染与无谓梯度

        P = torch.stack(view_vecs, dim=1)               # [B, K, D]

        # 视角位置嵌入（与先验向量同形相加）
        view_idx = torch.arange(self.prior_length, device=images.device)
        view_emb = self.view_embed(view_idx).view(1, self.prior_length, self.hidden_size)  # [1, K, D]

        tokens = P + view_emb                           # [B, K, D]

        # 预测像素参数与外观
        uv, depth, scales, quats, opacities, colors = self.predictor(tokens)  # uv:[B,N,2], depth:[B,N,1]

        # 相机锚定反投影：将 (u,v,depth) 转到世界坐标的 means
        # K_ref: [B,3,3]; cam_to_world_ref: [B,4,4]
        fx = K_ref[:, 0, 0].unsqueeze(1)  # [B,1]
        fy = K_ref[:, 1, 1].unsqueeze(1)
        cx = K_ref[:, 0, 2].unsqueeze(1)
        cy = K_ref[:, 1, 2].unsqueeze(1)

        u_px = uv[..., 0] * float(W)  # [B,N]
        v_px = uv[..., 1] * float(H)  # [B,N]
        z = depth.squeeze(-1)         # [B,N]

        x = (u_px - cx) / fx          # [B,N]
        y = (v_px - cy) / fy          # [B,N]
        X = x * z
        Y = y * z
        Z = z
        p_cam = torch.stack([X, Y, Z], dim=-1)  # [B,N,3]
        ones = torch.ones(B, p_cam.size(1), 1, device=p_cam.device, dtype=p_cam.dtype)
        p_cam_h = torch.cat([p_cam, ones], dim=-1)      # [B,N,4]
        means_h = torch.einsum('bij,bnj->bni', cam_to_world_ref, p_cam_h)  # [B,N,4]
        means = means_h[..., :3]

        return means, quats, scales, opacities, colors
