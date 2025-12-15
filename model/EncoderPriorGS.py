import torch
import torch.nn as nn

from model.Predictor import SimpleSlotPredictor


class EncoderPriorGS(nn.Module):
    """
    极简二阶段模型（不依赖 SVJ）：
    - 冻结 Stage-1 的 convnext + encoder，按 K 步动作生成多视角先验 P；
    - 极简预测头输出 (uv, depth, scales, quats, opacities, colors)；
    - 通过参考相机 (K_ref, cam_to_world_ref) 将 (u,v,depth) 反投影为世界系 means。
    """

    def __init__(
        self,
        convnext: nn.Module,
        encoder: nn.Module,
        hidden_size: int,
        num_latent: int = 2048,
        prior_length: int = 3,
        d_min: float = 1.0,
        d_max: float = 4.0,
        base_scale: float = 0.05,
        opacity_bias: float = 3.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.prior_length = prior_length
        self.d_min = float(d_min)
        self.d_max = float(d_max)

        self.convnext = convnext
        self.encoder = encoder
        self.view_embed = nn.Embedding(prior_length, hidden_size)
        nn.init.normal_(self.view_embed.weight, mean=0.0, std=0.02)

        # 极简预测器（无注意力），输出 (u,v,depth,scales,quats,opacity,colors)
        self.predictor = SimpleSlotPredictor(
            hidden_size=hidden_size,
            num_latent=num_latent,
            d_min=d_min,
            d_max=d_max,
            base_scale=base_scale,
            color_dim=3,
            opacity_bias=opacity_bias,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结 backbone 与 encoder
        self.convnext.eval()
        self.encoder.eval()
        return self

    def _unproject_to_world(self, uv, depth, K_ref: torch.Tensor, cam_to_world_ref: torch.Tensor, H: int, W: int):
        """
        将 (u,v,depth) 反投影为世界坐标。
        uv: [B,N,2] in (0,1)；depth: [B,N,1]
        K_ref: [B,3,3]；cam_to_world_ref: [B,4,4]
        """
        B = uv.size(0)
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
        return means

    def forward(self, images, actions, K_ref: torch.Tensor, cam_to_world_ref: torch.Tensor, H: int, W: int):
        """
        返回：
        - means, quats, scales, opacities, colors（渲染参数）
        - 以及 (uv, depth) 便于外部进行可选的残差微调
        """
        B, T, C, H_img, W_img = images.shape
        assert H_img == H and W_img == W, "H/W mismatch with images"

        # 初始图像特征
        x0 = images[:, 0]
        with torch.no_grad():
            feat0 = self.convnext(x0)
        B_, C_, H_, W_ = feat0.shape
        x_prev = feat0.view(B_, C_, H_ * W_).transpose(1, 2)

        # 与 Stage-1 对齐：每步仅使用图像 tokens，丢弃动作 token，并 detach 跨步梯度
        assert self.prior_length <= actions.size(1), "prior_length 不应超过动作序列长度"
        view_vecs = []  # prior P: [B, K, D]
        for t in range(self.prior_length):
            action_t = actions[:, t, :]                 # [B, 2] (long ids)
            tokens_all = self.encoder(x_prev, action_t) # [B, 1+L, D]
            img_tok = tokens_all[:, 1:, :]
            p_t = img_tok.mean(dim=1)                   # [B, D]
            view_vecs.append(p_t)
            x_prev = img_tok.detach()

        P = torch.stack(view_vecs, dim=1)               # [B, K, D]

        # 步嵌入
        view_idx = torch.arange(self.prior_length, device=images.device)
        view_emb = self.view_embed(view_idx).view(1, self.prior_length, self.hidden_size)
        tokens = P + view_emb                           # [B, K, D]

        # 预测像素参数与外观
        uv, depth, scales, quats, opacities, colors = self.predictor(tokens)

        # 反投影为世界坐标
        means = self._unproject_to_world(uv, depth, K_ref, cam_to_world_ref, H, W)

        return means, quats, scales, opacities, colors, (uv, depth)
