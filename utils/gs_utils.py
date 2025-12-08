import math
import torch
import torch.nn.functional as F
import gsplat


def make_intrinsics_from_fov(H: int,
                             W: int,
                             fov_x: float = math.pi / 2.0,
                             device: str = "cuda") -> torch.Tensor:
    device = torch.device(device)
    focal = 0.5 * float(W) / math.tan(0.5 * fov_x)
    K = torch.tensor(
        [
            [focal, 0.0,   W / 2.0],
            [0.0,   focal, H / 2.0],
            [0.0,   0.0,   1.0    ],
        ],
        device=device,
        dtype=torch.float32,
    )
    return K


def render_gaussians_single_cam(means: torch.Tensor, # [N, 3]
                                quats: torch.Tensor, # [N, 4]
                                scales: torch.Tensor, # [N, 3]
                                opacities: torch.Tensor, # [N]
                                colors: torch.Tensor, # [N, 3] / [N, D]
                                viewmat: torch.Tensor, # [4, 4] world -> cam
                                K: torch.Tensor, # [3, 3]
                                H: int, W: int):
    quats_normed = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    render_colors, render_alphas, meta = gsplat.rasterization(
        means,
        quats_normed,
        scales,
        torch.sigmoid(opacities),
        torch.sigmoid(colors),
        viewmat[None],
        K[None],
        W,
        H,
        packed=False
    )

    return render_colors[0], render_alphas[0], meta


# TODO
def render_gaussians_batch(
        means, quats, scales_raw,  # scales_raw是未处理的
        opacity_logits, color_logits,
        viewmats, Ks, H, W
):
    B, N, _ = means.shape

    # 四元数归一化
    quats_normed = F.normalize(quats, dim=-1)

    # 确保scales为正（这里scales_raw已经是softplus的输出）
    scales = scales_raw.clamp(min=1e-4, max=10.0)  # 添加clamp防止极值

    opacities = torch.sigmoid(opacity_logits)
    colors = torch.sigmoid(color_logits)

    viewmats = viewmats.unsqueeze(1)
    Ks = Ks.unsqueeze(1)

    render_colors, render_alphas, meta = gsplat.rasterization(
        means,
        quats_normed,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        W, H,
        packed=False
    )
    return render_colors[:, 0], render_alphas[:, 0], meta
