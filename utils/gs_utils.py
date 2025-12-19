import math
import torch
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


def rotmat3_to_quat_xyzw(R: torch.Tensor):
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    q_abs = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    quat = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)

    m = torch.argmax(q_abs, dim=-1, keepdim=True)
    q_max = torch.gather(q_abs, -1, m).squeeze(-1)
    m = m.squeeze(-1)
    q_v = 0.5 * torch.sqrt(torch.maximum(q_max, torch.zeros_like(q_max)))

    idx = (m == 0)
    if idx.any():
        denom = 0.25 / q_v[idx]
        quat[idx, 0] = (m12[idx] - m21[idx]) * denom
        quat[idx, 1] = (m20[idx] - m02[idx]) * denom
        quat[idx, 2] = (m01[idx] - m10[idx]) * denom
        quat[idx, 3] = q_v[idx]

    idx = (m == 1)
    if idx.any():
        denom = 0.25 / q_v[idx]
        quat[idx, 0] = q_v[idx]
        quat[idx, 1] = (m01[idx] + m10[idx]) * denom
        quat[idx, 2] = (m20[idx] + m02[idx]) * denom
        quat[idx, 3] = (m12[idx] - m21[idx]) * denom

    idx = (m == 2)
    if idx.any():
        denom = 0.25 / q_v[idx]
        quat[idx, 0] = (m01[idx] + m10[idx]) * denom
        quat[idx, 1] = q_v[idx]
        quat[idx, 2] = (m12[idx] + m21[idx]) * denom
        quat[idx, 3] = (m20[idx] - m02[idx]) * denom

    idx = (m == 3)
    if idx.any():
        denom = 0.25 / q_v[idx]
        quat[idx, 0] = (m20[idx] + m02[idx]) * denom
        quat[idx, 1] = (m12[idx] + m21[idx]) * denom
        quat[idx, 2] = q_v[idx]
        quat[idx, 3] = (m01[idx] - m10[idx]) * denom

    return quat


def quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)
