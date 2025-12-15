import math
import torch
import gsplat

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.dataset_utils import ShapeNetDataset


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


def render_gaussians_single_cam(means: torch.Tensor,      # [N, 3]
                                quats: torch.Tensor,      # [N, 4]
                                scales: torch.Tensor,     # [N, 3]
                                opacities: torch.Tensor,  # [N]
                                colors: torch.Tensor,     # [N, 3] / [N, D]
                                viewmat: torch.Tensor,    # [4, 4] world -> cam
                                K: torch.Tensor,          # [3, 3]
                                H: int,
                                W: int):
    # 归一化 quaternion
    quats_normed = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    render_colors, render_alphas, meta = gsplat.rasterization(
        means,                      # [N, 3]
        quats_normed,               # [N, 4]
        scales,                     # [N, 3]
        torch.sigmoid(opacities),   # [N]
        torch.sigmoid(colors),      # [N, D]
        viewmat[None],              # [1, 4, 4]
        K[None],                    # [1, 3, 3]
        W,
        H,
        packed=False
    )

    # 返回 [H, W, D] / [H, W, 1]
    return render_colors[0], render_alphas[0], meta


# =============== 下面是仅用于测试的小工具 ===============

def _init_random_gaussians(num_points: int,
                           device: str = "cuda",
                           color_dim: int = 3):
    """
    只给 test 用的随机初始化，高斯参数形状和以后 JEPA 输出对齐。
    """
    device = torch.device(device)

    bd = 2.0
    means = bd * (torch.rand(num_points, 3, device=device) - 0.5)
    scales = torch.rand(num_points, 3, device=device)
    colors = torch.rand(num_points, color_dim, device=device)

    # 随机 quaternion（和官方 example 一样）
    u = torch.rand(num_points, 1, device=device)
    v = torch.rand(num_points, 1, device=device)
    w = torch.rand(num_points, 1, device=device)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u)       * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u)       * torch.cos(2.0 * math.pi * w),
        ],
        dim=-1,
    )

    opacities = torch.ones((num_points,), device=device)

    for t in (means, scales, colors, quats, opacities):
        t.requires_grad_(True)

    return means, quats, scales, opacities, colors


def _fit_one_shapenet_view(root: str = "../data/3D",
                           split: str = "test",
                           num_points: int = 20000,
                           iters: int = 200,
                           lr: float = 1e-2,
                           device: str = "cuda",
                           save_path: str = "../data/3dgs_fit_debug.png"):
    """
    小测试：
      1. 从 ShapeNetDataset 里拿一张图 + 相机
      2. 随机初始化一堆 Gaussians
      3. 用上面的 render_gaussians_single_cam 去 fit 这张图
      4. 保存 GT vs Rendered 对比图到 save_path
    """
    device = torch.device(device)

    # 1. 取一个样本：imgs: [B, T, 3, H, W], cams: [B, T, 4, 4]
    ds = ShapeNetDataset(root=root,
                         split=split,
                         return_cam=True,
                         max_objs_per_synset=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    imgs, cams, meta = next(iter(loader))

    # 去掉 batch 维度
    imgs = imgs[0]   # [T, 3, H, W]
    cams = cams[0]   # [T, 4, 4]

    # 随便选一个视角 t = 0
    view_img = imgs[0].to(device)   # [3, H, W]
    view_cam = cams[0].to(device)   # [4, 4]
    _, H, W = view_img.shape

    # ground truth: [H, W, 3]
    gt_image = view_img.permute(1, 2, 0)  # [H, W, 3]

    print(f"[INFO] view_img.shape = {view_img.shape}")
    print(f"[INFO] view_cam.shape = {view_cam.shape}")
    print(f"[INFO] meta = {meta}")

    # 2. 用 FOV 构造 intrinsics
    K = make_intrinsics_from_fov(H, W, device=device)

    # 3. 随机初始化 Gaussians
    means, quats, scales, opacities, colors = _init_random_gaussians(
        num_points=num_points,
        device=device,
        color_dim=3,
    )

    optimizer = torch.optim.Adam(
        [means, quats, scales, opacities, colors],
        lr=lr
    )
    mse_loss = nn.MSELoss()

    for it in range(iters):
        optimizer.zero_grad()

        render_colors, _, _ = render_gaussians_single_cam(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmat=view_cam,
            K=K,
            H=H,
            W=W,
        )   # render_colors: [H, W, 3]

        loss = mse_loss(render_colors, gt_image)
        loss.backward()
        optimizer.step()

        if it % 20 == 0 or it == iters - 1:
            print(f"[ITER {it:03d}] loss={loss.item():.4f}")

    # 4. 保存对比图 (GT | PRED)
    with torch.no_grad():
        pred, _, _ = render_gaussians_single_cam(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmat=view_cam,
            K=K,
            H=H,
            W=W,
        )

    # [H, W, 3] -> [3, H, W]
    pred_chw = pred.permute(2, 0, 1).clamp(0.0, 1.0)
    gt_chw = gt_image.permute(2, 0, 1).clamp(0.0, 1.0)

    stacked = torch.stack([gt_chw, pred_chw], dim=0)  # [2, 3, H, W]
    save_image(stacked, save_path, nrow=2)
    print(f"[INFO] Saved comparison image to: {save_path}")


if __name__ == "__main__":
    _fit_one_shapenet_view(
        root="../data/3D",
        split="test",
        num_points=20000,
        iters=10000,     # 你之前觉得 10000 更香就自己调大
        lr=1e-2,
        device="cuda",
        save_path="./vis/3dgs_fit_debug.png",
    )