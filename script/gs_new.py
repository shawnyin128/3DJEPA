import math
import torch
import torch.nn.functional as F
import gsplat

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from typing import List, Optional

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


# =============== 仅用于测试的小工具（带 encoder 先验的初始化） ===============

def _init_random_gaussians(num_points: int,
                           device: str = "cuda",
                           color_dim: int = 3):
    device = torch.device(device)

    bd = 2.0
    means = bd * (torch.rand(num_points, 3, device=device) - 0.5)
    scales = torch.rand(num_points, 3, device=device)
    colors = torch.rand(num_points, color_dim, device=device)

    # 随机 quaternion
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

    opacities = torch.ones((num_points,), device=device)  # logit=1 → alpha≈0.73

    for t in (means, scales, colors, quats, opacities):
        t.requires_grad_(True)

    return means, quats, scales, opacities, colors


def _init_gaussians_with_encoder_prior(view_img: torch.Tensor,   # [3,H,W]
                                       view_cam: torch.Tensor,   # [4,4] cam->world
                                       K: torch.Tensor,          # [3,3]
                                       num_points: int,
                                       device: str = "cuda",
                                       ckpt_path: str = "../data/checkpoint/jepa_model_stage1_fixed.pth",
                                       prior_length: int = 2):
    """
    最小改动版：使用 Stage-1 的 convnext 特征做显著性先验，
    仅用于初始化高斯点（主训练/渲染循环不变）。
    """
    from model.JEPA import JEPAModel

    device = torch.device(device)
    # 1) 加载 Stage-1 模型（只用 convnext）
    jepa = JEPAModel(
        hidden_size=1024,
        head_dim=128,
        head_num=8,
        kv_head_num=4,
        num_yaw=2,
        num_pitch=3,
        num_layers=2,
    ).to(device)
    use_jepa = True
    try:
        state = torch.load(ckpt_path, map_location=device)
        jepa.load_state_dict(state, strict=False)
        jepa.eval()
        for p in jepa.parameters():
            p.requires_grad_(False)
    except Exception as e:
        # 最小改动：如果 JEPA 权重不可用，退回到 torchvision convnext 的特征
        print(f"[ENC_PRIOR][WARN] 加载 Stage-1 失败，改用 torchvision convnext 特征。原因: {e}")
        use_jepa = False
        import torchvision
        from torchvision.models import ConvNeXt_Base_Weights
        conv = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features.to(device)
        conv.eval()

    # 2) 显著性：优先用 encoder 的图像 tokens（引入 action 序列的前 K 步），否则退回 convnext 通道能量
    with torch.no_grad():
        if use_jepa:
            # convnext → 图像 tokens
            feat = jepa.convnext(view_img[None].to(device))           # [1,C,h,w]
            B, C, h, w = feat.shape
            L = h * w
            x_prev = feat.view(B, C, L).transpose(1, 2)               # [1,L,D]

            # 引入 action：取前 prior_length 步的动作序列（与 Stage-1 的顺序一致），累计显著性
            from utils.action_utils import ActionTokenizer, build_action_tensor
            tok = ActionTokenizer()
            action_seq = build_action_tensor()                         # [T,2] 连续视角差分（float）
            actions_ids = tok.encode_sequence(action_seq, batch_size=1, device=device)  # [1,T,2] (long)

            K_steps = int(min(prior_length, actions_ids.size(1)))
            sal_acc = 0.0
            for t in range(K_steps):
                action_t = actions_ids[:, t, :]                        # [1,2]
                tokens_all = jepa.encoder(x_prev, action_t)             # [1,1+L,D]
                img_tok = tokens_all[:, 1:, :]                          # [1,L,D]
                sal_t = img_tok.pow(2).sum(dim=-1).view(1, 1, h, w)     # [1,1,h,w]
                sal_acc = sal_acc + sal_t
                x_prev = img_tok.detach()                               # 与 Stage-1 对齐

            sal = F.interpolate(sal_acc, size=view_img.shape[1:], mode="bilinear", align_corners=False)[0, 0]
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        else:
            feat = conv(view_img[None].to(device))                     # [1,C,h,w]
            sal = feat.pow(2).sum(dim=1, keepdim=True)                 # [1,1,h,w]
            sal = F.interpolate(sal, size=view_img.shape[1:], mode="bilinear", align_corners=False)[0, 0]
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)

    # 3) 依据显著性抽样 N 个像素位置
    H, W = view_img.shape[1:]
    p = (sal + 1e-6).flatten()
    p = p / p.sum()
    idx = torch.multinomial(p, num_points, replacement=True)
    ys = (idx // W).float()
    xs = (idx % W).float()

    # 4) 反投影到世界系（采用适中的初始深度范围）
    d = torch.empty(num_points, device=device).uniform_(1.0, 3.0)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) / fx * d
    Y = (ys - cy) / fy * d
    Z = d
    p_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)  # [N,4]
    # 注意：数据里保存的是 cam.matrix_world（cam->world）
    means_h = (view_cam.to(device) @ p_cam.t()).t()
    means = means_h[:, :3].contiguous()

    # 5) 其他参数的先验初始化（稳健起步）
    # quats：单位朝向
    quats = torch.zeros(num_points, 4, device=device)
    quats[:, 3] = 1.0
    # scales：小范围，避免一开始就大面片
    scales = torch.full((num_points, 3), 0.05, device=device)  # 常值也可
    # opacities：logit=0 → alpha≈0.5
    opacities = torch.zeros(num_points, device=device)
    # colors：用 GT 像素颜色做 inverse-sigmoid 作为初值（贴近渲染域）
    gt = view_img.permute(1, 2, 0).to(device)  # [H,W,3]
    rgb = gt[ys.long(), xs.long()].clamp(1e-3, 1 - 1e-3)  # [N,3]
    colors = torch.log(rgb / (1 - rgb))

    for t in (means, scales, colors, quats, opacities):
        t.requires_grad_(True)

    print("[ENC_PRIOR] 使用 Encoder 先验完成初始化")
    return means, quats, scales, opacities, colors

def _render_views_grid(means: torch.Tensor,
                       quats: torch.Tensor,
                       scales: torch.Tensor,
                       opacities: torch.Tensor,
                       colors: torch.Tensor,
                       cams: torch.Tensor,   # [T,4,4] cam->world
                       K: torch.Tensor,      # [3,3]
                       view_ids: list,
                       imgs: torch.Tensor,   # [T,3,H,W] 可为 None（无 GT）
                       save_path: str,
                       device: torch.device):
    """将若干视角的 GT 与 Pred 拼网格保存。Pred 在前，GT 在后，两列为一组。"""
    tiles = []
    H = imgs.shape[-2] if imgs is not None else 256
    W = imgs.shape[-1] if imgs is not None else 256
    for t in view_ids:
        t_int = int(t)
        view_cam_t = cams[t_int].to(device)
        viewmat_t = torch.linalg.inv(view_cam_t)
        pred_t, alpha_t, _ = render_gaussians_single_cam(
            means, quats, scales, opacities, colors,
            viewmat_t, K, H, W
        )
        pred_chw = pred_t.permute(2, 0, 1).clamp(0.0, 1.0).cpu()
        tiles.append(pred_chw)
        if imgs is not None:
            gt_t = imgs[t_int].permute(1, 2, 0)
            gt_chw = gt_t.permute(2, 0, 1).clamp(0.0, 1.0).cpu()
            tiles.append(gt_chw)
    if len(tiles) == 0:
        return
    nrow = 2 if imgs is not None else len(view_ids)
    grid = make_grid(torch.stack(tiles, dim=0), nrow=nrow)
    save_image(grid, save_path)
    print(f"[EVAL] Saved multi-view renders to: {save_path}")


def _fit_one_shapenet_view(root: str = "../data/3D",
                           split: str = "test",
                           num_points: int = 20000,
                           iters: int = 200,
                           lr: float = 1e-2,
                           device: str = "cuda",
                           save_path: str = "../data/3dgs_fit_debug.png",
                           use_encoder_prior: bool = True,
                           eval_only: bool = False,
                           eval_iters: int = 300,
                           render_view_ids: Optional[List[int]] = None,
                           render_save_path: str = "./vis/3dgs_infer_multiview.png"):
    """
    小测试（单视角）：
      1. 从 ShapeNetDataset 里拿一张图 + 相机
      2. 用“随机初始化”或“Encoder 先验初始化”一堆 Gaussians
      3. 渲染→MSE 拟合 GT
      4. 保存 GT vs Rendered 对比图到 save_path
    """
    device = torch.device(device)

    # 1. 取一个样本：imgs: [B, T, 3, H, W], cams: [B, T, 4, 4]
    # 使用训练集 & 开启 batch（当前实现仍取 batch 的第一个样本进行拟合；
    # 完整的按批拟合需要为每个样本维护独立的高斯参数，这里保持最小改动）
    ds = ShapeNetDataset(root=root,
                         split="train",
                         return_cam=True,
                         return_obj=True,  # 加载对象矩阵以做姿态补偿
                         max_objs_per_synset=1,
                         synsets=["02958343"])  # 仅汽车类
    BATCH_SIZE = 8
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    # 新返回格式：images, cams, objs, meta
    batch = next(iter(loader))
    if len(batch) == 4:
        imgs, cams, objs, meta = batch
    else:
        # 兼容旧返回（无 obj 矩阵时退化为单位矩阵）
        imgs, cams, meta = batch
        T_tmp = cams.shape[0]
        objs = torch.stack([torch.eye(4) for _ in range(T_tmp)], dim=0)

    # 去掉 batch 维度
    imgs = imgs[0]   # [T, 3, H, W]
    cams = cams[0]   # [T, 4, 4]
    objs = objs[0]   # [T, 4, 4]

    # 选一个视角 t = 0 用于 prior 初始化（训练时将使用多视角监督）
    view_img = imgs[0].to(device)   # [3, H, W]
    view_cam = cams[0].to(device)   # [4, 4]（注意：cam->world）
    _, H, W = view_img.shape

    # ground truth: [H, W, 3]
    gt_image = view_img.permute(1, 2, 0)  # [H, W, 3]

    print(f"[INFO] view_img.shape = {view_img.shape}")
    print(f"[INFO] view_cam.shape = {view_cam.shape}")
    print(f"[INFO] meta = {meta}")

    # 2. 相机内参：改回基于 FOV 的形式（与原版保持一致）
    K = make_intrinsics_from_fov(H, W, device=device)

    # 3. 初始化 Gaussians（优先用 Encoder 先验，失败则回退随机）
    if use_encoder_prior:
        try:
            means, quats, scales, opacities, colors = _init_gaussians_with_encoder_prior(
                view_img, view_cam, K, num_points, device
            )
        except Exception as e:
            print(f"[WARN] 使用 Encoder 先验初始化失败，改用随机初始化。原因: {e}")
            means, quats, scales, opacities, colors = _init_random_gaussians(
                num_points=num_points,
                device=device,
                color_dim=3,
            )
    else:
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

    # 推理路径：仅用单视角 t0 优化若干步，然后渲染多视角
    if eval_only:
        print("[EVAL] Single-view optimization on t0, then render multi-view...")
        for ei in range(eval_iters):
            optimizer.zero_grad()
            viewmat0 = torch.linalg.inv(view_cam)
            pred0, _, _ = render_gaussians_single_cam(
                means, quats, scales, opacities, colors,
                viewmat0, K, H, W
            )
            loss0 = mse_loss(pred0, gt_image)
            loss0.backward(); optimizer.step()
            if ei % 50 == 0 or ei == eval_iters - 1:
                print(f"[EVAL-ITER {ei:03d}] loss0={loss0.item():.4f}")

        # 渲染多视角：默认选 [0, 8] 或 [0, 1]
        T_all = imgs.shape[0]
        if render_view_ids is None:
            render_view_ids = [0]
            if T_all > 1:
                render_view_ids.append(min(8, T_all - 1))
        _render_views_grid(
            means, quats, scales, opacities, colors,
            cams, K, render_view_ids, imgs,
            render_save_path, device
        )
        return

    # 训练目标：基于一个 input（t0），生成两个 view（t0 与 t1），并用两视角联合监督
    for it in range(iters):
        optimizer.zero_grad()

        # 视角集合与 encoder prior 一致：t=0 与 t=1（若 T==1 则退化为单视角）
        T_all = imgs.shape[0]
        idx_list = [0] if T_all == 1 else [0, 1]

        photo = 0.0
        per_view_losses = []

        # 对象姿态补偿：将 t0 姿态下学习到的点，变换到目标视角 t 的对象姿态
        obj0 = objs[0].to(device)
        obj0_inv = torch.linalg.inv(obj0)

        # 课程权重：逐步加大第二视角的权重（前 300 步从 0 线性升到 1）
        w1 = min(1.0, it / 300.0) if len(idx_list) > 1 else 0.0

        l_t0 = None
        l_t1 = None

        for t in idx_list:
            view_img_t = imgs[t].to(device)            # [3,H,W]
            gt_image_t = view_img_t.permute(1, 2, 0)   # [H,W,3]
            view_cam_t = cams[t].to(device)            # cam->world
            viewmat_t = torch.linalg.inv(view_cam_t)   # world->cam

            # means_t = (obj_t @ inv(obj_0)) · means
            obj_t = objs[t].to(device)
            delta = obj_t @ obj0_inv                   # [4,4]
            ones = torch.ones(num_points, 1, device=device)
            means_h = torch.cat([means, ones], dim=-1) # [N,4]
            means_t_h = (delta @ means_h.t()).t()      # [N,4]
            means_t = means_t_h[:, :3]

            # 训练渲染时对不透明度 logit 做降温，抑制早期饱和
            opac_eff = opacities * 0.5
            pred_t, _, _ = render_gaussians_single_cam(
                means=means_t,
                quats=quats,
                scales=scales,
                opacities=opac_eff,
                colors=colors,
                viewmat=viewmat_t,
                K=K,
                H=H,
                W=W,
            )
            l_t = mse_loss(pred_t, gt_image_t)
            if t == 0:
                l_t0 = l_t
            else:
                l_t1 = l_t
            per_view_losses.append(l_t.detach())

        if l_t1 is None:
            photo = l_t0
        else:
            photo = (l_t0 + w1 * l_t1) / (1.0 + w1)

        # 轻量正则：稀疏可见性 + 稳定尺度
        L_opacity = torch.sigmoid(opacities).mean()
        L_scale = (torch.log(scales + 1e-6) ** 2).mean()

        loss = photo + 1e-3 * L_opacity + 3e-3 * L_scale
        loss.backward()
        optimizer.step()

        if it % 20 == 0 or it == iters - 1:
            if len(per_view_losses) > 0:
                pv = torch.stack(per_view_losses)
                print(f"[ITER {it:03d}] views={idx_list} Photo={loss.item():.4f} per-view[min={pv.min().item():.4f} max={pv.max().item():.4f} mean={pv.mean().item():.4f}]")
            else:
                print(f"[ITER {it:03d}] loss={loss.item():.4f}")

    # 4. 保存对比图 (GT | PRED)
    # 保存两视角（或单视角）GT|Pred 拼图，动作/顺序与 prior 保持一致
    with torch.no_grad():
        tiles = []
        out_views = [0] if imgs.shape[0] == 1 else [0, 1]
        obj0 = objs[0].to(device)
        obj0_inv = torch.linalg.inv(obj0)
        for t in out_views:
            view_img_t = imgs[t].to(device)
            gt_image_t = view_img_t.permute(1, 2, 0)
            view_cam_t = cams[t].to(device)
            viewmat_t = torch.linalg.inv(view_cam_t)
            obj_t = objs[t].to(device)
            delta = obj_t @ obj0_inv
            ones = torch.ones(num_points, 1, device=device)
            means_h = torch.cat([means, ones], dim=-1)
            means_t_h = (delta @ means_h.t()).t()
            means_t = means_t_h[:, :3]

            pred_t, _, _ = render_gaussians_single_cam(
                means=means_t,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmat=viewmat_t,
                K=K,
                H=H,
                W=W,
            )
            pred_chw = pred_t.permute(2, 0, 1).clamp(0.0, 1.0)
            gt_chw = gt_image_t.permute(2, 0, 1).clamp(0.0, 1.0)
            tiles.extend([gt_chw.cpu(), pred_chw.cpu()])
        grid = make_grid(torch.stack(tiles, dim=0), nrow=2)
        save_image(grid, save_path)
        print(f"[INFO] Saved comparison image to: {save_path}")


if __name__ == "__main__":
    _fit_one_shapenet_view(
        root="../data/3D",
        split="test",
        num_points=20000,
        iters=600,
        lr=1e-2,
        device="cuda",
        save_path="./vis/3dgs_fit_debug_prior.png",
        use_encoder_prior=True,
        eval_only=False,
        eval_iters=300,
        render_view_ids=None,
        render_save_path="./vis/3dgs_infer_multiview.png",
    )
