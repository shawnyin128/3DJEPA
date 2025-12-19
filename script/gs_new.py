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
    # Normalize quaternion
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

    # Return [H, W, D] / [H, W, 1]
    return render_colors[0], render_alphas[0], meta


# =============== Testing utilities (with encoder prior initialization) ===============

def _init_random_gaussians(num_points: int,
                           device: str = "cuda",
                           color_dim: int = 3):
    device = torch.device(device)

    bd = 2.0
    means = bd * (torch.rand(num_points, 3, device=device) - 0.5)
    scales = torch.rand(num_points, 3, device=device)
    colors = torch.rand(num_points, color_dim, device=device)

    # Random quaternion
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
    Minimal modification version: uses Stage-1 convnext features for saliency prior,
    only for initializing Gaussian points (main training/rendering loop unchanged).
    """
    from model.JEPA import JEPAModel

    device = torch.device(device)
    # 1) Load Stage-1 model (only use convnext)
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
        # Minimal modification: if JEPA weights unavailable, fall back to torchvision convnext features
        print(f"[ENC_PRIOR][WARN] Failed to load Stage-1, using torchvision convnext features. Reason: {e}")
        use_jepa = False
        import torchvision
        from torchvision.models import ConvNeXt_Base_Weights
        conv = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features.to(device)
        conv.eval()

    # 2) Saliency: prioritize encoder's image tokens (introducing first K steps of action sequence), otherwise fallback to convnext channel energy
    with torch.no_grad():
        if use_jepa:
            # convnext → image tokens
            feat = jepa.convnext(view_img[None].to(device))           # [1,C,h,w]
            B, C, h, w = feat.shape
            L = h * w
            x_prev = feat.view(B, C, L).transpose(1, 2)               # [1,L,D]

            # Introduce action: take first prior_length steps of action sequence (consistent with Stage-1 order), accumulate saliency
            from utils.action_utils import ActionTokenizer, build_action_tensor
            tok = ActionTokenizer()
            action_seq = build_action_tensor()                         # [T,2] continuous view differences (float)
            actions_ids = tok.encode_sequence(action_seq, batch_size=1, device=device)  # [1,T,2] (long)

            K_steps = int(min(prior_length, actions_ids.size(1)))
            sal_acc = 0.0
            for t in range(K_steps):
                action_t = actions_ids[:, t, :]                        # [1,2]
                tokens_all = jepa.encoder(x_prev, action_t)             # [1,1+L,D]
                img_tok = tokens_all[:, 1:, :]                          # [1,L,D]
                sal_t = img_tok.pow(2).sum(dim=-1).view(1, 1, h, w)     # [1,1,h,w]
                sal_acc = sal_acc + sal_t
                x_prev = img_tok.detach()                               # Align with Stage-1

            sal = F.interpolate(sal_acc, size=view_img.shape[1:], mode="bilinear", align_corners=False)[0, 0]
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        else:
            feat = conv(view_img[None].to(device))                     # [1,C,h,w]
            sal = feat.pow(2).sum(dim=1, keepdim=True)                 # [1,1,h,w]
            sal = F.interpolate(sal, size=view_img.shape[1:], mode="bilinear", align_corners=False)[0, 0]
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)

    # 3) Sample N pixel locations based on saliency
    H, W = view_img.shape[1:]
    p = (sal + 1e-6).flatten()
    p = p / p.sum()
    idx = torch.multinomial(p, num_points, replacement=True)
    ys = (idx // W).float()
    xs = (idx % W).float()

    # 4) Backproject to world coordinates (using moderate initial depth range)
    d = torch.empty(num_points, device=device).uniform_(1.0, 3.0)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) / fx * d
    Y = (ys - cy) / fy * d
    Z = d
    p_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)  # [N,4]
    # Note: data stores cam.matrix_world (cam->world)
    means_h = (view_cam.to(device) @ p_cam.t()).t()
    means = means_h[:, :3].contiguous()

    # 5) Prior initialization for other parameters (stable start)
    # quats: unit orientation
    quats = torch.zeros(num_points, 4, device=device)
    quats[:, 3] = 1.0
    # scales: small range, avoid starting with large patches
    scales = torch.full((num_points, 3), 0.05, device=device)  # constant is also fine
    # opacities: logit=0 → alpha≈0.5
    opacities = torch.zeros(num_points, device=device)
    # colors: use GT pixel colors with inverse-sigmoid as initial value (close to rendering domain)
    gt = view_img.permute(1, 2, 0).to(device)  # [H,W,3]
    rgb = gt[ys.long(), xs.long()].clamp(1e-3, 1 - 1e-3)  # [N,3]
    colors = torch.log(rgb / (1 - rgb))

    for t in (means, scales, colors, quats, opacities):
        t.requires_grad_(True)

    print("[ENC_PRIOR] Initialization completed using Encoder prior")
    return means, quats, scales, opacities, colors

def _rotmat3_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion (x,y,z,w), w at the end.
    Input: R [...,3,3]
    Output: q [...,4], format [x,y,z,w]
    """
    # Reference from classic implementation, with numerical clamping
    eps = 1e-8
    r00 = R[..., 0, 0]
    r11 = R[..., 1, 1]
    r22 = R[..., 2, 2]
    trace = r00 + r11 + r22

    qw = torch.empty_like(trace)
    qx = torch.empty_like(trace)
    qy = torch.empty_like(trace)
    qz = torch.empty_like(trace)

    cond = trace > 0
    S = torch.sqrt(torch.clamp(trace + 1.0, min=eps)) * 2.0  # S=4*qw
    qw_pos = 0.25 * S
    qx_pos = (R[..., 2, 1] - R[..., 1, 2]) / (S + eps)
    qy_pos = (R[..., 0, 2] - R[..., 2, 0]) / (S + eps)
    qz_pos = (R[..., 1, 0] - R[..., 0, 1]) / (S + eps)

    # Branch: r00 is the largest diagonal element
    cond1 = (r00 > r11) & (r00 > r22)
    S1 = torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=eps)) * 2.0  # S=4*qx
    qw_1 = (R[..., 2, 1] - R[..., 1, 2]) / (S1 + eps)
    qx_1 = 0.25 * S1
    qy_1 = (R[..., 0, 1] + R[..., 1, 0]) / (S1 + eps)
    qz_1 = (R[..., 0, 2] + R[..., 2, 0]) / (S1 + eps)

    # Branch: r11 largest
    cond2 = ~cond1 & (r11 > r22)
    S2 = torch.sqrt(torch.clamp(1.0 + r11 - r00 - r22, min=eps)) * 2.0  # S=4*qy
    qw_2 = (R[..., 0, 2] - R[..., 2, 0]) / (S2 + eps)
    qx_2 = (R[..., 0, 1] + R[..., 1, 0]) / (S2 + eps)
    qy_2 = 0.25 * S2
    qz_2 = (R[..., 1, 2] + R[..., 2, 1]) / (S2 + eps)

    # Branch: r22 largest
    S3 = torch.sqrt(torch.clamp(1.0 + r22 - r00 - r11, min=eps)) * 2.0  # S=4*qz
    qw_3 = (R[..., 1, 0] - R[..., 0, 1]) / (S3 + eps)
    qx_3 = (R[..., 0, 2] + R[..., 2, 0]) / (S3 + eps)
    qy_3 = (R[..., 1, 2] + R[..., 2, 1]) / (S3 + eps)
    qz_3 = 0.25 * S3

    # Combine branches
    # First trace>0 uses pos branch
    qw = torch.where(cond, qw_pos, qw)
    qx = torch.where(cond, qx_pos, qx)
    qy = torch.where(cond, qy_pos, qy)
    qz = torch.where(cond, qz_pos, qz)

    # trace<=0, then choose 1/2/3 branch based on largest diagonal
    qw = torch.where(~cond & cond1, qw_1, qw)
    qx = torch.where(~cond & cond1, qx_1, qx)
    qy = torch.where(~cond & cond1, qy_1, qy)
    qz = torch.where(~cond & cond1, qz_1, qz)

    qw = torch.where(~cond & cond2, qw_2, qw)
    qx = torch.where(~cond & cond2, qx_2, qx)
    qy = torch.where(~cond & cond2, qy_2, qy)
    qz = torch.where(~cond & cond2, qz_2, qz)

    qw = torch.where(~cond & ~cond1 & ~cond2, qw_3, qw)
    qx = torch.where(~cond & ~cond1 & ~cond2, qx_3, qx)
    qy = torch.where(~cond & ~cond1 & ~cond2, qy_3, qy)
    qz = torch.where(~cond & ~cond1 & ~cond2, qz_3, qz)

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    # Normalize
    q = q / (q.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    return q

def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiplication (left multiply): q = q1 ⊗ q2, input/output format both [x, y, z, w].
    Supports broadcasting: q1 [...,4], q2 [...,4] → q [...,4]
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    # v = (x, y, z), w is scalar (last dimension)
    x = w1 * x2 + w2 * x1 + y1 * z2 - z1 * y2
    y = w1 * y2 + w2 * y1 + z1 * x2 - x1 * z2
    z = w1 * z2 + w2 * z1 + x1 * y2 - y1 * x2
    w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    q = torch.stack([x, y, z, w], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    return q

def _render_views_grid(means: torch.Tensor,
                       quats: torch.Tensor,
                       scales: torch.Tensor,
                       opacities: torch.Tensor,
                       colors: torch.Tensor,
                       cams: torch.Tensor,   # [T,4,4] cam->world
                       K: torch.Tensor,      # [3,3]
                       view_ids: list,
                       imgs: torch.Tensor,   # [T,3,H,W] can be None (no GT)
                       save_path: str,
                       device: torch.device):
    """Stitch GT and Pred for multiple views into a grid and save. Pred first, GT second, two columns per group."""
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
    Small test (single view):
      1. Take one image + camera from ShapeNetDataset
      2. Use "random initialization" or "Encoder prior initialization" for a bunch of Gaussians
      3. Render → MSE fit to GT
      4. Save GT vs Rendered comparison image to save_path
    """
    device = torch.device(device)

    # 1. Take one sample: imgs: [B, T, 3, H, W], cams: [B, T, 4, 4]
    ds = ShapeNetDataset(root=root,
                         split=split,
                         return_cam=True,
                         return_obj=True,  # Load object matrix for pose compensation
                         max_objs_per_synset=1,
                         synsets=["02958343"])  # Cars only
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    # New return format: images, cams, objs, meta
    batch = next(iter(loader))
    if len(batch) == 4:
        imgs, cams, objs, meta = batch
    else:
        # Compatible with old return (degrades to identity when no obj matrix)
        imgs, cams, meta = batch
        T_tmp = cams.shape[0]
        objs = torch.stack([torch.eye(4) for _ in range(T_tmp)], dim=0)

    # Remove batch dimension
    imgs = imgs[0]   # [T, 3, H, W]
    cams = cams[0]   # [T, 4, 4]
    objs = objs[0]   # [T, 4, 4]

    # Select a view t = 0 for prior initialization (training will use multi-view supervision)
    view_img = imgs[0].to(device)   # [3, H, W]
    view_cam = cams[0].to(device)   # [4, 4] (Note: cam->world)
    _, H, W = view_img.shape

    # ground truth: [H, W, 3]
    gt_image = view_img.permute(1, 2, 0)  # [H, W, 3]

    print(f"[INFO] view_img.shape = {view_img.shape}")
    print(f"[INFO] view_cam.shape = {view_cam.shape}")
    print(f"[INFO] meta = {meta}")

    # 2. Camera intrinsics: revert to FOV-based form (keep consistent with original)
    K = make_intrinsics_from_fov(H, W, device=device)

    # 3. Initialize Gaussians (prioritize Encoder prior, fallback to random on failure)
    if use_encoder_prior:
        try:
            means, quats, scales, opacities, colors = _init_gaussians_with_encoder_prior(
                view_img, view_cam, K, num_points, device, prior_length=4
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize with Encoder prior, using random initialization. Reason: {e}")
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

    # Inference path: only optimize on single view t0 for several steps, then render multi-view
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

        # Render multi-view: default select [0, 8] or [0, 1]
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

    # Training objective: based on one input (t0), generate two views (t0 and t1), and supervise with both views
    for it in range(iters):
        optimizer.zero_grad()

        # View set consistent with encoder prior: t=0 and t=1 (degrades to single view if T==1)
        T_all = imgs.shape[0]
        idx_list = [0] if T_all == 1 else list(range(min(4, T_all)))

        photo = 0.0
        per_view_losses = []

        # Object pose compensation: transform points learned in t0 pose to target view t's object pose
        obj0 = objs[0].to(device)
        obj0_inv = torch.linalg.inv(obj0)

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

            # quats_t = q_delta ⊗ quats (quaternion left multiply), rotate ellipsoid orientation with object
            R_delta = delta[:3, :3].unsqueeze(0)       # [1,3,3]
            q_delta = _rotmat3_to_quat_xyzw(R_delta)[0]  # [4]
            q_delta_expand = q_delta.unsqueeze(0).expand_as(quats)  # [N,4]
            quats_t = _quat_mul_xyzw(q_delta_expand, quats)

            pred_t, _, _ = render_gaussians_single_cam(
                means=means_t,
                quats=quats_t,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmat=viewmat_t,
                K=K,
                H=H,
                W=W,
            )
            l_t = mse_loss(pred_t, gt_image_t)
            photo = photo + l_t
            per_view_losses.append(l_t.detach())

        loss = photo / float(len(idx_list))
        loss.backward()
        optimizer.step()

        if it % 20 == 0 or it == iters - 1:
            if len(per_view_losses) > 0:
                pv = torch.stack(per_view_losses)
                print(f"[ITER {it:03d}] views={idx_list} Photo={loss.item():.4f} per-view[min={pv.min().item():.4f} max={pv.max().item():.4f} mean={pv.mean().item():.4f}]")
            else:
                print(f"[ITER {it:03d}] loss={loss.item():.4f}")

    # 4. Save comparison image (GT | PRED)
    # Save two views (or single view) GT|Pred stitched image, action/order consistent with prior
    with torch.no_grad():
        tiles = []
        out_views = [0] if imgs.shape[0] == 1 else list(range(min(4, imgs.shape[0])))
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

            R_delta = delta[:3, :3].unsqueeze(0)
            q_delta = _rotmat3_to_quat_xyzw(R_delta)[0]
            quats_t = _quat_mul_xyzw(q_delta.unsqueeze(0).expand_as(quats), quats)

            pred_t, _, _ = render_gaussians_single_cam(
                means=means_t,
                quats=quats_t,
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
        split="train",
        num_points=40960,
        iters=20000,
        lr=1e-3,
        device="cuda",
        save_path="./vis/3dgs_fit_debug_prior.png",
        use_encoder_prior=True,
        eval_only=False,
        eval_iters=300,
        render_view_ids=None,
        render_save_path="./vis/3dgs_infer_multiview.png",
    )