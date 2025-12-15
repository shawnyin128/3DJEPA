"""
Stage-2 Hybrid 训练脚本（写死配置，风格对齐 Stage-1）
- Encoder 先验 +（可选）短内循环直接优化 Δ
- 所有配置在本文件顶部常量中写死（无命令行参数）
"""

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# 兼容无 RMSNorm 的 PyTorch 版本
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


if not hasattr(nn, 'RMSNorm'):
    nn.RMSNorm = RMSNorm

from model.JEPA import JEPAModel
from model.EncoderPriorGS import EncoderPriorGS
from utils.dataset_utils import ShapeNetDataset
from utils.action_utils import ActionTokenizer, build_action_tensor
from utils.gs_utils import (
    make_intrinsics_from_blender,
    render_gaussians_batch,
)
from torchvision.utils import save_image


def inv_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


# =====================
# 写死的训练配置（对齐 Stage-1）
# =====================
DATA_ROOT = "../data/3D"
SPLIT = "train"
SYNSET = "02958343"
CKPT_PATH = "../data/checkpoint/jepa_model_stage1_fixed.pth"

BATCH_SIZE = 16
NUM_WORKERS = 8
ITERS = 1000

# Prior 与容量
PRIOR_LENGTH = 3
NUM_LATENT = 512

# 优化器
LR = 1e-3
WEIGHT_DECAY = 1e-4

# 短内循环（0 关闭）
INNER_REFINE_STEPS = 10

# 损失权重
LAMBDA_SCALE = 5e-3
LAMBDA_FRONT = 1e-2
LAMBDA_OPACITY = 3e-3
LAMBDA_DELTA = 1e-4

# 训练期不透明度温度基准（实际训练中按 step 做简单调度）
OPACITY_TEMP_BASE = 0.5

# 蒸馏对齐（仅在启用内循环时生效；0 关闭）
BETA_ALIGN = 1e-2

# 日志
SAVE_VIS_EVERY = 20


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('Stage-2 Hybrid 训练（Encoder 先验 + 可选内循环）')
    print('=' * 70)

    # 1) 加载 Stage-1 模型（与训练配置对齐）
    hidden_size = 1024
    head_dim = 128
    head_num = 8
    kv_head_num = 4
    num_yaw = 2
    num_pitch = 3
    num_layers = 2

    print('\n[1] 加载 Stage-1...')
    jepa = JEPAModel(
        hidden_size=hidden_size,
        head_dim=head_dim,
        head_num=head_num,
        kv_head_num=kv_head_num,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
        num_layers=num_layers,
    ).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    jepa.load_state_dict(state, strict=True)
    convnext = jepa.convnext
    encoder = jepa.encoder
    for p in convnext.parameters():
        p.requires_grad_(False)
    for p in encoder.parameters():
        p.requires_grad_(False)
    del jepa
    print('  ✓ 加载完成且冻结')

    # 2) 数据
    print('\n[2] 加载数据...')
    dataset = ShapeNetDataset(root=DATA_ROOT, split=SPLIT, return_cam=True, synsets=[SYNSET])
    print(f'  数据集大小: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
    print('  ✓ DataLoader 创建完毕')

    # 3) 模型（不依赖 SVJ）
    print('\n[3] 初始化 EncoderPriorGS...')
    model = EncoderPriorGS(
        convnext=convnext,
        encoder=encoder,
        hidden_size=hidden_size,
        num_latent=NUM_LATENT,
        prior_length=PRIOR_LENGTH,
        d_min=0.5,
        d_max=6.0,
        base_scale=0.10,
        opacity_bias=0.0,
    ).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f'  ✓ 初始化完成 (lr={LR:g}, wd={WEIGHT_DECAY:g}, num_latent={NUM_LATENT})')

    # 4) 动作编码器
    tokenizer = ActionTokenizer()
    action_seq = build_action_tensor()

    # 5) 输出目录
    os.makedirs('./vis', exist_ok=True)

    it = 0
    while it < ITERS:
        for batch in loader:
            it += 1
            if it > ITERS:
                break

            images, cams, meta = batch
            images = images.to(device)
            cams = cams.to(device)
            B, T, C, H, W = images.shape
            actions = tokenizer.encode_sequence(action_seq, B, device=device)

            # 参考相机（第 0 帧）：内参用 Blender 推导；外参 cam_to_world 直接用 matrix_world
            K_single = make_intrinsics_from_blender(H, W, device=device)
            K_ref = K_single.unsqueeze(0).expand(B, -1, -1)
            cam_to_world_ref = cams[:, 0]

            # 6) 前向：先验 → 像素参数 → 反投影
            means, quats, scales, opacities, colors, (uv, depth) = model(
                images, actions, K_ref, cam_to_world_ref, H, W
            )

            # 7) 多视角监督（V 个随机视角）
            V = min(3, T)
            idx = torch.randperm(T, device=device)[:V]

            # 简单 curriculum：前 300 步降低不透明度温度并关闭 alpha 稀疏，先确保可见性；之后再恢复
            if it <= 300:
                opac_temp = 0.2
                lambda_opacity_eff = 0.0
            else:
                opac_temp = OPACITY_TEMP_BASE  # 0.5
                lambda_opacity_eff = LAMBDA_OPACITY

            def render_with_params(means_, quats_, scales_, opacities_, colors_, t_indices):
                L_photo_local = 0.0
                per_view = []
                for t in t_indices:
                    t_int = int(t.item()) if hasattr(t, 'item') else int(t)
                    viewmats_t = torch.linalg.inv(cams[:, t_int])  # world->cam
                    Ks_t = K_ref
                    # 训练期对不透明度 logit 做温度缩放，抑制饱和（只在训练渲染用，保存可视化仍用原值）
                    opac_eff = opacities_ * opac_temp
                    pred_c, pred_a, _ = render_gaussians_batch(
                        means_, quats_, scales_, opac_eff, colors_,
                        viewmats_t, Ks_t, H, W
                    )
                    target_t = images[:, t_int].permute(0, 2, 3, 1)
                    l_t = F.mse_loss(pred_c, target_t)
                    per_view.append(l_t.detach())
                    L_photo_local = L_photo_local + l_t
                L_photo_local = L_photo_local / float(V)
                return L_photo_local, per_view

            # 8) 可选：短内循环直接优化（仅优化 uv/depth/scale/opacity 的小残差）
            use_inner = INNER_REFINE_STEPS > 0
            if use_inner:
                # 到 logit/对数域
                # 注意：所有“基值”都要从计算图分离，避免内循环的多次 backward 复用同一图
                uv_base = uv.detach()
                depth_base = depth.detach()
                scales_base = scales.detach()
                opacities_base = opacities.detach()
                quats_base = quats.detach()
                colors_base = colors.detach()

                u_logit = inv_sigmoid(uv_base.clamp(1e-4, 1.0 - 1e-4))
                d01 = ((depth_base - 1.0) / (4.0 - 1.0)).clamp(1e-4, 1.0 - 1e-4)
                d_logit = inv_sigmoid(d01)
                s_log = torch.log(scales_base.clamp(min=1e-4))
                o_logit = opacities_base  # 本身为 logit

                Δu = nn.Parameter(torch.zeros_like(u_logit))
                Δd = nn.Parameter(torch.zeros_like(d_logit))
                Δs = nn.Parameter(torch.zeros_like(s_log))
                Δo = nn.Parameter(torch.zeros_like(o_logit))
                opt_inner = torch.optim.Adam([Δu, Δd, Δs, Δo], lr=1e-2)

                for _ in range(INNER_REFINE_STEPS):
                    uv_r = torch.sigmoid(u_logit + Δu)
                    d01_r = torch.sigmoid(d_logit + Δd)
                    depth_r = 1.0 + (4.0 - 1.0) * d01_r
                    scales_r = (s_log + Δs).exp().clamp(min=1e-4, max=1.0)
                    opac_r = (o_logit + Δo)

                    # 反投影
                    means_r = model._unproject_to_world(uv_r, depth_r, K_ref, cam_to_world_ref, H, W)

                    # 使用已分离的 quats_base / colors_base，防止构建到外层图
                    L_photo_in, _ = render_with_params(means_r, quats_base, scales_r, opac_r, colors_base, idx)
                    L_delta = LAMBDA_DELTA * (
                        Δu.pow(2).mean() + Δd.pow(2).mean() + Δs.pow(2).mean() + Δo.pow(2).mean()
                    )
                    loss_in = L_photo_in + L_delta
                    opt_inner.zero_grad(); loss_in.backward(); opt_inner.step()

                # 内循环完成后，作为当步训练使用的参数
                uv_used = torch.sigmoid(u_logit + Δu).detach()
                d01_used = torch.sigmoid(d_logit + Δd).detach()
                depth_used = (1.0 + (4.0 - 1.0) * d01_used).detach()
                scales_used = (s_log + Δs).exp().clamp(min=1e-4, max=1.0).detach()
                opac_used = (o_logit + Δo).detach()
                means_used = model._unproject_to_world(uv_used, depth_used, K_ref, cam_to_world_ref, H, W).detach()
            else:
                uv_used, depth_used = uv, depth
                scales_used, opac_used = scales, opacities
                means_used = means

            # 9) 外层损失：基于预测头输出（确保梯度回传）
            L_photo, L_views = render_with_params(means, quats, scales, opacities, colors, idx)

            # 几何前方约束（对被监督视角求均值）——基于预测的 means
            L_front = 0.0
            for t in idx:
                t_int = int(t.item()) if hasattr(t, 'item') else int(t)
                viewmats_t = torch.linalg.inv(cams[:, t_int])
                ones = torch.ones(B, means.size(1), 1, device=device)
                p_world_h = torch.cat([means, ones], dim=-1)
                p_cam = torch.einsum('bij,bnj->bni', viewmats_t, p_world_h)
                z = p_cam[..., 2]
                L_front = L_front + F.softplus(0.05 - z).mean()
            L_front = L_front / float(V)

            alpha = torch.sigmoid(opacities)
            L_opacity = alpha.mean()
            L_scale = (torch.log(scales + 1e-6) ** 2).mean()

            loss = L_photo + LAMBDA_SCALE * L_scale + LAMBDA_FRONT * L_front + lambda_opacity_eff * L_opacity

            # 蒸馏：对齐 refined 与原预测（仅在启用内循环时）
            if use_inner and BETA_ALIGN > 0.0:
                loss = loss + BETA_ALIGN * (
                    F.smooth_l1_loss(uv, uv_used) +
                    F.smooth_l1_loss(depth, depth_used) +
                    F.smooth_l1_loss(torch.log(scales + 1e-6), torch.log(scales_used + 1e-6)) +
                    F.smooth_l1_loss(opacities, opac_used)
                )

            # 反向
            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            # 日志
            if it % SAVE_VIS_EVERY == 0 or it == 1:
                if it == 1:
                    K_dbg = make_intrinsics_from_blender(H, W, device=device)
                    print(
                        f"K(Blender): fx={K_dbg[0,0].item():.2f}, fy={K_dbg[1,1].item():.2f}, cx={K_dbg[0,2].item():.1f}, cy={K_dbg[1,2].item():.1f}"
                    )
                    print('[INFO] 渲染/前向一致使用 world->cam = inverse(cam.matrix_world)')

                per_view = torch.stack(L_views)
                print(f"Step {it}/{ITERS}:")
                print(f"  Supervised views: {idx.tolist()}")
                print(
                    f"  Loss: {loss.item():.4f} (Photo={L_photo.item():.4f}, Opacity={L_opacity.item():.4f}, "
                    f"Scale={L_scale.item():.4f}, Front={L_front.item():.4f})"
                )
                print(
                    f"  per-view L_photo: min={per_view.min().item():.4f} max={per_view.max().item():.4f} "
                    f"mean={per_view.mean().item():.4f}"
                )
                print(f"  grad_norm={float(grad_norm):.3f}  lr={optim.param_groups[0]['lr']:.2e}")

                # 简易几何与参数统计（基于预测头输出）
                s_mean, s_std = scales.mean().item(), scales.std().item()
                a_mean = alpha.mean().item()
                print(f"  scale: mean={s_mean:.3f} std={s_std:.3f}  alpha.mean={a_mean:.3f}")

                # 可视化保存（GT | Pred | Alpha）—— 仅保存 batch[0], 取本次监督的第一个视角
                try:
                    with torch.no_grad():
                        # 额外几何诊断：前方点比例/落入画面比例（基于预测头输出）
                        t_dbg = int(idx[0].item()) if hasattr(idx[0], 'item') else int(idx[0])
                        V_dbg = torch.linalg.inv(cams[:, t_dbg])  # world->cam
                        ones_dbg = torch.ones(B, means.size(1), 1, device=device)
                        p_world_h = torch.cat([means, ones_dbg], dim=-1)
                        p_cam_dbg = torch.einsum('bij,bnj->bni', V_dbg, p_world_h)  # [B,N,4]
                        z_dbg = p_cam_dbg[..., 2]
                        front_ratio = (z_dbg > 0).float().mean().item()
                        fx = K_ref[:, 0, 0].unsqueeze(1)
                        fy = K_ref[:, 1, 1].unsqueeze(1)
                        cx = K_ref[:, 0, 2].unsqueeze(1)
                        cy = K_ref[:, 1, 2].unsqueeze(1)
                        x_n = p_cam_dbg[..., 0] / (z_dbg + 1e-6)
                        y_n = p_cam_dbg[..., 1] / (z_dbg + 1e-6)
                        u = fx * x_n + cx
                        v = fy * y_n + cy
                        in_frame = ((u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_dbg > 0)).float().mean().item()
                        a = torch.sigmoid(opacities)
                        a_low = (a < 0.01).float().mean().item()
                        a_high = (a > 0.99).float().mean().item()
                        s_min = scales.min().item(); s_max = scales.max().item()
                        s_sat_min = (scales <= 0.0105).float().mean().item()
                        s_sat_max = (scales >= 0.145).float().mean().item()
                        print(
                            f"  geom: front_ratio={front_ratio:.3f} in_frame={in_frame:.3f}  "
                            f"alpha%(<0.01)={a_low:.3f} (>0.99)={a_high:.3f}  "
                            f"scale[min={s_min:.3f} max={s_max:.3f} sat_min%={s_sat_min:.3f} sat_max%={s_sat_max:.3f}]"
                        )

                        t_show = int(idx[0].item()) if hasattr(idx[0], 'item') else int(idx[0])
                        viewmats_show = torch.linalg.inv(cams[:, t_show])  # [B,4,4]
                        Ks_show = K_ref
                        pred_c_show, pred_a_show, _ = render_gaussians_batch(
                            means, quats, scales, opacities, colors,
                            viewmats_show, Ks_show, H, W
                        )
                        gt_show = images[:, t_show].permute(0, 2, 3, 1)

                        pred_chw = pred_c_show[0].permute(2, 0, 1).clamp(0.0, 1.0).cpu()  # [3,H,W]
                        gt_chw = gt_show[0].permute(2, 0, 1).clamp(0.0, 1.0).cpu()      # [3,H,W]

                        # 兼容 alpha 的形状：[H,W,1] 或 [H,W]
                        a = pred_a_show[0]
                        if a.dim() == 3 and a.size(-1) == 1:
                            a = a[..., 0]                 # [H,W]
                        elif a.dim() == 3 and a.size(-1) == 3:
                            a = a.mean(dim=-1)            # 兜底，[H,W]
                        # 转为 [3,H,W]
                        alpha_chw = a.unsqueeze(0).repeat(3, 1, 1).clamp(0.0, 1.0).cpu()

                        grid = torch.stack([gt_chw, pred_chw, alpha_chw], dim=0)
                        save_path = f'./vis/hybrid_step{it:04d}.png'
                        save_image(grid, save_path, nrow=3)
                        print(f"  ✓ 保存可视化: {save_path}")
                except Exception as e:
                    print(f"  [WARN] 保存可视化失败: {e}")

    print('\n✓ 训练完成')


if __name__ == '__main__':
    main()
