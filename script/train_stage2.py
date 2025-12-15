"""
Stage2训练 - 修复版
"""
import sys
import os

# RMSNorm补丁
import torch.nn as nn


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

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from model.JEPA import JEPAModel
from model.SVJ import SVJ
from utils.action_utils import ActionTokenizer, build_action_tensor
from utils.dataset_utils import ShapeNetDataset
from utils.gs_utils import make_intrinsics_from_fov, make_intrinsics_from_blender, render_gaussians_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("Stage2训练 - 修复版")
print("=" * 70)

# JEPA配置
hidden_size = 1024
head_dim = 128
head_num = 8
kv_head_num = 4
num_yaw = 2
num_pitch = 3
num_layers = 2

# 加载Stage1模型
print("\n[1] 加载Stage1模型...")
jepa = JEPAModel(
    hidden_size=hidden_size,
    head_dim=head_dim,
    head_num=head_num,
    kv_head_num=kv_head_num,
    num_yaw=num_yaw,
    num_pitch=num_pitch,
    num_layers=num_layers
)
jepa.load_state_dict(
    torch.load("/scratch/xy2053/course_project/cv-final/data/checkpoint/jepa_model_stage1_fixed.pth", map_location=device),
    strict=True
)
# 诊断性打印，确认与Stage1结构一致
print(f"  encoder depth: {len(jepa.encoder.blocks)}  (expect {num_layers})")
qw_shape = tuple(jepa.encoder.blocks[0].attn.q_proj.weight.shape)
ow_shape = tuple(jepa.encoder.blocks[0].attn.o_proj.weight.shape)
print(f"  q_proj.weight shape: {qw_shape}  o_proj.weight shape: {ow_shape}")
convnext = jepa.convnext
encoder = jepa.encoder
for p in convnext.parameters():
    p.requires_grad_(False)
for p in encoder.parameters():
    p.requires_grad_(False)
del jepa
print("  ✓ Stage1模型加载完成")

# 加载数据
print("\n[2] 加载数据...")
dataset = ShapeNetDataset(root="/scratch/xy2053/course_project/cv-final/data/3D", split="train", return_cam=True, synsets=["02958343"])
print(f"  数据集大小: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=True)
print("  ✓ DataLoader创建完成")

# 初始化SVJ
print("\n[3] 初始化SVJ...")
num_latent = 2048
svj = SVJ(convnext, encoder, hidden_size, head_num, kv_head_num, head_dim, num_latent=num_latent).to(device)
svj.train()

optimizer = torch.optim.AdamW(svj.parameters(), lr=1e-3, weight_decay=1e-4)
print("  ✓ SVJ初始化完成 (lr=1e-3, wd=1e-4, num_latent=2048)")

# Action
action_tokenizer = ActionTokenizer()
action_sequence = build_action_tensor()
action_tensor = action_tokenizer.encode_sequence(action_sequence, batch_size=16, device=device)

# 创建输出目录
os.makedirs('./vis', exist_ok=True)
print("\n[4] 训练1000步...")
print("=" * 70)

losses = []
data_iter = iter(dataloader)

for step in range(1000):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    imgs, cams, meta = batch
    imgs = imgs.to(device)
    cams = cams.to(device)
    B, T, C, H, W = imgs.shape

    # 参考相机（用第0帧）内参与外参（cam_to_world）
    K_single = make_intrinsics_from_blender(
        H, W, device=device, lens_mm=55.0, sensor_width_mm=36.0, sensor_height_mm=24.0
    )
    K_ref = K_single.unsqueeze(0).expand(B, -1, -1)
    cam_to_world_ref = cams[:, 0]  # 数据生成脚本保存的是 cam.matrix_world（cam->world）

    # 前向传播（生成一套全局 3DGS 参数；相机锚定反投影）
    means, quats, scales, opacities, colors = svj(imgs, action_tensor[:B], K_ref, cam_to_world_ref, H, W)

    # 多视角监督：随机抽取 V 个视角做联合监督
    V = min(3, T)
    idx = torch.randperm(T)[:V]
    L_photo = 0.0
    L_views = []  # 记录分视角损失，便于诊断
    for t in idx:
        t_int = int(t.item()) if hasattr(t, 'item') else int(t)
        # 训练/渲染阶段使用 world->cam，需要将 cam.matrix_world 取逆
        viewmats_t = torch.linalg.inv(cams[:, t_int])  # [B, 4, 4]
        # 使用与数据生成脚本一致的 Blender 相机内参
        Ks_t = K_ref

        render_c, render_a, _ = render_gaussians_batch(
            means, quats, scales, opacities, colors,
            viewmats_t, Ks_t, H, W
        )
        target_t = imgs[:, t_int].permute(0, 2, 3, 1)
        l_t = F.mse_loss(render_c, target_t)
        L_views.append(l_t.detach())
        L_photo = L_photo + l_t
    L_photo = L_photo / float(V)

    # 几何正则：鼓励点在相机前方（对被监督视角求均值）
    L_front = 0.0
    for t in idx:
        t_int = int(t.item()) if hasattr(t, 'item') else int(t)
        viewmats_t = torch.linalg.inv(cams[:, t_int])  # world->cam
        ones = torch.ones(B, means.size(1), 1, device=device)
        p_world_h = torch.cat([means, ones], dim=-1)  # [B, N, 4]
        p_cam = torch.einsum('bij,bnj->bni', viewmats_t, p_world_h)  # [B, N, 4]
        z = p_cam[..., 2]
        # margin=0.05，z小于margin则惩罚，softplus平滑
        L_front = L_front + F.softplus(0.05 - z).mean()
    L_front = L_front / float(V)

    # 稳定正则：opacity 稀疏 + scale 稳定
    alpha = torch.sigmoid(opacities)
    L_opacity = alpha.mean()  # encourage sparsity of alpha
    L_scale = (torch.log(scales + 1e-6) ** 2).mean()

    # 总损失
    # 关闭早期alpha稀疏项，防止可见性被压到0导致无梯度
    lambda_opacity = 0.0
    lambda_scale = 1e-3
    lambda_front = 1e-2
    loss = L_photo + lambda_opacity * L_opacity + lambda_scale * L_scale + lambda_front * L_front

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(svj.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())

    # 统计
    if step % 20 == 0:
        # 一次性打印相机内参（确认与数据一致）
        if step == 0:
            K_dbg = make_intrinsics_from_blender(H, W, device=device)
            print(
                f"K (Blender): fx={K_dbg[0,0].item():.2f}, fy={K_dbg[1,1].item():.2f}, "
                f"cx={K_dbg[0,2].item():.1f}, cy={K_dbg[1,2].item():.1f}"
            )
            print("[INFO] Using world->cam = inverse(cam.matrix_world) for rendering and front-loss.")

        print(f"Step {step + 1}/1000:")
        print(f"  Supervised views: {idx.tolist()}")
        print(f"  Loss: {loss.item():.4f} (Photo={L_photo.item():.4f}, Opacity={L_opacity.item():.4f}, Scale={L_scale.item():.4f}, Front={L_front.item():.4f})")
        # per-view 明细
        if len(L_views) > 0:
            lv = torch.stack(L_views)
            print(f"  per-view L_photo: min={lv.min().item():.4f} max={lv.max().item():.4f} mean={lv.mean().item():.4f}")

        # 基础统计
        print(f"  Means: std={means.std().item():.3f}")
        print(f"  Colors: std={torch.sigmoid(colors).std().item():.3f}")
        print(f"  Opacities: mean={alpha.mean().item():.3f}")

        # grad 与 lr
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  grad_norm={grad_norm.item():.3f}  lr={lr_now:.2e}")

        # scale/alpha 饱和度
        s_mean = scales.mean().item(); s_std = scales.std().item()
        s_min_sat = (scales <= 0.0105).float().mean().item()
        s_max_sat = (scales >= 4.99).float().mean().item()
        a = torch.sigmoid(opacities)
        a_min = (a < 0.01).float().mean().item(); a_max = (a > 0.99).float().mean().item()
        print(f"  scale: mean={s_mean:.3f} std={s_std:.3f} sat_min={s_min_sat:.3f} sat_max={s_max_sat:.3f}")
        print(f"  alpha: mean={a.mean().item():.3f} min%={a_min:.3f} max%={a_max:.3f}")

        # means 范数范围
        m_norm = means.norm(dim=-1)
        print(f"  means: |p| mean={m_norm.mean().item():.3f} max={m_norm.max().item():.3f}")

        # 几何可视化指标：前方点比例 + 投影覆盖度（以第一个被监督视角为准）
        try:
            t_log = int(idx[0].item()) if hasattr(idx[0], 'item') else int(idx[0])
            Vt = cams[:, t_log]  # [B,4,4]
            ones_h = torch.ones(B, means.size(1), 1, device=device)
            homog = torch.cat([means, ones_h], dim=-1)  # [B,N,4]
            p_cam = torch.einsum('bij,bnj->bni', Vt, homog)  # [B,N,4]
            z = p_cam[..., 2]
            front_ratio = (z > 0).float().mean().item()

            K_stat = make_intrinsics_from_blender(H, W, device=device)
            fx, fy, cx, cy = K_stat[0,0], K_stat[1,1], K_stat[0,2], K_stat[1,2]
            x = p_cam[..., 0] / (z + 1e-6)
            y = p_cam[..., 1] / (z + 1e-6)
            u = fx * x + cx
            v = fy * y + cy
            in_frame = ((u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)).float().mean().item()
            print(f"  geom: front_ratio={front_ratio:.3f}  in_frame={in_frame:.3f}")
        except Exception as _e:
            # 几何统计失败不影响训练
            pass

    # 每20步保存可视化
    if step % 20 == 0:
        fig, axes = plt.subplots(B, 3, figsize=(12, 3 * B))
        if B == 1:
            axes = axes.reshape(1, -1)

        # 选择一个固定视角做可视化（例如视角 0）
        t_vis = 0
        viewmats_vis = cams[:, t_vis]
        K_vis = make_intrinsics_from_blender(
            H, W, device=device, lens_mm=55.0, sensor_width_mm=36.0, sensor_height_mm=24.0
        ).unsqueeze(0).expand(B, -1, -1)
        render_vis_c, render_vis_a, _ = render_gaussians_batch(
            means, quats, scales, opacities, colors,
            viewmats_vis, K_vis, H, W
        )

        target_vis = imgs[:, t_vis].permute(0, 2, 3, 1)

        for b in range(B):
            gt = target_vis[b].cpu().numpy()
            axes[b, 0].imshow(gt)
            axes[b, 0].set_title(f'GT (mean={gt.mean():.3f})')
            axes[b, 0].axis('off')

            render = render_vis_c[b].detach().cpu().numpy()
            axes[b, 1].imshow(render)
            axes[b, 1].set_title(f'Render (mean={render.mean():.3f})')
            axes[b, 1].axis('off')

            alpha_img = render_vis_a[b].detach().cpu().numpy()
            im = axes[b, 2].imshow(alpha_img, cmap='gray', vmin=0, vmax=1)
            axes[b, 2].set_title(f'Alpha (mean={alpha_img.mean():.3f})')
            axes[b, 2].axis('off')
            plt.colorbar(im, ax=axes[b, 2])

        plt.tight_layout()
        plt.savefig(f'./vis/step{step + 1:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        if step % 20 == 0:
            print(f"  ✓ 保存: ./vis/step{step + 1:04d}.png")
            print()

print("=" * 70)
print("训练完成！")
print("=" * 70)

# 保存模型
model_path = "/scratch/xy2053/course_project/cv-final/data/checkpoint/stage2_fixed.pth"
torch.save(svj.state_dict(), model_path)
print(f"✓ 模型已保存到: {model_path}")