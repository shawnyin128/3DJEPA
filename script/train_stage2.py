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
from utils.gs_utils import make_intrinsics_from_fov, render_gaussians_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("Stage2训练 - 修复版")
print("=" * 70)

# JEPA配置
hidden_size = 1024
head_dim = 128
head_num = 16
kv_head_num = 4
num_yaw = 2
num_pitch = 3
num_layers = 8

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
jepa.load_state_dict(torch.load("/scratch/xy2053/course_project/cv-final/data/checkpoint/jepa_model_stage1.pth", map_location=device))
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
num_latent = 10240
svj = SVJ(convnext, encoder, hidden_size, head_num, kv_head_num, head_dim, num_latent=num_latent).to(device)
svj.train()

optimizer = torch.optim.AdamW(svj.parameters(), lr=1e-5)  # 更小的学习率
print("  ✓ SVJ初始化完成 (lr=1e-5)")

# Action
action_tokenizer = ActionTokenizer()
action_sequence = build_action_tensor()
action_tensor = action_tokenizer.encode_sequence(action_sequence, batch_size=16, device=device)

# 创建输出目录
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

    # 前向传播
    means, quats, scales, opacities, colors = svj(imgs, action_tensor[:B])

    # 渲染
    viewmats = cams[:, 0]
    K_single = make_intrinsics_from_fov(H, W, device=device)
    Ks = K_single.unsqueeze(0).expand(B, -1, -1)

    render_colors, render_alphas, _ = render_gaussians_batch(
        means, quats, scales, opacities, colors,
        viewmats, Ks, H, W,
    )

    target = imgs[:, 0].permute(0, 2, 3, 1)

    # 主loss
    mse_loss = F.mse_loss(render_colors, target)

    # 正则化：颜色多样性
    colors_sigmoid = torch.sigmoid(colors)
    color_mean_per_sample = colors_sigmoid.mean(dim=1, keepdim=True)
    color_variance = ((colors_sigmoid - color_mean_per_sample) ** 2).mean()
    diversity_loss = 1.0 / (color_variance + 1e-4)

    # 正则化：位置分散
    means_std = means.std(dim=1).mean()
    position_loss = 1.0 / (means_std + 1e-4)

    # 总loss
    loss = mse_loss + 0.1 * diversity_loss + 0.05 * position_loss

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(svj.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())

    # 统计
    if step % 20 == 0:
        render_mean = render_colors.mean().item()
        print(f"Step {step + 1}/1000:")
        print(
            f"  Loss: {loss.item():.4f} (MSE={mse_loss.item():.4f}, Div={diversity_loss.item():.4f}, Pos={position_loss.item():.4f})")
        print(f"  Render: mean={render_mean:.3f}")
        print(f"  Means: std={means.std().item():.3f}")
        print(f"  Colors: std={colors_sigmoid.std().item():.3f}")
        print(f"  Opacities: mean={torch.sigmoid(opacities).mean().item():.3f}")

    # 每20步保存可视化
    if step % 20 == 0:
        fig, axes = plt.subplots(B, 3, figsize=(12, 3 * B))
        if B == 1:
            axes = axes.reshape(1, -1)

        for b in range(B):
            gt = target[b].cpu().numpy()
            axes[b, 0].imshow(gt)
            axes[b, 0].set_title(f'GT (mean={gt.mean():.3f})')
            axes[b, 0].axis('off')

            render = render_colors[b].detach().cpu().numpy()
            axes[b, 1].imshow(render)
            axes[b, 1].set_title(f'Render (mean={render.mean():.3f})')
            axes[b, 1].axis('off')

            alpha = render_alphas[b].detach().cpu().numpy()
            im = axes[b, 2].imshow(alpha, cmap='gray', vmin=0, vmax=1)
            axes[b, 2].set_title(f'Alpha (mean={alpha.mean():.3f})')
            axes[b, 2].axis('off')
            plt.colorbar(im, ax=axes[b, 2])

        plt.tight_layout()
        plt.savefig(f'./vis/step{step + 1:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        if step % 20 == 0:
            print(f"  ✓ 保存: stage2_renders_fixed/step{step + 1:04d}.png")
            print()

print("=" * 70)
print("训练完成！")
print("=" * 70)

# 保存模型
model_path = "/scratch/xy2053/course_project/cv-final/data/checkpoint/stage2_fixed.pth"
torch.save(svj.state_dict(), model_path)
print(f"✓ 模型已保存到: {model_path}")