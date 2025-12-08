import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.JEPA import JEPAModel
from model.SVJ import SVJ
from model.Predictor import TDGSPredictor, TDGSParamHead
from utils.action_utils import ActionTokenizer, build_action_tensor
from utils.dataset_utils import ShapeNetDataset
from utils.gs_utils import make_intrinsics_from_fov

device = "cuda" if torch.cuda.is_available() else "cpu"


def render_gaussians_batch_fixed(
        means, quats, scales,
        opacity_logits, color_logits,
        viewmats, Ks, H, W
):
    """修正版的渲染函数，确保参数在合理范围"""
    import gsplat

    B, N, _ = means.shape

    # 处理参数
    opacities = torch.sigmoid(opacity_logits)
    colors = torch.sigmoid(color_logits)

    # 确保scales为正且在合理范围
    scales_clamped = scales.clamp(min=1e-4, max=10.0)

    # 确保四元数已归一化（TDGSParamHead应该已经做了）
    quats_normed = F.normalize(quats, dim=-1)

    viewmats = viewmats.unsqueeze(1) if viewmats.dim() == 3 else viewmats
    Ks = Ks.unsqueeze(1) if Ks.dim() == 3 else Ks

    render_colors, render_alphas, meta = gsplat.rasterization(
        means,
        quats_normed,
        scales_clamped,
        opacities,
        colors,
        viewmats,
        Ks,
        W, H,
        packed=False
    )
    return render_colors[:, 0], render_alphas[:, 0], meta


def check_gradients(model, name="model"):
    """检查模型梯度情况"""
    grad_info = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norm = p.grad.norm().item()
            grad_mean = p.grad.mean().item()
            grad_info[n] = {'norm': grad_norm, 'mean': grad_mean}
    return grad_info


def main():
    torch.manual_seed(42)
    torch.set_grad_enabled(True)

    print("=" * 60)
    print("SVJ with TDGSPredictor Training Test")
    print("=" * 60)

    # ============= 加载预训练的backbone =============
    hidden_size = 1024
    head_dim = 128
    head_num = 16
    kv_head_num = 4
    num_yaw = 2
    num_pitch = 3
    num_layers_jepa = 8

    jepa = JEPAModel(
        hidden_size=hidden_size,
        head_dim=head_dim,
        head_num=head_num,
        kv_head_num=kv_head_num,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
        num_layers=num_layers_jepa,
    )

    # 加载checkpoint
    ckpt_path = "../data/checkpoint/jepa_model_stage1.pth"
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    jepa.load_state_dict(ckpt)

    convnext = jepa.convnext.to(device)
    encoder = jepa.encoder.to(device)

    # 冻结backbone
    for p in convnext.parameters():
        p.requires_grad_(False)
    for p in encoder.parameters():
        p.requires_grad_(False)

    del jepa
    print("Backbone loaded and frozen")

    # ============= 准备数据 =============
    dataset = ShapeNetDataset(root="../data/3D", split="train", return_cam=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    imgs, cams, meta = next(iter(dataloader))
    imgs = imgs.to(device)  # [1, T, 3, H, W]
    cams = cams.to(device)  # [1, T, 4, 4]
    B, T, C, H, W = imgs.shape
    print(f"Data shape: imgs={imgs.shape}, cams={cams.shape}")

    # ============= 准备action序列 =============
    action_tokenizer = ActionTokenizer()
    action_sequence = build_action_tensor()
    prior_length = len(action_sequence)
    action_tensor = action_tokenizer.encode_sequence(
        action_sequence, batch_size=B, device=device
    )
    print(f"Actions shape: {action_tensor.shape}")

    # ============= 创建SVJ模型 =============
    num_layers_predictor = 2  # TDGSPredictor的层数
    num_latent = 128

    svj = SVJ(
        convnext=convnext,
        encoder=encoder,
        hidden_size=hidden_size,
        head_num=head_num,
        kv_head_num=kv_head_num,
        head_dim=head_dim,
        num_layer=num_layers_predictor,
        num_latent=num_latent,
        prior_length=prior_length,
    ).to(device)

    # 检查可训练参数
    trainable_params = []
    for name, param in svj.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            print(f"Trainable: {name:50s} {tuple(param.shape)}")

    total_trainable = sum(p.numel() for p in svj.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_trainable:,}")

    # ============= 优化器 =============
    optimizer = torch.optim.AdamW(
        [p for p in svj.parameters() if p.requires_grad],
        lr=1e-4,  # 较小的学习率
        weight_decay=1e-4
    )

    # ============= 准备渲染 =============
    target_img = imgs[0, 0].permute(1, 2, 0)  # [H, W, 3]
    viewmat = cams[0, 0]  # [4, 4]
    K = make_intrinsics_from_fov(H, W, device=device)

    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60)

    # ============= 训练循环 =============
    num_steps = 100

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass through SVJ
        means, quats, scales, opacity_logits, color_logits = svj(imgs, action_tensor)

        # 监控输出范围（每10步）
        if step % 10 == 0:
            with torch.no_grad():
                print(f"\n[Step {step:03d}] Output statistics:")
                print(f"  means:  [{means[0].min():.3f}, {means[0].max():.3f}] "
                      f"(mean={means[0].mean():.3f})")
                print(f"  scales: [{scales[0].min():.3f}, {scales[0].max():.3f}] "
                      f"(mean={scales[0].mean():.3f})")
                print(f"  opacity_logits: [{opacity_logits[0].min():.3f}, {opacity_logits[0].max():.3f}]")
                print(f"  color_logits:   [{color_logits[0].min():.3f}, {color_logits[0].max():.3f}]")

        # 渲染
        render_colors, render_alphas, _ = render_gaussians_batch_fixed(
            means,
            quats,
            scales,
            opacity_logits,
            color_logits,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            H=H, W=W,
        )

        pred_img = render_colors[0]  # [H, W, 3]

        # 计算损失
        rgb_loss = F.mse_loss(pred_img, target_img)

        # 防止opacity过小（鼓励一定的不透明度）
        opacity_penalty = -torch.mean(torch.sigmoid(opacity_logits))  # 负号使其最大化opacity
        opacity_target = torch.ones_like(opacity_logits) * 0.8  # 目标opacity
        opacity_loss = F.mse_loss(torch.sigmoid(opacity_logits), opacity_target) * 0.1

        # 防止scales过小
        scale_penalty = -torch.log(scales.mean() + 1e-8)  # 对数惩罚，防止scales过小

        # 组合损失
        loss = rgb_loss + 0.1 * opacity_loss + 0.01 * scale_penalty

        # Backward
        loss.backward()

        # 检查梯度
        if step % 10 == 0:
            grad_info = check_gradients(svj)

            # 计算总梯度norm
            total_grad_norm = sum(info['norm'] ** 2 for info in grad_info.values()) ** 0.5

            # 选几个关键参数查看
            key_params = [
                'predictor.latent',
                'predictor.head.out_proj.weight',
                'view_embed.weight'
            ]

            print(f"\n[Step {step:03d}] Gradient info:")
            print(f"  Total grad norm: {total_grad_norm:.6f}")
            for param_name in key_params:
                if param_name in grad_info:
                    info = grad_info[param_name]
                    print(f"  {param_name}: norm={info['norm']:.6f}, mean={info['mean']:.6e}")

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(svj.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 输出损失
        if step % 5 == 0:
            print(f"[Step {step:03d}] loss={loss.item():.6f}, "
                  f"rgb_loss={rgb_loss.item():.6f}, "
                  f"opacity_reg={opacity_penalty.item():.4f}, "
                  f"scale_reg={scale_penalty.item():.4f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # 最终检查
    with torch.no_grad():
        means, quats, scales, opacity_logits, color_logits = svj(imgs, action_tensor)
        print("\nFinal output statistics:")
        print(f"  means:  std={means[0].std():.3f}")
        print(f"  scales: mean={scales[0].mean():.3f}, std={scales[0].std():.3f}")
        print(f"  opacity (sigmoid): mean={torch.sigmoid(opacity_logits).mean():.3f}")
        print(f"  colors (sigmoid):  mean={torch.sigmoid(color_logits).mean():.3f}")


if __name__ == "__main__":
    main()
