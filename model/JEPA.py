import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ConvNeXt_Base_Weights
import copy

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Predictor import EncoderPredictor
from utils.loss_utils import byol_loss, variance_term, covariance_term, correlation_term, vic_loss


class JEPAModel(nn.Module):
    def __init__(self, hidden_size, head_dim, head_num, kv_head_num, num_yaw, num_pitch, num_layers, momentum: float = 0.998):
        super().__init__()
        self.convnext = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        for p in self.convnext.parameters():
            p.requires_grad_(False)

        self.encoder = Encoder(head_num=head_num,
                               kv_head_num=kv_head_num,
                               head_dim=head_dim,
                               hidden_size=hidden_size,
                               num_yaw=num_yaw,
                               num_pitch=num_pitch,
                               num_layers=num_layers)
        self.predictor = EncoderPredictor(hidden_size=hidden_size,
                                          proj_dim=hidden_size)
        self.decoder = Decoder()

        # 强化防塌缩：目标投影头 + EMA teacher（不反传）
        self.target_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self._init_projector(self.target_projector)
        self.momentum = momentum
        self.target_projector_ema = copy.deepcopy(self.target_projector)
        for p in self.target_projector_ema.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        return self

    def _init_projector(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _ema_update(self):
        m = self.momentum
        for p_ema, p in zip(self.target_projector_ema.parameters(), self.target_projector.parameters()):
            p_ema.data.mul_(m).add_(p.data, alpha=1 - m)

    def forward(self, images, actions):
        B, T, C, H, W = images.shape

        x0 = images[:, 0]
        with torch.no_grad():
            feat0 = self.convnext(x0)
        B_, C_, H_, W_ = feat0.shape
        x_prev = feat0.view(B_, C_, H_ * W_).transpose(1, 2)

        total_loss = 0.0
        sim_list = []
        # 为避免显存累积，这些统计用标量求和，不保留计算图
        v_online_sum = 0.0
        v_target_sum = 0.0
        std_online_sum = 0.0
        std_target_sum = 0.0
        cov_online_sum = 0.0
        cov_target_sum = 0.0
        pc1_ratio_sum = 0.0

        if self.training:
            self._ema_update()

        for t in range(1, T):
            action = actions[:, t-1, :]

            tokens_all = self.encoder(x_prev, action)
            action_tok = tokens_all[:, :1, :]
            img_tok = tokens_all[:, 1:, :]
            # 截断跨时间步的反向图，避免 BPTT 导致显存占用线性增长
            x_prev = img_tok.detach()
            z_online = self.predictor(tokens_all)
            # Encoder-only pooled 表征（仅图像 tokens），用于在 Encoder 上直接施加防塌缩约束
            enc_pooled = img_tok.mean(dim=1)  # [B, D]

            x_t = images[:, t]
            # 目标侧：Decoder + target projector（在线/EMA），仅对 EMA 反向阻断
            feat_t = self.decoder(x_t)  # [B, D]
            z_target_online = self.target_projector(feat_t)
            with torch.no_grad():
                z_target_ema = self.target_projector_ema(feat_t)

            # 主损失：VICReg 风格（在线分支 var+cov 正则，目标对齐 EMA）
            # 强化去相关，避免 cov_online 飙升；同时控制方差不过度增大
            loss_t, stats_t = vic_loss(
                z_online, z_target_ema,
                sim_coeff=25.0, var_coeff=25.0, cov_coeff=10.0,
                eps=1e-4, gamma=0.9, online_only_reg=True
            )
            # 在 Encoder 本身落防塌缩（无 sim，不触 GT）：方差 + 去相关（相关矩阵口径，尺度无关）
            loss_t = loss_t + 10.0 * variance_term(enc_pooled, eps=1e-4, gamma=0.9)
            loss_t = loss_t + 5e-3 * correlation_term(enc_pooled)
            total_loss = total_loss + loss_t

            sim_list.append(stats_t["sim"])

            # 额外统计用于日志与 collapse 监测（不参与主损，且在 no_grad 下避免显存累积）
            with torch.no_grad():
                v_online = variance_term(z_online, eps=1e-4, gamma=0.9).item()
                v_target = variance_term(z_target_ema, eps=1e-4, gamma=0.9).item()
                std_online = (z_online.var(dim=0, unbiased=False) + 1e-4).sqrt().mean().item()
                std_target = (z_target_ema.var(dim=0, unbiased=False) + 1e-4).sqrt().mean().item()
                Bz, Dz = z_online.shape
                cov_online = covariance_term(z_online, Bz, Dz).item()
                cov_target = covariance_term(z_target_ema, Bz, Dz).item()
                corr_online = correlation_term(z_online).item()
                corr_target = correlation_term(z_target_ema).item()

                # 低秩指标（在线分支）
                zo = z_online - z_online.mean(dim=0, keepdim=True)
                try:
                    S = torch.linalg.svdvals(zo)
                    pc1_ratio = ((S[0] ** 2) / (S.pow(2).sum() + 1e-12)).item()
                except RuntimeError:
                    pc1_ratio = 0.0

                v_online_sum += v_online
                v_target_sum += v_target
                std_online_sum += std_online
                std_target_sum += std_target
                cov_online_sum += cov_online
                cov_target_sum += cov_target
                # 累加相关矩阵离对角能量（用于日志）
                try:
                    corr_online_sum
                except NameError:
                    corr_online_sum = 0.0
                    corr_target_sum = 0.0
                corr_online_sum += corr_online
                corr_target_sum += corr_target
                pc1_ratio_sum += pc1_ratio

        num_steps = T - 1
        total_loss = total_loss / num_steps

        # 将标量均值打包为张量，便于统一 .item() 日志
        # 将标量均值打包为张量，便于统一 .item() 日志
        stats = {
            "loss": total_loss.detach(),
            "sim": torch.stack(sim_list).mean(),
            "v_online": torch.tensor(v_online_sum / max(num_steps, 1), device=images.device),
            "v_target": torch.tensor(v_target_sum / max(num_steps, 1), device=images.device),
            "std_online": torch.tensor(std_online_sum / max(num_steps, 1), device=images.device),
            "std_target": torch.tensor(std_target_sum / max(num_steps, 1), device=images.device),
            "cov_online": torch.tensor(cov_online_sum / max(num_steps, 1), device=images.device),
            "cov_target": torch.tensor(cov_target_sum / max(num_steps, 1), device=images.device),
            "pc1_ratio": torch.tensor(pc1_ratio_sum / max(num_steps, 1), device=images.device),
            "corr_online": torch.tensor(locals().get('corr_online_sum', 0.0) / max(num_steps, 1), device=images.device),
            "corr_target": torch.tensor(locals().get('corr_target_sum', 0.0) / max(num_steps, 1), device=images.device),
        }
        return total_loss, stats
