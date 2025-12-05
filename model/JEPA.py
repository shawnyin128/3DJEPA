import torch
import torch.nn as nn
import torchvision
from torchvision.models import ConvNeXt_Base_Weights

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Predictor import EncoderPredictor
from utils.loss_utils import vic_loss


class JEPAModel(nn.Module):
    def __init__(self, hidden_size, head_dim, head_num, kv_head_num, num_yaw, num_pitch, num_layers):
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
                                          proj_dim=hidden_size // 2)
        self.decoder = Decoder(hidden_size=hidden_size,
                               proj_dim=hidden_size // 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        return self

    def forward(self, images, actions):
        B, T, C, H, W = images.shape

        x0 = images[:, 0]
        with torch.no_grad():
            feat0 = self.convnext(x0)
        B_, C_, H_, W_ = feat0.shape
        x_prev = feat0.view(B_, C_, H_ * W_).transpose(1, 2)

        total_loss = 0.0
        sim_list, var_list, cov_list = [], [], []

        for t in range(1, T):
            action = actions[:, t-1, :]

            tokens = self.encoder(x_prev, action)
            x_prev = tokens
            z_online = self.predictor(tokens)

            x_t = images[:, t]
            z_target = self.decoder(x_t)

            loss_t, stats_t = vic_loss(z_online, z_target)
            total_loss = total_loss + loss_t

            sim_list.append(stats_t["sim"])
            var_list.append(stats_t["var"])
            cov_list.append(stats_t["cov"])

        num_steps = T - 1
        total_loss = total_loss / num_steps

        stats = {
            "loss": total_loss.detach(),
            "sim": torch.stack(sim_list).mean(),
            "var": torch.stack(var_list).mean(),
            "cov": torch.stack(cov_list).mean(),
        }
        return total_loss, stats
