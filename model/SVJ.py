import torch
import torch.nn as nn

from model.Predictor import TDGSPredictor


class SVJ(nn.Module):
    def __init__(self, convnext, encoder, hidden_size, head_num, kv_head_num, head_dim, num_layer=2, num_latent=128, prior_length = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.prior_length = prior_length

        self.convnext = convnext
        self.encoder = encoder
        self.view_embed = nn.Embedding(prior_length, hidden_size)
        nn.init.normal_(self.view_embed.weight, mean=0.0, std=0.02)

        self.predictor = TDGSPredictor(
            hidden_size=hidden_size,
            head_num=head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            num_layers=num_layer,
            num_latent=num_latent,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        self.encoder.eval()
        return self

    def forward(self, images, actions):
        B, T, C, H, W = images.shape

        x0 = images[:, 0]
        with torch.no_grad():
            feat0 = self.convnext(x0)
        B_, C_, H_, W_ = feat0.shape
        x_prev = feat0.view(B_, C_, H_ * W_).transpose(1, 2)

        view_tokens = []
        for t in range(self.prior_length):
            action = actions[:, t, :]
            tokens = self.encoder(x_prev, action)
            view_tokens.append(tokens)
            x_prev = tokens

        view_tokens = torch.stack(view_tokens, dim=1)

        view_idx = torch.arange(self.prior_length, device=images.device)
        view_emb = self.view_embed(view_idx)
        view_emb = view_emb.view(1, self.prior_length, 1, self.hidden_size)

        tokens = view_tokens + view_emb
        tokens = tokens.view(B, -1, self.hidden_size)

        means, quats, scales, opacities, colors = self.predictor(tokens)
        return means, quats, scales, opacities, colors
