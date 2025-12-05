import torch
import torch.nn as nn
import torchvision
from torchvision.models import ConvNeXt_Base_Weights


class SVJ(nn.Module):
    def __init__(self, encoder, hidden_size, head_dim, head_num, kv_head_num, num_yaw, num_pitch, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.convnext = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        self.encoder = encoder
        self.view_embed = nn.Embedding(3, hidden_size)

    def forward(self, images, actions):
