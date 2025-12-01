import torch
import torch.nn as nn

from model.Encoder import Encoder
from utils.action_utils import ActionTokenizer
from utils.dataset_utils import ShapeNetDataset


class EncoderPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int, proj_dim: int):
        super().__init__()
        # TODO: adaptiveAvgPool2d + 2layer MLP with SiLU

    def forward(self, tokens):
        pass
        # TODO


class TDGSPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int, proj_dim: int):
        super().__init__()
        pass


if __name__ == '__main__':
    ds = ShapeNetDataset(root="../data/3D")
    imgs, meta = ds[0]
    imgs = imgs[0].unsqueeze(0)

    action = (0.0, 0.0)
    action_tokenizer = ActionTokenizer()
    yaw_id = action_tokenizer.yaw_to_id(action[0])
    pitch_id = action_tokenizer.pitch_to_id(action[1])

    action_tensor = torch.tensor([yaw_id, pitch_id], dtype=torch.long)

    model = Encoder(head_num=32,
                    kv_head_num=8,
                    head_dim=128,
                    hidden_size=1024,
                    num_yaw=8,
                    num_pitch=4,
                    num_layers=8)
    out = model(imgs, action_tensor) # [1, 66, 1024], batch = 1, tokens = 66, hidden size = 1024
    print(out.shape)