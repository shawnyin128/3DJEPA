import math
import torch
import torch.nn as nn

from model.Encoder import Encoder
from utils.action_utils import ActionTokenizer
from utils.dataset_utils import ShapeNetDataset


class EncoderPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int, proj_dim: int):
        super().__init__()
        # TODO: adaptiveAvgPool2d + 2layer MLP with SiLU
        self.hidden_size=hidden_size
        self.num_img_tokens=num_tokens
        self.proj_dim=proj_dim
        self.pool2d=nn.AdaptiveAvgPool2d((1,1))
        self.action_proj=nn.Linear(hidden_size,hidden_size)
        
        # Fusion: concat[img, action] -> linear -> norm
        self.fusion=nn.Sequential(
            nn.Linear(hidden_size*2,hidden_size),
            nn.LayerNorm(hidden_size),
        )
        
        # 2-layer MLP: predict image token delta
        self.mlp=nn.Sequential(
            nn.Linear(hidden_size,proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim,hidden_size*num_tokens),
        )

        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-4)

    def forward(self, tokens):
        #pass
        # TODO
        # tokens: [B, 2+num_tokens, C]
        # returns: [B, num_tokens, C]
        B, T, C=tokens.shape
        action_tokens=tokens[:,:2,:]
        img_tokens=tokens[:,2:,:]
        T_img=T-2
        
        assert T_img==self.num_img_tokens,\
            f"Expected {self.num_img_tokens} image tokens, got {T_img}"
        
        # Restore 2D structure: [B, T_img, C] -> [B, C, H, W]
        side=int(math.isqrt(T_img))
        assert side*side==T_img,\
            f"num_img_tokens={T_img} must be perfect square"
        img_feat=img_tokens.contiguous().view(B,side,side,C).permute(0,3,1,2)

        
        # 2D global pooling: [B, C, H, W] -> [B, C]
        pooled_img=self.pool2d(img_feat).view(B,C)
        
        # Action global feature
        action_global=action_tokens.mean(dim=1)
        action_global=self.action_proj(action_global)
        
        # Fuse image + action
        fused=torch.cat([pooled_img,action_global],dim=-1)
        fused=self.fusion(fused)
        
        # Predict delta and add to original
        delta_img=self.mlp(fused).view(B,T_img,C)
        pred_img_tokens=img_tokens+delta_img
        
        return pred_img_tokens

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