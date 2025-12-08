import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler

from model.JEPA import JEPAModel
from model.SVJ import SVJ
from utils.action_utils import ActionTokenizer, build_action_tensor
from utils.dataset_utils import ShapeNetDataset
from utils.gs_utils import make_intrinsics_from_fov, render_gaussians_batch


# wandb login
wandb.login(key="c607812d07dd287739ac6ae32c2be43cea6dc664")

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# JEPA configuration
hidden_size = 1024
head_dim = 128
head_num = 16
kv_head_num = 4
num_yaw = 2
num_pitch = 3
num_layers = 8

# define and load JEPA model
jepa = JEPAModel(
    hidden_size=hidden_size,
    head_dim=head_dim,
    head_num=head_num,
    kv_head_num=kv_head_num,
    num_yaw=num_yaw,
    num_pitch=num_pitch,
    num_layers=num_layers
)
jepa.load_state_dict(torch.load("../data/checkpoint/jepa_model_stage1.pth", map_location=torch.device(device)))
convnext = jepa.convnext
encoder = jepa.encoder
for p in convnext.parameters():
    p.requires_grad_(False)
for p in encoder.parameters():
    p.requires_grad_(False)
del jepa

# train setup
epoch = 10
batch_size = 64
lr = 1e-4

# dataset initialization
dataset = ShapeNetDataset(root="../data/3D", split="train", return_cam=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

# action initialization
action_tokenizer = ActionTokenizer()
action_sequence = build_action_tensor()
action_tensor = action_tokenizer.encode_sequence(action_sequence, batch_size, device=device)

# define svj
svj = SVJ(convnext, encoder, hidden_size, head_num, kv_head_num, head_dim).to(device)
svj.train()

# optimizer and scheduler initialization
optimizer = torch.optim.AdamW(svj.parameters(), lr=3e-4)
scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epoch * len(dataloader))

# wandb init
wandb.init(
    project="CV-JEPA-3DGS",
    name="jepa_run_002",
    config={
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "head_num": head_num,
        "kv_head_num": kv_head_num,
        "num_yaw": num_yaw,
        "num_pitch": num_pitch,
        "num_jepa_layers": num_layers,
        "num_svj_layers": 4,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
)

# training loop
global_step = 0
for _ in tqdm(range(epoch), leave=False):
    for batch in tqdm(dataloader, leave=False):
        imgs, cams, meta = batch
        imgs = imgs.to(device)
        cams = cams.to(device)
        B, T, C, H, W = imgs.shape

        means, quats, scales, opacities, colors = svj(imgs, action_tensor)

        viewmats = cams[:, 0]
        K_single = make_intrinsics_from_fov(H, W, device=device)
        Ks = K_single.unsqueeze(0).expand(B, -1, -1)

        render_colors, render_alphas, _ = render_gaussians_batch(
            means, quats, scales, opacities, colors,
            viewmats, Ks,
            H, W,
        )

        target = imgs[:, 0].permute(0, 2, 3, 1)

        loss = F.mse_loss(render_colors, target)

        optimizer.zero_grad()
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(svj.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        wandb.log(
            {
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
            },
            step=global_step,
        )

        if global_step == 0 or global_step % 10 == 0:
            b = 0

            target_img = imgs[b, 0]
            render_img = render_colors[b]
            render_img = render_img.permute(2, 0, 1)

            grid = torch.cat([
                target_img.unsqueeze(0),
                render_img.unsqueeze(0),
            ], dim=0)  # [2,3,H,W]

            grid = vutils.make_grid(grid, nrow=2)

            wandb.log({"gt_vs_render": wandb.Image(grid)}, step=global_step)

        global_step += 1

model_state = svj.state_dict()
torch.save(model_state, "../data/checkpoint/jepa_model_stage2.pth")

wandb.finish()