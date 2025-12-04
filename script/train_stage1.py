import wandb
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler

from model.JEPA import JEPAModel
from utils.action_utils import ActionTokenizer, build_action_tensor
from utils.dataset_utils import ShapeNetDataset


# wandb login
wandb.login(key="c607812d07dd287739ac6ae32c2be43cea6dc664")

# train configuration
hidden_size = 1024
head_dim = 128
head_num = 32
kv_head_num = 8
num_yaw = 2
num_pitch = 3
num_layers = 8
epoch = 20
batch_size = 16
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# wandb project initialization
wandb.init(
    project="CV-JEPA-3DGS",
    name="jepa_run_001",
    config={
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "head_num": head_num,
        "kv_head_num": kv_head_num,
        "num_yaw": num_yaw,
        "num_pitch": num_pitch,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
)

# model initialization
jepa = JEPAModel(
    hidden_size=hidden_size,
    head_dim=head_dim,
    head_num=head_num,
    kv_head_num=kv_head_num,
    num_yaw=num_yaw,
    num_pitch=num_pitch,
    num_layers=num_layers
).to(device)
jepa.train()
optimizer = torch.optim.AdamW(jepa.parameters(), lr=lr)

# dataset initialization
dataset = ShapeNetDataset(root="../data/3D", split="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epoch * len(dataloader))

# action initialization
action_tokenizer = ActionTokenizer()
action_sequence = build_action_tensor()
action_tensor = action_tokenizer.encode_sequence(action_sequence, batch_size, device=device)

# training loop
global_step = 0
for _ in tqdm(range(epoch), leave=False):
    for batch in tqdm(dataloader, leave=False):
        imgs, meta = batch
        imgs = imgs.to(device)

        loss, stats = jepa(imgs, action_tensor)

        optimizer.zero_grad()
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)
        optimizer.step()

        wandb.log({
            "loss": loss.item(),
            "sim": stats["sim"].item(),
            "var": stats["var"].item(),
            "cov": stats["cov"].item(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm.item(),
        }, step=global_step)

        scheduler.step()

        global_step += 1

        tqdm.write(f"loss={loss.item():.4f}  sim={stats['sim']:.4f}  var={stats['var']:.4f}")

model_state = jepa.state_dict()
torch.save(model_state, "../data/checkpoint/jepa_model_stage1.pth")

wandb.finish()
