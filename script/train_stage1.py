import os
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

# Training configuration
hidden_size = 1024
head_dim = 128
head_num = 8
kv_head_num = 4
num_yaw = 2
num_pitch = 3
num_layers = 2
epoch = 10
batch_size = 64
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Wandb project initialization
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

"""
Support resuming from existing checkpoint:
- Priority: use resume file with optimizer/scheduler/steps (jepa_stage1_resume.pth)
- Fallback: load model weights only (jepa_model_stage1_fixed.pth)
"""

# Model & optimizer initialization
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

# Dataset initialization
dataset = ShapeNetDataset(root="../data/3D", split="train", synsets=["02958343"])
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epoch * len(dataloader))

# Checkpoint paths
ckpt_dir = os.path.join("..", "data", "checkpoint")
os.makedirs(ckpt_dir, exist_ok=True)
model_ckpt_path = os.path.join(ckpt_dir, "jepa_model_stage1_fixed.pth")
resume_ckpt_path = os.path.join(ckpt_dir, "jepa_stage1_resume.pth")

# Try to resume from checkpoint
start_epoch = 0
global_step = 0
if os.path.exists(resume_ckpt_path):
    state = torch.load(resume_ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        jepa.load_state_dict(state["model"], strict=False)
        if "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        if "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        print(f"[RESUME] Loaded training state from {resume_ckpt_path} (epoch={start_epoch}, global_step={global_step})")
elif os.path.exists(model_ckpt_path):
    try:
        weights = torch.load(model_ckpt_path, map_location=device)
        jepa.load_state_dict(weights, strict=False)
        print(f"[RESUME] Loaded model weights from {model_ckpt_path}")
    except Exception as e:
        print(f"[RESUME] Failed to load model weights: {e}")

# Action initialization
action_tokenizer = ActionTokenizer()
action_sequence = build_action_tensor()
action_tensor = action_tokenizer.encode_sequence(action_sequence, batch_size, device=device)

# Training loop
for ep in tqdm(range(start_epoch, epoch), leave=False):
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
            "sim": stats.get("sim", 0.0).item(),
            "v_online": stats.get("v_online", torch.tensor(0.0)).item(),
            "v_target": stats.get("v_target", torch.tensor(0.0)).item(),
            "std_online": stats.get("std_online", torch.tensor(0.0)).item(),
            "std_target": stats.get("std_target", torch.tensor(0.0)).item(),
            "cov_online": stats.get("cov_online", torch.tensor(0.0)).item(),
            "cov_target": stats.get("cov_target", torch.tensor(0.0)).item(),
            "pc1_ratio": stats.get("pc1_ratio", torch.tensor(0.0)).item(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm.item(),
        }, step=global_step)

        scheduler.step()

        global_step += 1

        tqdm.write(
            " ".join([
                f"loss={loss.item():.4f}",
                f"sim={stats.get('sim', torch.tensor(0.0)).item():.4f}",
                f"v_on={stats.get('v_online', torch.tensor(0.0)).item():.4f}",
                f"v_tg={stats.get('v_target', torch.tensor(0.0)).item():.4f}",
                f"std_on={stats.get('std_online', torch.tensor(0.0)).item():.4f}",
                f"std_tg={stats.get('std_target', torch.tensor(0.0)).item():.4f}",
                f"cov_on={stats.get('cov_online', torch.tensor(0.0)).item():.4f}",
                f"cov_tg={stats.get('cov_target', torch.tensor(0.0)).item():.4f}",
                f"pc1={stats.get('pc1_ratio', torch.tensor(0.0)).item():.4f}",
            ])
        )

    # Save checkpoint at end of each epoch
    save_state = {
        "model": jepa.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": ep + 1,
        "global_step": global_step,
        "config": dict(wandb.config) if wandb.run else {},
    }
    torch.save(save_state, resume_ckpt_path)
    torch.save(jepa.state_dict(), model_ckpt_path)

# Final save (redundant but explicit)
final_state = {
    "model": jepa.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "config": dict(wandb.config) if wandb.run else {},
}
torch.save(final_state, resume_ckpt_path)
torch.save(jepa.state_dict(), model_ckpt_path)

wandb.finish()