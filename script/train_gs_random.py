import torch
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils.dataset_utils import ShapeNetDataset
from utils.gs_utils import make_intrinsics_from_fov, render_gaussians_single_cam, rotmat3_to_quat_xyzw, quat_mul_xyzw


# configuration
root = "../data/3D"
split = "test"
synsets = ["02958343"]
num_points = 40960
iters = 20000
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "./vis/test_car_fit_random.png"

# wandb login
wandb.login(key="c607812d07dd287739ac6ae32c2be43cea6dc664")

# wandb initialization
wandb.init(project="CV-JEPA-3DGS", name="3DGS_run_random", config={
    "split": split,
    "num_points": num_points,
    "iters": iters,
    "lr": lr,
    "use_prior": False
})

def _init_gaussians_random(view_img, view_cam, K, num_points, device):
    device = torch.device(device)
    H, W = view_img.shape[1], view_img.shape[2]
    
    # 均匀随机采样像素坐标
    ys = torch.rand(num_points, device=device) * H
    xs = torch.rand(num_points, device=device) * W
    
    d = torch.empty(num_points, device=device).uniform_(1.0, 3.0)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = torch.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d, torch.ones_like(d)], dim=-1)
    means = (view_cam.to(device) @ p_cam.t()).t()[:, :3].contiguous()
    
    quats = torch.tensor([0., 0., 0., 1.], device=device).repeat(num_points, 1)
    scales = torch.full((num_points, 3), 0.05, device=device)
    opacities = torch.zeros(num_points, device=device)
    
    rgb = view_img.permute(1, 2, 0).to(device)[ys.long().clamp(0, H-1), xs.long().clamp(0, W-1)].clamp(1e-3, 1-1e-3)
    colors = torch.log(rgb / (1 - rgb))
    
    for t in (means, scales, colors, quats, opacities):
        t.requires_grad_(True)
    return means, quats, scales, opacities, colors

# data initialization
device = torch.device(device)
ds = ShapeNetDataset(root=root, split=split, return_cam=True, return_obj=True, max_objs_per_synset=1, synsets=synsets)
imgs, cams, objs, _ = next(iter(DataLoader(ds, batch_size=1, shuffle=False)))
imgs, cams, objs = imgs[0], cams[0], objs[0]

H, W = imgs.shape[2], imgs.shape[3]
K = make_intrinsics_from_fov(H, W, device=device)

# gaussian initialization
means, quats, scales, opacities, colors = _init_gaussians_random(
    imgs[0], cams[0], K, num_points, device
)

optimizer = torch.optim.Adam([means, quats, scales, opacities, colors], lr=lr)
obj0_inv = torch.linalg.inv(objs[0].to(device))

# optimization loop
for it in range(iters):
    optimizer.zero_grad()
    loss = 0.0
    for t in range(min(4, imgs.shape[0])):
        target = imgs[t].permute(1, 2, 0).to(device)
        viewmat = torch.linalg.inv(cams[t].to(device))
        
        delta = objs[t].to(device) @ obj0_inv
        m_t = (delta @ torch.cat([means, torch.ones(num_points, 1, device=device)], dim=-1).t()).t()[:, :3]
        
        R_delta = delta[:3, :3].unsqueeze(0)
        q_delta = rotmat3_to_quat_xyzw(R_delta)[0]
        q_t = quat_mul_xyzw(q_delta.unsqueeze(0).expand_as(quats), quats)
        
        pred, _, _ = render_gaussians_single_cam(m_t, q_t, scales, opacities, colors, viewmat, K, H, W)
        loss += F.mse_loss(pred, target)
        
    loss /= min(4, imgs.shape[0])
    loss.backward()
    optimizer.step()
    
    wandb.log({"loss": loss.item()}, step=it)
    
    if it % 100 == 0:
        print(f"Iter {it:04d} | Loss: {loss.item():.6f}")

# visualization
with torch.no_grad():
    tiles = []
    for t in range(min(4, imgs.shape[0])):
        target = imgs[t].to(device)
        viewmat = torch.linalg.inv(cams[t].to(device))
        delta = objs[t].to(device) @ obj0_inv
        m_t = (delta @ torch.cat([means, torch.ones(num_points, 1, device=device)], dim=-1).t()).t()[:, :3]
        q_t = quat_mul_xyzw(rotmat3_to_quat_xyzw(delta[:3, :3].unsqueeze(0))[0].unsqueeze(0).expand_as(quats), quats)
        pred, _, _ = render_gaussians_single_cam(m_t, q_t, scales, opacities, colors, viewmat, K, H, W)
        tiles.extend([target.cpu(), pred.permute(2, 0, 1).clamp(0, 1).cpu()])
    save_image(make_grid(torch.stack(tiles), nrow=2), save_path)

wandb.finish()
