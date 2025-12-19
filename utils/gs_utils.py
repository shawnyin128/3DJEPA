import math
import torch
import torch.nn.functional as F
import gsplat


def make_intrinsics_from_fov(H: int,
                             W: int,
                             fov_x: float = math.pi / 2.0,
                             device: str = "cuda") -> torch.Tensor:
    """
    Create camera intrinsics matrix from field of view.
    Args:
        H, W: image height and width
        fov_x: horizontal field of view in radians
        device: torch device
    Returns:
        K: [3, 3] intrinsics matrix
    """
    device = torch.device(device)
    focal = 0.5 * float(W) / math.tan(0.5 * fov_x)
    K = torch.tensor(
        [
            [focal, 0.0,   W / 2.0],
            [0.0,   focal, H / 2.0],
            [0.0,   0.0,   1.0    ],
        ],
        device=device,
        dtype=torch.float32,
    )
    return K


def make_intrinsics_from_blender(H: int,
                                 W: int,
                                 device: str = "cuda",
                                 lens_mm: float = 55.0,
                                 sensor_width_mm: float = 36.0,
                                 sensor_height_mm: float = 24.0) -> torch.Tensor:
    """
    Calculate pixel intrinsics from Blender camera parameters:
    - fx = W * lens_mm / sensor_width_mm
    - fy = H * lens_mm / sensor_height_mm
    - cx = W/2, cy = H/2
    
    Default parameters match data generation script: lens=55mm, sensor=36×24mm.
    
    Args:
        H, W: image height and width
        device: torch device
        lens_mm: lens focal length in mm
        sensor_width_mm: sensor width in mm
        sensor_height_mm: sensor height in mm
    Returns:
        K: [3, 3] intrinsics matrix
    """
    device = torch.device(device)
    fx = float(W) * float(lens_mm) / float(sensor_width_mm)
    fy = float(H) * float(lens_mm) / float(sensor_height_mm)
    cx = float(W) / 2.0
    cy = float(H) / 2.0
    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    return K

# def render_gaussians_single_cam(means: torch.Tensor, # [N, 3]
#                                 quats: torch.Tensor, # [N, 4]
#                                 scales: torch.Tensor, # [N, 3]
#                                 opacities: torch.Tensor, # [N]
#                                 colors: torch.Tensor, # [N, 3] / [N, D]
#                                 viewmat: torch.Tensor, # [4, 4] world -> cam
#                                 K: torch.Tensor, # [3, 3]
#                                 H: int, W: int):
#     """
#     Render Gaussians from a single camera viewpoint.
    
#     NOTE: This function is defined in gs_utils.py but not used in the main training/inference pipeline.
#     The actual rendering implementation is in gs_new.py.
#     """
#     quats_normed = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

#     render_colors, render_alphas, meta = gsplat.rasterization(
#         means,
#         quats_normed,
#         scales,
#         torch.sigmoid(opacities),
#         torch.sigmoid(colors),
#         viewmat[None],
#         K[None],
#         W,
#         H,
#         packed=False
#     )

#     return render_colors[0], render_alphas[0], meta

# def render_gaussians_batch(
#         means, quats, scales_raw,  # scales_raw is unprocessed
#         opacity_logits, color_logits,
#         viewmats, Ks, H, W
# ):
#     """
#     Batch rendering of Gaussians for multiple views.
    
#     NOTE: This function is not currently used in the main pipeline.
    
#     Args:
#         means: [B, N, 3] Gaussian centers
#         quats: [B, N, 4] rotations
#         scales_raw: [B, N, 3] scales (already softplus output)
#         opacity_logits: [B, N] opacity logits
#         color_logits: [B, N, C] color logits
#         viewmats: [B, 4, 4] world->cam transforms
#         Ks: [B, 3, 3] intrinsics
#         H, W: image dimensions
#     Returns:
#         render_colors: [B, H, W, C]
#         render_alphas: [B, H, W, 1]
#         meta: rendering metadata
#     """
#     B, N, _ = means.shape

#     # Normalize quaternions
#     quats_normed = F.normalize(quats, dim=-1)

#     # Ensure scales are positive (scales_raw is already softplus output)
#     scales = scales_raw.clamp(min=1e-4, max=10.0)  # Add clamp to prevent extreme values

#     opacities = torch.sigmoid(opacity_logits)
#     colors = torch.sigmoid(color_logits)

#     viewmats = viewmats.unsqueeze(1)
#     Ks = Ks.unsqueeze(1)

#     render_colors, render_alphas, meta = gsplat.rasterization(
#         means,
#         quats_normed,
#         scales,
#         opacities,
#         colors,
#         viewmats,
#         Ks,
#         W, H,
#         packed=False
#     )
#     return render_colors[:, 0], render_alphas[:, 0], meta