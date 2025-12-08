import torch
import torch.nn.functional as F


def variance_term(z: torch.Tensor, eps: float):
    var = z.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_term(z: torch.Tensor, B: int, D: int):
    z = z - z.mean(dim=0, keepdim=True)

    denom = max(B - 1, 1)
    cov = (z.T @ z) / denom

    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / D


def vic_loss(z1: torch.Tensor,
             z2: torch.Tensor,
             sim_coeff: float = 25.0,
             var_coeff: float = 25.0,
             cov_coeff: float = 1.0,
             eps: float = 1e-4):
    B, D = z1.shape

    sim_loss = F.mse_loss(z1, z2)
    var_loss = variance_term(z1, eps) + variance_term(z2, eps)
    cov_loss = covariance_term(z1, B, D) + covariance_term(z2, B, D)

    total = (
        sim_coeff * sim_loss
        + var_coeff * var_loss
        + cov_coeff * cov_loss
    )

    stats = {
        "sim": sim_loss.detach(),
        "var": var_loss.detach(),
        "cov": cov_loss.detach(),
    }
    return total, stats
