import torch
import torch.nn.functional as F


def variance_term(z: torch.Tensor, eps: float, gamma: float = 1.0):
    var = z.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_term(z: torch.Tensor, B: int, D: int):
    z = z - z.mean(dim=0, keepdim=True)

    denom = max(B - 1, 1)
    cov = (z.T @ z) / denom

    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / D


def correlation_term(z: torch.Tensor, eps: float = 1e-4):
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    z_n = z / (std + eps)
    B, D = z_n.shape
    denom = max(B - 1, 1)
    corr = (z_n.T @ z_n) / denom
    off_diag = corr - torch.diag(torch.diag(corr))
    return (off_diag ** 2).sum() / D


def vic_loss(z1: torch.Tensor,
             z2: torch.Tensor,
             sim_coeff: float = 25.0,
             var_coeff: float = 75.0,
             cov_coeff: float = 3.0,
             eps: float = 1e-4,
             gamma: float = 1.0,
             online_only_reg: bool = True):
    B, D = z1.shape

    sim_loss = F.mse_loss(z1, z2)

    var1 = variance_term(z1, eps, gamma)
    cov1 = covariance_term(z1, B, D)

    if online_only_reg:
        var_loss = var1
        cov_loss = cov1
    else:
        var2 = variance_term(z2, eps, gamma)
        cov2 = covariance_term(z2, B, D)
        var_loss = var1 + var2
        cov_loss = cov1 + cov2

    total = (
        sim_coeff * sim_loss
        + var_coeff * var_loss
        + cov_coeff * cov_loss
    )

    stats = {
        "sim": sim_loss.detach(),
        "var": (var_loss).detach(),
        "cov": (cov_loss).detach(),
    }
    return total, stats

