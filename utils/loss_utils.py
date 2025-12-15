import torch
import torch.nn.functional as F


def variance_term(z: torch.Tensor, eps: float, gamma: float = 1.0):
    """
    Encourage per-dimension std >= gamma (default 1.0).
    Using a slightly smaller gamma (e.g., 0.9) can stabilize training with small batches
    or when LayerNorm is applied before.
    """
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
    """
    Off-diagonal energy of the correlation matrix (batch-standardized features).
    对每个维度做批内标准化后再计算相关矩阵的离对角能量，
    相比协方差更稳定，数值尺度可控，适合作为去相关正则。
    """
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    z_n = z / (std + eps)
    B, D = z_n.shape
    denom = max(B - 1, 1)
    corr = (z_n.T @ z_n) / denom
    off_diag = corr - torch.diag(torch.diag(corr))
    return (off_diag ** 2).sum() / D


def barlow_correlation_loss(z: torch.Tensor,
                            eps: float = 1e-4,
                            lambda_offdiag: float = 1.0,
                            lambda_diag: float = 1.0):
    """
    Barlow Twins 风格的去相关正则：
    对标准化后的相关矩阵 corr，最小化 (corr - I)^2 的对角与离对角两部分。
    - lambda_offdiag: 离对角权重
    - lambda_diag: 对角(与 1 的偏差)权重
    返回：标量 loss
    """
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    z_n = z / (std + eps)
    B, D = z_n.shape
    denom = max(B - 1, 1)
    corr = (z_n.T @ z_n) / denom
    diag = torch.diag(corr)
    off = corr - torch.diag(diag)
    off_loss = (off ** 2).sum() / D
    diag_loss = ((diag - 1.0) ** 2).sum() / D
    return lambda_offdiag * off_loss + lambda_diag * diag_loss


def vic_loss(z1: torch.Tensor,
             z2: torch.Tensor,
             sim_coeff: float = 25.0,
             var_coeff: float = 75.0,
             cov_coeff: float = 3.0,
             eps: float = 1e-4,
             gamma: float = 1.0,
             online_only_reg: bool = True):
    """
    VICReg-style loss with configurable variance threshold and optional
    online-only regularization (recommended when target is stop-grad + normalized).
    """
    B, D = z1.shape

    sim_loss = F.mse_loss(z1, z2)

    # Online branch regularization (always)
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


def byol_loss(z1: torch.Tensor, z2: torch.Tensor):
    """
    简化版 BYOL 损失：对归一化后的向量做 MSE（等价于余弦距离的线性变换），
    仅返回 sim（越小越好）。
    适用于我们已采用 EMA teacher 的场景，作为最小稳定器，便于精简 Stage-1。
    """
    z1_n = torch.nn.functional.normalize(z1, dim=-1)
    z2_n = torch.nn.functional.normalize(z2, dim=-1)
    sim_loss = torch.nn.functional.mse_loss(z1_n, z2_n)
    stats = {"sim": sim_loss.detach()}
    return sim_loss, stats
