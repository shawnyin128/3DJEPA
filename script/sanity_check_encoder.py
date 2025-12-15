import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model.JEPA import JEPAModel
from utils.dataset_utils import ShapeNetDataset
from utils.action_utils import ActionTokenizer, build_action_tensor
from utils.loss_utils import variance_term, covariance_term


@torch.no_grad()
def _corr_offdiag_metrics(z: torch.Tensor, eps: float = 1e-4):
    """
    返回两个与尺度无关的去相关指标：
    - corr_offdiag_mse: 相关矩阵离对角元素的均方（越小越好）
    - corr_offdiag_mean_abs: 相关矩阵离对角元素的平均绝对值（越小越好）
    实现：批内标准化每个维度后，计算相关矩阵 corr=(Z^T Z)/(B-1)。
    """
    B, D = z.shape
    if B < 2:
        return float('nan'), float('nan')
    z = z - z.mean(dim=0, keepdim=True)
    std = (z.var(dim=0, unbiased=False) + eps).sqrt()
    z_n = z / (std + eps)
    denom = max(B - 1, 1)
    corr = (z_n.T @ z_n) / denom  # [D,D]
    off_mask = ~torch.eye(D, dtype=torch.bool, device=corr.device)
    off_vals = corr[off_mask]
    corr_offdiag_mse = (off_vals.pow(2).mean()).item()
    corr_offdiag_mean_abs = (off_vals.abs().mean()).item()
    return corr_offdiag_mse, corr_offdiag_mean_abs


@torch.no_grad()
def compute_batch_metrics(jepa, images, actions, device, encoder_only=False):
    """
    计算一批数据上的方差/去相关等指标，衡量是否塌缩。
    返回：dict（各项标量的 Python float）
    """
    jepa.eval()

    B, T, C, H, W = images.shape

    # 初始帧特征（冻结 convnext）
    x0 = images[:, 0]
    feat0 = jepa.convnext(x0)
    B_, C_, H_, W_ = feat0.shape
    x_prev = feat0.view(B_, C_, H_ * W_).transpose(1, 2)

    v_online_list, v_target_list = [], []
    cov_online_list, cov_target_list = [], []
    corr_mse_online_list, corr_mse_target_list = [], []
    corr_abs_online_list, corr_abs_target_list = [], []
    std_online_list, std_target_list = [], []
    pc1_ratio_list = []

    for t in range(1, T):
        action = actions[:, t - 1, :]

        tokens_all = jepa.encoder(x_prev, action)
        x_prev = tokens_all[:, 1:, :]  # 下一步的图像 token

        if encoder_only:
            # 仅检查 Encoder：用图像 tokens 的均值作为 encoder 表征
            z_online = tokens_all[:, 1:, :].mean(dim=1)  # [B, D]
        else:
            z_online = jepa.predictor(tokens_all)  # [B, D]

        x_t = images[:, t]
        z_target = jepa.decoder(x_t)  # [B, D]

        # 方差惩罚项（VICReg 中用于检测塌缩）
        v_online = variance_term(z_online, eps=1e-4, gamma=GAMMA)
        v_target = variance_term(z_target, eps=1e-4, gamma=GAMMA)

        # 批内真实 std（越接近 1 越健康）
        std_online = (z_online.var(dim=0, unbiased=False) + 1e-4).sqrt().mean()
        std_target = (z_target.var(dim=0, unbiased=False) + 1e-4).sqrt().mean()

        # 去相关项（协方差的离对角能量，越小越好）
        Bz, Dz = z_online.shape
        cov_online = covariance_term(z_online, Bz, Dz)
        cov_target = covariance_term(z_target, Bz, Dz)

        # 相关矩阵离对角能量（尺度无关，更稳，越小越好）
        corr_mse_online, corr_abs_online = _corr_offdiag_metrics(z_online)
        corr_mse_target, corr_abs_target = _corr_offdiag_metrics(z_target)

        # 低秩检查：第一主成分能量占比（过高意味着低秩）
        zo = z_online - z_online.mean(dim=0, keepdim=True)
        # 使用 SVD 的奇异值做能量比
        try:
            S = torch.linalg.svdvals(zo)  # [min(B,D)]
            pc1_ratio = (S[0] ** 2) / (S.pow(2).sum() + 1e-12)
        except RuntimeError:
            pc1_ratio = torch.tensor(float('nan'), device=zo.device)

        v_online_list.append(v_online.item())
        v_target_list.append(v_target.item())
        std_online_list.append(std_online.item())
        std_target_list.append(std_target.item())
        cov_online_list.append(cov_online.item())
        cov_target_list.append(cov_target.item())
        corr_mse_online_list.append(corr_mse_online)
        corr_mse_target_list.append(corr_mse_target)
        corr_abs_online_list.append(corr_abs_online)
        corr_abs_target_list.append(corr_abs_target)
        pc1_ratio_list.append(pc1_ratio.item())

    # 聚合
    def _mean(x):
        return float(sum(x) / max(len(x), 1))

    return {
        "v_online": _mean(v_online_list),
        "v_target": _mean(v_target_list),
        "std_online": _mean(std_online_list),
        "std_target": _mean(std_target_list),
        "cov_online": _mean(cov_online_list),
        "cov_target": _mean(cov_target_list),
        "corr_mse_online": _mean(corr_mse_online_list),
        "corr_mse_target": _mean(corr_mse_target_list),
        "corr_abs_online": _mean(corr_abs_online_list),
        "corr_abs_target": _mean(corr_abs_target_list),
        "pc1_ratio": _mean(pc1_ratio_list),
    }


@torch.no_grad()
def action_sensitivity_check(jepa, images, num_yaw, num_pitch, device, encoder_only=False):
    """
    动作敏感性检测：固定同一输入图像 token，喂不同动作，观察预测表征的变化。
    返回：平均余弦相似度（越低越敏感）、最小/最大余弦相似度、方差。
    """
    jepa.eval()

    B, T, C, H, W = images.shape
    x0 = images[:, 0]
    feat0 = jepa.convnext(x0)
    B_, C_, H_, W_ = feat0.shape
    x_prev = feat0.view(B_, C_, H_ * W_).transpose(1, 2)

    # 取第一张样本，固定其图像 token
    x_prev_1 = x_prev[:1]  # [1, L, D]

    # 构造所有动作组合
    actions = []
    for y in range(num_yaw):
        for p in range(num_pitch):
            actions.append([y, p])
    actions = torch.tensor(actions, device=device, dtype=torch.long)  # [K, 2]
    K = actions.size(0)

    # 扩展图像 token 到 K
    x_prev_tile = x_prev_1.expand(K, -1, -1)

    tokens_all = jepa.encoder(x_prev_tile, actions)  # [K, 1+L, D]
    if encoder_only:
        z = tokens_all[:, 1:, :].mean(dim=1)        # [K, D]
    else:
        z = jepa.predictor(tokens_all)              # [K, D]

    # 计算 K×K 的余弦相似度矩阵
    z_n = F.normalize(z, dim=-1)
    sim = z_n @ z_n.t()  # [K, K]
    off_diag = sim[~torch.eye(K, dtype=torch.bool, device=device)]
    mean_off = off_diag.mean().item()
    min_off = off_diag.min().item()
    max_off = off_diag.max().item()
    var_z = z.var(dim=0, unbiased=False).mean().item()

    return {
        "mean_offdiag_cos": float(mean_off),
        "min_offdiag_cos": float(min_off),
        "max_offdiag_cos": float(max_off),
        "z_dim_var_mean": float(var_z),
        "K": int(K),
    }


def main():
    # 固定配置，按现有训练设置写死
    CHECKPOINT_PATH = "../data/checkpoint/jepa_model_stage1_fixed.pth"
    DATA_ROOT = "../data/3D"
    SYNSET = "02958343"
    SPLIT = "train"  # 用训练集做 sanity，样本更多、更稳定
    BATCH_SIZE = 8  # 测试集很小时避免 0 个 batch
    NUM_BATCHES = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ENCODER_ONLY = True  # 仅检查 encoder（推荐）。如要检查 predictor 输出，置为 False。

    # 模型超参需与训练时一致
    HIDDEN_SIZE = 1024
    HEAD_DIM = 128
    HEAD_NUM = 8
    KV_HEAD_NUM = 4
    NUM_YAW = 2
    NUM_PITCH = 3
    NUM_LAYERS = 4

    # 与训练对齐的 VICReg 方差门槛
    global GAMMA
    GAMMA = 0.9

    device = DEVICE

    # 构建模型 & 加载权重
    jepa = JEPAModel(
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        head_num=HEAD_NUM,
        kv_head_num=KV_HEAD_NUM,
        num_yaw=NUM_YAW,
        num_pitch=NUM_PITCH,
        num_layers=NUM_LAYERS,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    jepa.load_state_dict(ckpt, strict=False)
    jepa.eval()

    # 数据集 & 动作序列
    dataset = ShapeNetDataset(root=DATA_ROOT, split=SPLIT, synsets=[SYNSET])
    print(f"[INFO] dataset size = {len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

    action_tokenizer = ActionTokenizer()
    action_sequence = build_action_tensor()

    # 统计聚合
    agg = {
        "v_online": [], "v_target": [], "std_online": [], "std_target": [],
        "cov_online": [], "cov_target": [],
        "corr_mse_online": [], "corr_mse_target": [],
        "corr_abs_online": [], "corr_abs_target": [],
        "pc1_ratio": []
    }
    sens_report = None

    processed = 0
    for batch in loader:
        imgs, meta = batch
        imgs = imgs.to(device)
        B, T, C, H, W = imgs.shape

        actions = action_tokenizer.encode_sequence(action_sequence, B, device=device)

        metrics = compute_batch_metrics(jepa, imgs, actions, device, encoder_only=ENCODER_ONLY)
        for k in agg.keys():
            agg[k].append(metrics[k])

        # 第一批做动作敏感性检查
        if sens_report is None:
            sens_report = action_sensitivity_check(jepa, imgs, NUM_YAW, NUM_PITCH, device, encoder_only=ENCODER_ONLY)

        processed += 1
        if processed >= NUM_BATCHES:
            break

    def _mean(lst):
        return float(sum(lst) / max(len(lst), 1))

    print("==== Sanity Check (Encoder Collapse) ====")
    print(f"config: synset={SYNSET} split={SPLIT} batch_size={BATCH_SIZE} batches={processed} device={DEVICE} gamma={GAMMA} encoder_only={ENCODER_ONLY}")
    if processed == 0:
        print("[WARN] 没有任何 batch 被处理。请确认 split 下样本数 >= batch_size，或稍后再试 train split。")
        return
    print("-- Variance / Std --")
    print(f"v_online:  {_mean(agg['v_online']):.4f}   v_target:  {_mean(agg['v_target']):.4f}")
    print(f"std_online:{_mean(agg['std_online']):.4f}   std_target:{_mean(agg['std_target']):.4f}")
    print("-- De-correlation (Cov off-diag energy) --")
    print(f"cov_online:{_mean(agg['cov_online']):.4f}   cov_target:{_mean(agg['cov_target']):.4f}")
    print("-- Correlation (scale-invariant) --")
    print(f"corr_mse_online:{_mean(agg['corr_mse_online']):.6f}   corr_mse_target:{_mean(agg['corr_mse_target']):.6f}")
    print(f"corr_abs_online:{_mean(agg['corr_abs_online']):.6f}   corr_abs_target:{_mean(agg['corr_abs_target']):.6f}")
    print("-- Low-rank indicator --")
    print(f"pc1_ratio: {_mean(agg['pc1_ratio']):.4f}")

    if sens_report is not None:
        print("-- Action Sensitivity (first batch, all yaw×pitch) --")
        print(f"K={sens_report['K']}  mean_offdiag_cos={sens_report['mean_offdiag_cos']:.4f}  "
              f"min={sens_report['min_offdiag_cos']:.4f}  max={sens_report['max_offdiag_cos']:.4f}")
        print(f"z_dim_var_mean={sens_report['z_dim_var_mean']:.6f}" )

    # 风险提示（基于经验阈值）
    vo, so = _mean(agg['v_online']), _mean(agg['std_online'])
    co = _mean(agg['cov_online'])
    cro = _mean(agg['corr_mse_online'])
    p1 = _mean(agg['pc1_ratio'])
    warn = []
    if vo > 0.9 or so < 0.3:
        warn.append("[RISK] 在线方差偏低（可能塌缩趋势）：建议提高 variance 正则或增大 batch 多样性")
    if cro > 0.3:
        warn.append("[RISK] 在线相关性偏高：建议提高去相关正则（corr/cov）或增大 dropout")
    if p1 > 0.99:
        warn.append("[RISK] 在线低秩：pc1_ratio 过高，建议增强去相关/降低 variance 压力")
    if sens_report is not None and sens_report['mean_offdiag_cos'] > 0.9:
        warn.append("[RISK] 动作敏感性欠佳：offdiag 余弦≈1，建议前移动作条件或检查 action 覆盖")

    print()
    print("Tips to interpret:")
    print("- v_online/v_target -> 越接近 0 越好；在线 ≈1 表示接近塌缩；两者都≈1（总≈2）为典型塌缩。")
    print("- std_online/std_target -> 在线 0.3~0.8 较健康；长期 <0.3 且无回升，为塌缩风险。")
    print("- corr_mse_online/abs -> 越小越好（尺度无关），“mse”为离对角平方均值，“abs”为离对角绝对均值。")
    print("- cov_online -> 绝对值口径，受尺度影响大，仅作参考。")
    print("- pc1_ratio -> 若 >0.99 则强低秩风险；越低越好。")
    print("- Action Sensitivity -> offdiag 余弦越低越敏感；≈1 为功能性塌缩（对动作不敏感）。")
    if warn:
        print("\nWarnings:")
        for w in warn:
            print(w)


if __name__ == "__main__":
    main()
