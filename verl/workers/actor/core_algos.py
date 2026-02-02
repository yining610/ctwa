import torch
import math
import numpy as np
from collections import defaultdict


def compute_ctwa_stats(
    old_log_prob: torch.Tensor,  # (B, T)
    log_prob_new: torch.Tensor,  # (B, T)
    advantages: torch.Tensor,  # (B, T)
    response_mask: torch.Tensor,  # (B, T)
    per_objective_scores: torch.Tensor,  # (B, M)
    uids,  # np.ndarray | list (B,)
    cliprange_low: float,
    cliprange_high: float,
    w_agg: str = "mean",  # "mean" | "mean_unclipped",
    global_metrics: bool = False,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Compute CTWA statistics using the induced clipped advantage weight.

    Completion-level weight:
      - w_agg="mean":            w = mean_t W_t  (masked by response_mask)
      - w_agg="mean_unclipped":  w = sum_t W_t / (sum_t 1_t + eps)
    """

    # Sequence-level advantage proxy (B,)
    denom = response_mask.sum(dim=-1)
    A_hat = (advantages * response_mask).sum(dim=-1) / denom

    # Token-level importance ratio (B, T)
    neg_approx_kl = log_prob_new - old_log_prob
    neg_approx_kl = torch.clamp(neg_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(neg_approx_kl)

    # Unclipped indicator (B, T)
    a_pos = (A_hat >= 0).unsqueeze(-1)
    ind = torch.where(
        a_pos,
        ratio <= (1.0 + float(cliprange_high)),
        ratio >= (1.0 - float(cliprange_low)),
    )
    ind_f = ind.to(dtype=torch.float32) * response_mask

    # Token weights W_t (B, T)
    W_tok = A_hat.unsqueeze(-1) * ratio * ind_f

    # Completion-level weight (B,)
    if w_agg == "mean":
        w = W_tok.sum(dim=-1) / denom
    elif w_agg == "mean_unclipped":
        unclipped_cnt = ind_f.sum(dim=-1).clamp_min(0.0)
        w = W_tok.sum(dim=-1) / (unclipped_cnt + eps)
    elif w_agg == "sum":
        w = W_tok.sum(dim=-1)
    else:
        raise ValueError(f"Invalid w_agg={w_agg}.")

    # Global clip fraction (for diagnostics)
    unclip_frac = float(ind_f.sum().item() / response_mask.sum().clamp_min(1.0).item())

    w_cpu_all = w.detach().to("cpu").numpy()  # (B,)
    u_cpu_all = per_objective_scores.detach().to("cpu").numpy()  # (B, M)
    uid_list_all = uids.tolist()

    # Simple outlier removal: drop samples with |w| > 100x median(|w|).
    finite_mask = np.isfinite(w_cpu_all)
    w_abs = np.abs(w_cpu_all[finite_mask])
    if w_abs.size > 0:
        med_abs = float(np.median(w_abs))
    else:
        med_abs = 0.0

    if med_abs > 0.0:
        keep_mask = finite_mask & (np.abs(w_cpu_all) <= (100.0 * med_abs))
    else:
        keep_mask = finite_mask

    keep_idxs = np.nonzero(keep_mask)[0]
    w_cpu = w_cpu_all[keep_idxs]
    u_cpu = u_cpu_all[keep_idxs, :]
    uid_list = [uid_list_all[i] for i in keep_idxs.tolist()]

    # Group by uid
    uid2idxs: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(uid_list):
        uid2idxs[str(uid)].append(i)

    M = u_cpu.shape[1]  # number of objectives

    # Per-uid averaging accumulators
    cov_sum = np.zeros((M,), dtype=np.float64)
    corr_sum = np.zeros((M,), dtype=np.float64)
    corr_cnt = np.zeros((M,), dtype=np.float64)

    # global accumulators
    cross_sum = np.zeros((M,), dtype=np.float64)
    u2_sum = np.zeros((M,), dtype=np.float64)
    w2_sum = 0.0
    n_total = 0
    n_groups = 0

    for idxs in uid2idxs.values():
        if len(idxs) < 2:
            continue
        wg = w_cpu[idxs]
        wc = wg - wg.mean()
        ug = u_cpu[idxs, :]  # (K, M)
        uc = ug - ug.mean(axis=0, keepdims=True)

        n_groups += 1

        if global_metrics:
            # Pooled over all completions, after within-uid centering.
            cross_sum += np.sum(uc * wc[:, None], axis=0)
            u2_sum += np.sum(uc * uc, axis=0)
            w2_sum += float(np.sum(wc * wc))
            n_total += len(idxs)
        else:
            # Average per-uid covariance/correlation across uids.
            cov = np.mean(uc * wc[:, None], axis=0)  # (M,)
            cov_sum += cov

            std_w = np.std(wc)
            std_u = np.std(uc, axis=0)
            denom_corr = std_u * std_w
            valid = denom_corr > eps
            if np.any(valid):
                corr = np.zeros_like(cov)
                corr[valid] = cov[valid] / denom_corr[valid]
                corr_sum += corr
                corr_cnt += valid.astype(np.float64)

    metrics: dict[str, float] = {
        "ctwa/w_mean": float(np.mean(w_cpu)) if w_cpu.size else 0.0,
        "ctwa/w_std": float(np.std(w_cpu)) if w_cpu.size else 0.0,
        "ctwa/unclip_frac": float(unclip_frac),
        "ctwa/n_groups": float(n_groups),
    }

    if global_metrics:
        metrics["ctwa/n_total"] = float(n_total)
        if n_total > 0:
            cov_mean = cross_sum / float(n_total)
            std_w_pool = math.sqrt(w2_sum / float(n_total))
            std_u_pool = np.sqrt(u2_sum / float(n_total))
            denom_corr = std_u_pool * std_w_pool
            corr_mean = np.zeros((M,), dtype=np.float64)
            valid = denom_corr > eps
            if np.any(valid):
                corr_mean[valid] = cov_mean[valid] / denom_corr[valid]
        else:
            cov_mean = np.zeros((M,), dtype=np.float64)
            corr_mean = np.zeros((M,), dtype=np.float64)
    else:
        if n_groups > 0:
            cov_mean = cov_sum / float(n_groups)
        else:
            cov_mean = np.zeros((M,), dtype=np.float64)

        # Correlation mean averaged over groups where denom valid
        corr_mean = np.zeros((M,), dtype=np.float64)
        if np.any(corr_cnt > 0):
            corr_mean = corr_sum / np.maximum(corr_cnt, 1.0)

    for j in range(M):
        metrics[f"ctwa/cov_mean/{j}"] = float(cov_mean[j])
        metrics[f"ctwa/corr_mean/{j}"] = float(corr_mean[j])

    return metrics

def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project vector v onto the probability simplex.
    
    Implementation based on the paper: Efficient Projections onto the l1-Ball for Learning in High Dimensions
    """
    device = v.device
    dtype = v.dtype
    n = v.numel()
    
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, n + 1, device=device, dtype=dtype)
    cond = u - cssv / ind > 0
    rho = torch.nonzero(cond, as_tuple=False).max() # find the pivot index
    theta = cssv[rho] / (rho.to(dtype) + 1)
    w = torch.clamp(v - theta, min=0)
    
    # TODO: check if we need to normalize w
    return w


def compute_mgda_weights(grad_mat: torch.Tensor) -> torch.Tensor:
    """Compute MGDA weights given per-objective flattened gradients."""
    
    obj_count = grad_mat.shape[0]
    device = grad_mat.device
    dtype = grad_mat.dtype
    if obj_count == 1:
        return torch.ones(1, device=device, dtype=dtype)
    if obj_count == 2:
        # analytical solution
        g1, g2 = grad_mat[0], grad_mat[1]
        d = g1 - g2
        denom = torch.dot(d, d).clamp_min(1e-12)
        w1 = torch.dot(g2, g2 - g1) / denom
        w1 = torch.clamp(w1, 0.0, 1.0)
        return torch.stack([w1, 1 - w1]).to(dtype=dtype)

    # General case: minimize f(w) = w^T H w over the simplex
    # we use projected gradient descent with a fixed step size
    H = grad_mat @ grad_mat.T  # (S, S), positive semidefinite
    try:
        eigvals = torch.linalg.eigvals(H).real
        # we choose the largest eigenvalue as the Lipschitz constant for better convergence
        L = torch.clamp(eigvals.max(), min=1e-6).item()
    except Exception:
        L = max(H.diag().max().item(), 1e-3)
    step = 1.0 / L
    w = torch.full((obj_count,), 1.0 / obj_count, device=device, dtype=dtype)
    max_iter = 200
    tol = 1e-6

    for _ in range(max_iter):
        w_prev = w
        grad = 2 * (H @ w)  # gradient of f(w)
        w = project_to_simplex(w - step * grad)
        if torch.norm(w - w_prev, p=2).item() < tol:
            break

    if not torch.isfinite(w).all():
        w = torch.full((obj_count,), 1.0 / obj_count, device=device, dtype=dtype)
    return w