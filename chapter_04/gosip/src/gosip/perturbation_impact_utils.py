import numpy as np
import torch
from IPython.display import clear_output
import scipy.stats as stats
from .neural_network_utils import calculate_category_total_accuracy
import torch.distributions as dist


import torch
import numpy as np
import scipy.stats as stats   # ← requires  “pip/conda install scipy”



def calculate_valid_ratio(tensor1, tensor2, epsilon: float = 1e-12, lambda_param: float = 0.0):
    """
    Compute a similarity score scaled to [-1, 1] for two distance tensors/arrays:

        score = (tensor1 − tensor2) / (tensor1 + tensor2)

    An optional exponential penalty exp(−λ · tensor1) can be applied.
    A small positive `epsilon` is used to avoid divide‑by‑zero.

    Parameters
    ----------
    tensor1, tensor2 : torch.Tensor | np.ndarray
        Element‑wise distances of identical shape.
    epsilon : float, default 1e‑12
        Minimum value to clamp each element and the denominator.
    lambda_param : float, default 0.0
        Weight of the exponential penalty. 0 disables the penalty.

    Returns
    -------
    torch.Tensor | np.ndarray
        Element‑wise scores in the same type as the inputs.
    """
    # ── PyTorch branch ──────────────────────────────────────────────────────────
    if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
        if tensor1.shape != tensor2.shape:
            raise ValueError("tensor1 and tensor2 must have the same shape.")
        if tensor1.device != tensor2.device or tensor1.dtype != tensor2.dtype:
            raise ValueError("tensor1 and tensor2 must share device and dtype.")

        # Clamp to avoid zeros / negatives
        t1_safe = tensor1.clamp(min=epsilon)
        t2_safe = tensor2.clamp(min=epsilon)

        denom   = (t1_safe + t2_safe).clamp(min=epsilon)
        scores  = (t1_safe - t2_safe) / denom          # in [-1, 1]

        if lambda_param != 0.0:
            scores = scores / torch.exp(-lambda_param * t1_safe)

        return scores

    # ── NumPy branch ───────────────────────────────────────────────────────────
    elif isinstance(tensor1, np.ndarray) and isinstance(tensor2, np.ndarray):
        if tensor1.shape != tensor2.shape:
            raise ValueError("tensor1 and tensor2 must have the same shape.")

        t1_safe = np.clip(tensor1, epsilon, None)
        t2_safe = np.clip(tensor2, epsilon, None)

        denom   = np.clip(t1_safe + t2_safe, epsilon, None)
        scores  = (t1_safe - t2_safe) / denom          # in [-1, 1]

        if lambda_param != 0.0:
            scores = scores / np.exp(-lambda_param * t1_safe)

        return scores

    # ── Type error ─────────────────────────────────────────────────────────────
    else:
        raise TypeError("Both inputs must be either PyTorch tensors or NumPy arrays.")




    
def bh_tstat_torch_precomputed(t_stats: torch.Tensor, crit: torch.Tensor) -> torch.BoolTensor:
    flat = t_stats.flatten()
    M = flat.numel()
    device = flat.device

    abs_flat = flat.abs()
    order = torch.argsort(abs_flat, descending=True)
    t_desc = abs_flat[order]

    hits = t_desc >= crit
    keep_flat = torch.zeros(M, dtype=torch.bool, device=device)
    if hits.any():
        last = torch.nonzero(hits, as_tuple=False).max()
        keep_flat[order[: last.item() + 1 ]] = True

    return keep_flat.view(t_stats.shape)
    


def full_perturbation_analysis(
    start_tensor: torch.Tensor,
    transition_label_tensor: torch.Tensor,
    stargan,
    oracle,
    propagator,
    num_categories: list[int],
    *,
    lambda_param: float = 0.0,
    propagation: bool = True,
    network_fdr_pval_threshold=0.10
):
    device = stargan.device
    batch  = start_tensor.size(0)
    d      = start_tensor.size(1)

    # ── allocate on device ─────────────────────────────────────────────── 🆕
    violins_t   = torch.empty((d, batch), device=device)   # ← rows = features, cols = samples
    acc_t       = torch.empty(d, device=device)
    change_t    = torch.empty(d, device=device)
    impact_mat  = torch.zeros((d, d), device=device, dtype=start_tensor.dtype)
    t_stat_t = torch.empty(d, device=device, dtype=torch.float32)   # 1‑D, length d
    t_stats, p_vals = [], []               # will live on CPU after SciPy call

    # ── baselines (unchanged) ─────────────────────────────────────────────
    td      = start_tensor.to(device).clone()
    p_td    = propagator.G(td) if propagation else td
    gtd     = stargan.G(td, transition_label_tensor)
    p_gtd   = propagator.G(gtd)

    base_logits = oracle.D(p_td if propagation else td)
    base_acc_list, [base_dist_vec] = calculate_category_total_accuracy(
        base_logits, transition_label_tensor, num_categories
    )
    eps           = torch.finfo(p_gtd.dtype).eps
    baseline_dist_per_feature = torch.abs(p_td - p_gtd).clamp_min(eps)
    ##CHANGES##
    alpha_r = np.arange(1, d+1) * (network_fdr_pval_threshold / d)
    crit_np  = stats.t.ppf(1 - (alpha_r/2), df=batch-1)              # length-M
    crit_torch = torch.tensor(crit_np, device=device, dtype=torch.float32)
    ##CHANGES##
    with torch.inference_mode():
        for col in range(d):
            orig_col, goal_col = td[:, col].clone(), gtd[:, col].clone()
            td[:, col] = goal_col.clone()

            p_td_pert          = propagator.G(td) if propagation else td
            p_td_pert[:, col]  = goal_col.clone()

            preds = oracle.D(p_td_pert if propagation else td)
            acc_list, [dist_vec] = calculate_category_total_accuracy(
                preds, transition_label_tensor, num_categories
            )
            ### calculate oracle score
            dist_improv  = calculate_valid_ratio(base_dist_vec, dist_vec)
            violins_t[col] = dist_improv        

            acc_t[col]   = torch.tensor(acc_list[0], device=device) 

            ### calculate t statistics
            diff = base_dist_vec - dist_vec           # [batch]  on‑device
            mean = diff.mean()                        # scalar
            std  = diff.std(unbiased=True).clamp_min(1e-8)
            se   = std / torch.sqrt(torch.tensor(float(batch), dtype=mean.dtype)) # batch = N
            t_stat_t[col] = mean / se                 # store;   NO SciPy here
            change_t[col] = torch.median(goal_col - orig_col)
            
            ### get the per gene impact
            pert_dist_per_feature        = torch.abs(p_td_pert - p_gtd).clamp_min(eps)
            closeness        = calculate_valid_ratio(baseline_dist_per_feature, pert_dist_per_feature,
                                                     lambda_param=lambda_param)
            mean = closeness.mean(axis=0)
            std  = closeness.std(axis=0,unbiased=True)
            se   = std / np.sqrt(batch)
            t = mean / (se + 1e-12)                 # (N,)          

            keep  = bh_tstat_torch_precomputed(t, crit_torch)
            filtered_median = torch.median(closeness, dim=0).values * keep.to(mean.dtype)#torch.median(closeness, dim=0).values
            
            impact_mat[col]  = filtered_median
            td[:, col] = orig_col
        t_np   = t_stat_t.cpu().numpy()                 # one copy
        df     = batch - 1
        p_np   = 2.0 * stats.t.sf(np.abs(t_np), df)     # vectorised sf; shape (d,)

    # ── single off‑device copy at the end ────────────────────────────────
    violins_np  = violins_t.cpu().numpy()
    accuracy_np = acc_t.cpu().numpy()
    change_np   = change_t.cpu().numpy()
    impact_np   = impact_mat.clamp(min=0.0).cpu().numpy()
    t_stat_np   = np.asarray(t_np)
    p_val_np    = np.asarray(p_np)

    return violins_np, accuracy_np, change_np, t_stat_np, p_val_np, impact_np


