

# ========== by cosine similarity ==========

import torch, torch.nn.functional as F, yaml, json, random
from pathlib import Path
from tqdm import tqdm

def find_critical_layers(Encoder, model, tok, device, decomposer,
                         ben_texts, jb_texts,
                         candidate_layers):
    best_f, best_g, max_df, max_dg = None, None, -1, -1
    for ℓ in tqdm(candidate_layers):
        encoder = Encoder(
            model=model,
            tokenizer=tok,
            device=device,
            layers=[ℓ],
            layer_combine='mean',
        )
        rep_b = encoder(ben_texts)
        rep_j = encoder(jb_texts)

        vg_b, vf_b, _ = decomposer(rep_b)
        vg_j, vf_j, _ = decomposer(rep_j)

        if vf_b.dim()==3:  # token-wise
            vf_b = vf_b.mean(1); vf_j = vf_j.mean(1)
            vg_b = vg_b.mean(1); vg_j = vg_j.mean(1)

        Δf = 1 - (F.normalize(vf_b,dim=-1) @
                  F.normalize(vf_j,dim=-1).T).mean().item()
        Δg = 1 - (F.normalize(vg_b,dim=-1) @
                  F.normalize(vg_j,dim=-1).T).mean().item()
        print(f"layer {ℓ}: Δ_f: {Δf}, Δ_g: {Δg}")

        if Δf > max_df: best_f, max_df = ℓ, Δf
        if Δg > max_dg: best_g, max_dg = ℓ, Δg

    return {"l_f": best_f, "l_g": best_g,
            "Δ_f": max_df, "Δ_g": max_dg}



# ========== by nsp distance and anomaly score ==========

# utils/critical_layer.py  (add / replace with this block)
import torch, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix
)
from sklearn.decomposition import PCA
from scipy.stats import chi2


# ------------------------------------------------------------------------ #
# detector helpers (same as your NSP code)                                 #
# ------------------------------------------------------------------------ #
def _fit_whiten_pca(X: np.ndarray, alpha=0.9):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False) + 1e-5 * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(vals ** -0.5) @ vecs.T
    Z = Xc @ W.T

    pca = PCA().fit(Z)
    r   = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), alpha)) + 1
    return {"mu": mu, "W": W, "P": pca.components_[:r].T}


def _residual_score(V: np.ndarray, det):
    Z = (V - det["mu"]) @ det["W"].T
    proj = det["P"] @ (det["P"].T @ Z.T)
    return np.linalg.norm(Z - proj.T, axis=1)


# ------------------------------------------------------------------------ #
# metric helpers                                                           #
# ------------------------------------------------------------------------ #
def _metric(sb, sj, criterion, tau=None):
    y_true = np.concatenate([np.zeros(len(sb)), np.ones(len(sj))])
    scores = np.concatenate([sb, sj])

    if criterion == "auroc":
        return roc_auc_score(y_true, scores)

    if criterion in {"f1", "acc", "fpr"}:
        if tau is None:
            raise ValueError("tau required for F1/ACC/FPR metrics")
        y_pred = (scores > tau).astype(int)
        if criterion == "f1":
            return f1_score(y_true, y_pred)
        if criterion == "acc":
            return accuracy_score(y_true, y_pred)
        if criterion == "fpr":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fp, tn = cm[0, 1], cm[0, 0]
            return fp / (fp + tn + 1e-12)
    raise ValueError(f"unknown criterion {criterion}")


def _cohen_d(x, y):
    mu_x, mu_y = x.mean(), y.mean()
    var_pool   = (x.var(ddof=1) + y.var(ddof=1)) / 2
    return abs(mu_y - mu_x) / (np.sqrt(var_pool) + 1e-12)


# ------------------------------------------------------------------------ #
# joint selector                                                            #
# ------------------------------------------------------------------------ #
def find_critical_layers_dist(
    Encoder, model, tok, device,
    dec,
    ben_texts, jb_texts,
    candidate_layers,
    criterion: str = "auroc",          # "auroc"|"f1"|"acc"|"fpr"|"cohen_d"
    alpha: float = 0.9,
    verbose: bool = True,
):
    """
    Returns best layer *for framing*  and best layer *for goal* in one shot.
    """
    criterion = criterion.lower()
    better    = (lambda a, b: a > b) if criterion != "fpr" else (lambda a, b: a < b)
    use_perf  = criterion != "cohen_d"

    best_f, best_g = None, None
    best_val_f = -np.inf if criterion != "fpr" else np.inf
    best_val_g = best_val_f

    scores_f, scores_g = {}, {}

    for ℓ in tqdm(candidate_layers):
        enc = Encoder(model=model, tokenizer=tok, device=device,
                      layers=[ℓ], layer_combine="mean")

        with torch.no_grad():
            vg_b, vf_b, _ = dec(enc(ben_texts))
            vg_j, vf_j, _ = dec(enc(jb_texts))

        if vf_b.dim() == 3:          # pool tokens
            vf_b = vf_b.mean(1); vf_j = vf_j.mean(1)
            vg_b = vg_b.mean(1); vg_j = vg_j.mean(1)

        # --------------- framing metric -----------------
        if use_perf:
            det_f = _fit_whiten_pca(vf_b.cpu().numpy(), alpha)
            df    = vf_b.size(-1) - det_f["P"].shape[1]
            tau_f = chi2.ppf(0.95, df=df) ** 0.5
            sb_f  = _residual_score(vf_b.cpu().numpy(), det_f)
            sj_f  = _residual_score(vf_j.cpu().numpy(), det_f)
            val_f = _metric(sb_f, sj_f, criterion, tau_f)
        else:
            det_f = _fit_whiten_pca(vf_b.cpu().numpy(), alpha)
            val_f = _cohen_d(
                _residual_score(vf_b.cpu().numpy(), det_f),
                _residual_score(vf_j.cpu().numpy(), det_f)
            )

        scores_f[ℓ] = val_f
        if better(val_f, best_val_f):
            best_f, best_val_f = ℓ, val_f

        # --------------- goal metric --------------------
        if use_perf:
            det_g = _fit_whiten_pca(vg_b.cpu().numpy(), alpha)
            dg    = vg_b.size(-1) - det_g["P"].shape[1]
            tau_g = chi2.ppf(0.95, df=dg) ** 0.5
            sb_g  = _residual_score(vg_b.cpu().numpy(), det_g)
            sj_g  = _residual_score(vg_j.cpu().numpy(), det_g)
            val_g = _metric(sb_g, sj_g, criterion, tau_g)
        else:
            det_g = _fit_whiten_pca(vg_b.cpu().numpy(), alpha)
            val_g = _cohen_d(
                _residual_score(vg_b.cpu().numpy(), det_g),
                _residual_score(vg_j.cpu().numpy(), det_g)
            )

        scores_g[ℓ] = val_g
        if better(val_g, best_val_g):
            best_g, best_val_g = ℓ, val_g

        if verbose:
            print(f"layer {ℓ:>2}:   framing={val_f:.4f} , goal={val_g:.4f}")

    return {
        "criterion":         criterion,
        "alpha":             alpha,
        "best_layer_framing": best_f,
        "score_framing":      best_val_f,
        "best_layer_goal":    best_g,
        "score_goal":         best_val_g,
        "all_scores_framing": scores_f,
        "all_scores_goal":    scores_g,
    }
