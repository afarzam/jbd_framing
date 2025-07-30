#!/usr/bin/env python3
"""
jbshield_core.py  –  faithful re-implementation of JBShield-D (USENIX'25).

Public API
----------
det = JBShieldDetector(backbone_name)           # same HF backbone you use elsewhere
det.calibrate(cal_ben, cal_har, cal_jb)         # lists[str]
y_hat = det.predict(list_of_prompts)            # np.ndarray of {0,1}
"""

from __future__ import annotations
import logging, random
import numpy as np
import torch, torch.nn.functional as F
from sklearn.metrics import roc_curve
from utils.model_utils import load_model           # already in your repo
from models.encoder import HFEncoder_notPooled
import tqdm 

LOG = logging.getLogger("jbshield")

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mean_pool_hidden(h):                         # (B,T,d) → (B,d)
    return h.mean(dim=1)

def _pairwise_mean_cosine(A: torch.Tensor, B: torch.Tensor) -> float:
    """mean_{i,j} cosine(A[i], B[j])  –  both (n,d) cpu tensors."""
    A_n = F.normalize(A, dim=1);  B_n = F.normalize(B, dim=1)
    cos = A_n @ B_n.T
    return cos.mean().item()


def _top_singular_vec(Δ: np.ndarray) -> np.ndarray:
    # NumPy SVD needs float32/64
    Δ = Δ.astype(np.float32, copy=False)          
    _, _, vT = np.linalg.svd(Δ, full_matrices=False)
    v = vT[0]
    return v / np.linalg.norm(v)


def _youden_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]

# --------------------------------------------------------------------------- #
class JBShieldDetector:
    def __init__(self,
                 model_name: str,
                 device: str = "cuda",
                 layers_search: list[int] | None = None,
                 batch_size: int = 32):
        self.device = device
        self.model, self.tok = load_model(model_name, device=device)
        self.bs   = batch_size
        self.layers = (layers_search if layers_search is not None
                       else list(range(self.model.config.num_hidden_layers)))
        # will be filled in calibrate()
        self.lt = self.lj = None           # critical layers
        self.v_tox = self.v_jb = None      # concept vectors
        self.th_t = self.th_j = None       # thresholds

    # ---------------- internal helpers -------------------------------- #
    def _embed(self, prompts: list[str], layer: int) -> torch.Tensor:
        enc = HFEncoder_notPooled(model=self.model,
                                  tokenizer=self.tok,
                                  device=self.device,
                                  layers=[layer],
                                  layer_combine="mean")
        outs = []
        with torch.no_grad():
            for i in range(0, len(prompts), self.bs):
                h = enc(prompts[i:i+self.bs])          # (B,T,d)
                # outs.append(_mean_pool_hidden(h).cpu()) # mean pool over tokens
                outs.append(h[:,-1,:].cpu()) # last token
        return torch.cat(outs)   # (N,d)

    # ---------------- public API -------------------------------------- #
    def calibrate(self,
                  cal_ben: list[str],
                  cal_har: list[str],
                  cal_jb : list[str],
                  pct: float = .95):
        LOG.info("Calibrating JBShield-D …  |B|=%d |H|=%d |J|=%d",
                 len(cal_ben), len(cal_har), len(cal_jb))

        # ---------- 1) critical layer for toxic concept ----------------
        min_cos, best_l = 1.0, None
        for l in self.layers:
            cos = _pairwise_mean_cosine(self._embed(cal_har, l),
                                         self._embed(cal_ben, l))
            if cos < min_cos:
                min_cos, best_l = cos, l
        self.lt = best_l
        LOG.info("Toxic concept layer = %d  (mean cosine %.3f)", self.lt, min_cos)

        # ---------- 1b) critical layer for JB concept ------------------
        min_cos, best_l = 1.0, None
        for l in self.layers:
            cos = _pairwise_mean_cosine(self._embed(cal_jb , l),
                                         self._embed(cal_har, l))
            if cos < min_cos:
                min_cos, best_l = cos, l
        self.lj = best_l
        LOG.info("JB concept layer   = %d  (mean cosine %.3f)", self.lj, min_cos)

        # ---------- 2) concept sub-spaces (rank-1) ---------------------
        E_h_t = self._embed(cal_har, self.lt)
        E_b_t = self._embed(cal_ben, self.lt)
        Δ_t   = (E_h_t.unsqueeze(1) - E_b_t.unsqueeze(0)).reshape(-1, E_h_t.size(1))
        self.v_tox = _top_singular_vec(Δ_t.numpy())

        E_j_j = self._embed(cal_jb, self.lj)
        E_h_j = self._embed(cal_har, self.lj)
        Δ_j   = (E_j_j.unsqueeze(1) - E_h_j.unsqueeze(0)).reshape(-1, E_j_j.size(1))
        self.v_jb  = _top_singular_vec(Δ_j.numpy())

        # ---------- 3) thresholds via Youden-J -------------------------
        s_pos_t = self._cos_scores(E_h_t , self.v_tox)
        s_neg_t = self._cos_scores(E_b_t , self.v_tox)
        self.th_t = _youden_threshold(s_pos_t, s_neg_t)

        s_pos_j = self._cos_scores(E_j_j , self.v_jb)
        s_neg_j = self._cos_scores(E_h_j , self.v_jb)
        self.th_j = _youden_threshold(s_pos_j, s_neg_j)

        LOG.info("Thresholds  θ_t=%.3f  θ_j=%.3f", self.th_t, self.th_j)

    # ------------------------------------------------------------------ #
    def _cos_scores(self, X: torch.Tensor, v: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(v).to(dtype=X.dtype)
        return F.cosine_similarity(X, v.expand_as(X), dim=1).cpu().numpy()

    def score(self, prompt: str) -> tuple[float, float]:
        e_t = self._embed([prompt], self.lt)
        e_j = self._embed([prompt], self.lj)
        return (float(self._cos_scores(e_t, self.v_tox)),
                float(self._cos_scores(e_j, self.v_jb )))

    def predict(self, prompts: list[str]) -> np.ndarray:
        lab = []
        for p in tqdm.tqdm(prompts, desc="Predicting JBShield scores"):
            s_t, s_j = self.score(p)
            lab.append(int((s_t >= self.th_t) and (s_j >= self.th_j)))
        return np.array(lab)
# --------------------------------------------------------------------------- #
