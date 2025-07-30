#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import yaml
import json
import random
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure project root is on path
sys.path.append(os.getcwd())

from utils.misc import set_seed
from utils.model_utils import load_model
from models.encoder import HFEncoder_notPooled
from models.decomposer import NonlinearDecomposer

def parse_args():
    p = argparse.ArgumentParser(description="Jailbreak Detection CLI")
    p.add_argument("--cfg-path",      type=str, required=True, help="Path to jb_detect.yaml")
    p.add_argument("--cfg-out",       type=str, required=True, help="Path to generated config_*.yaml")
    p.add_argument("--unique-id",     type=str, required=True, help="Unique run identifier")
    p.add_argument("--with-val",      action="store_true",       help="Use 80/20 validation split")
    p.add_argument("--use-multigpu",  action="store_true",       help="Enable multiple GPU visible devices")
    p.add_argument("--visible-devices", type=str, default="",    help="Comma-separated CUDA_VISIBLE_DEVICES")
    return p.parse_args()

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f if l.strip() and not l.strip().startswith('#')]

def fit_whiten_pca(X, alpha=0.9):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False) + 1e-5 * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(vals**-0.5) @ vecs.T
    Z = Xc @ W.T
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    r = int(np.searchsorted(cum, alpha)) + 1
    P = pca.components_[:r].T
    return {"mu": mu, "W": W, "P": P}

def residual_vec(V, det):
    z = (V - det["mu"]) @ det["W"].T
    proj = det["P"] @ (det["P"].T @ z.T)
    return z - proj.T

def framing_vecs(texts, encoder, decomposer, batch_size=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            rep   = encoder(batch)
            _, v_f, _ = decomposer(rep)
            if v_f.dim() == 3:
                v_f = v_f.mean(dim=1)
            out.append(v_f.cpu())
    return torch.cat(out, dim=0)

def eval_split(name, vf_ben, vf_jb, score_fn, tau):
    y = np.concatenate([np.zeros(len(vf_ben)), np.ones(len(vf_jb))])
    s = np.concatenate([score_fn(vf_ben.numpy()), score_fn(vf_jb.numpy())])
    au = roc_auc_score(y, s)
    tpr = (s[len(vf_ben):] > tau).mean()
    fpr = (s[:len(vf_ben)] > tau).mean()
    print(f"{name:>6} | AUROC {au:.3f}  TPR@τ {tpr:.3f}  FPR@τ {fpr:.3f}")

def main():
    args = parse_args()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if args.use_multigpu and args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configs
    with open(args.cfg_out, 'r') as f:
        cfg_out = yaml.safe_load(f)
    with open(args.cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override model settings
    config['model']['layers']        = cfg_out['model'].get('layers', 'last')
    config['model']['layer_combine'] = cfg_out['model'].get('layer_combine', 'mean')
    ENC_LLM_NAME = cfg_out['model']['name']

    # Seeds
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

    # Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Starting detection run {args.unique_id}")

    # Load data
    dcfg = config['data']
    rawF_id         = load_jsonl(dcfg["input_path_varyFraming_id"])
    rawG_id         = load_jsonl(dcfg["input_path_varyGoal_id"])
    rawF_benign_id  = load_jsonl(dcfg["input_path_varyFraming_benign_id"])
    rawG_benign_id  = load_jsonl(dcfg["input_path_varyGoal_benign_id"])
    rawF_ood        = load_jsonl(dcfg["input_path_varyFraming_ood"])
    rawG_ood        = load_jsonl(dcfg["input_path_varyGoal_ood"])
    rawF_benign_ood = load_jsonl(dcfg["input_path_varyFraming_benign_ood"])
    rawG_benign_ood = load_jsonl(dcfg["input_path_varyGoal_benign_ood"])

    benign_id    = rawF_benign_id + rawG_benign_id
    jailbrks_id  = rawF_id + rawG_id
    benign_ood   = rawF_benign_ood + rawG_benign_ood
    jailbrks_ood = rawF_ood + rawG_ood

    # Balance ID sets
    m = min(len(benign_id), len(jailbrks_id))
    benign_id   = random.sample(benign_id, m)
    jailbrks_id = random.sample(jailbrks_id, m)

    prompts = lambda data: [e["prompt"] for e in data]
    ben_ID   = prompts(benign_id)
    jb_ID    = prompts(jailbrks_id)
    ben_OOD  = prompts(benign_ood)
    jb_OOD   = prompts(jailbrks_ood)

    # Load encoder + decomposer
    model_llm, tokenizer = load_model(ENC_LLM_NAME, device=device)
    encoder = HFEncoder_notPooled(
        model=model_llm, tokenizer=tokenizer, device=device,
        layers=config['model']['layers'],
        layer_combine=config['model']['layer_combine']
    )
    ckpt_path = Path(f"./checkpoints/decomposer_simple/decomposer_{args.unique_id}") / "weights.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    enc_dim_ckpt = ckpt["Wg.0.weight"].shape[1]
    decomposer = NonlinearDecomposer(
        enc_dim=enc_dim_ckpt,
        d_g=config['d_g'],
        d_f=config['d_f'],
        hidden_dim=config.get('hidden_dim', 1024),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    decomposer.load_state_dict(ckpt)
    decomposer.half().eval()
    encoder.eval()

    # Build framing vectors
    vf_ben_ID  = framing_vecs(ben_ID, encoder, decomposer)
    vf_jb_ID   = framing_vecs(jb_ID,  encoder, decomposer)
    vf_ben_OOD = framing_vecs(ben_OOD, encoder, decomposer)
    vf_jb_OOD  = framing_vecs(jb_OOD,  encoder, decomposer)

    # Cosine-based detection
    detector_cos = fit_whiten_pca(vf_ben_ID.numpy())
    pickle.dump(detector_cos, open("checkpoints/nsp_detector.pkl", "wb"))
    R_ben_ID = residual_vec(vf_ben_ID.numpy(), detector_cos)
    R_ben_ID = R_ben_ID / (np.linalg.norm(R_ben_ID, axis=1, keepdims=True) + 1e-9)
    centroid = R_ben_ID.mean(0, keepdims=True)

    def cos_score(V): 
        R = residual_vec(V, detector_cos)
        R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        return 1.0 - (R * centroid).sum(1)

    tau_cos = np.percentile(cos_score(vf_ben_ID.numpy()), 95)
    print(f"Cosine τ = {tau_cos:.4f}")
    eval_split("ID", vf_ben_ID, vf_jb_ID, cos_score, tau_cos)
    eval_split("OOD", vf_ben_OOD, vf_jb_OOD, cos_score, tau_cos)

    # χ²-based L2 detection
    def run_l2(ben, jb, name, detector, tau):
        eval_split(name, ben, jb,
                   lambda V: np.linalg.norm(residual_vec(V, detector), axis=1),
                   tau)

    if args.with_val:
        ben_tr, ben_val = train_test_split(vf_ben_ID, test_size=0.2, random_state=seed)
        jb_tr, jb_val   = train_test_split(vf_jb_ID,  test_size=0.2, random_state=seed)
        det_l2 = fit_whiten_pca(ben_tr.numpy())
        df = ben_tr.shape[1] - det_l2["P"].shape[1]
        tau_l2 = chi2.ppf(0.95, df=df) ** 0.5
        print(f"χ² τ (with val) = {tau_l2:.4f}")
        run_l2(ben_val, jb_val, "ID-val", det_l2, tau_l2)
        run_l2(vf_ben_OOD, vf_jb_OOD, "OOD", det_l2, tau_l2)
    else:
        det_l2 = fit_whiten_pca(vf_ben_ID.numpy())
        df = vf_ben_ID.shape[1] - det_l2["P"].shape[1]
        tau_l2 = chi2.ppf(0.95, df=df) ** 0.5
        print(f"χ² τ = {tau_l2:.4f}")
        run_l2(vf_ben_ID, vf_jb_ID, "ID", det_l2, tau_l2)
        run_l2(vf_ben_OOD, vf_jb_OOD, "OOD", det_l2, tau_l2)

if __name__ == "__main__":
    main()
