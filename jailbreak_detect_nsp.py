#!/usr/bin/env python3
"""
Jailbreak-detection via L2-χ² NSP residuals.

Example
-------
python detect_jb_nsp.py \
       --cfg-path configs/jb_detect_nsp.yaml \
       --dec-unique-id 20250717_101812_06190f25-e1f1-4ed4-87ae-a51365b6061b
       --cfg-unique-id 20250717_101812_06190f25-e1f1-4ed4-87ae-a51365b6061b
"""

from __future__ import annotations
import argparse, os, sys, yaml, json, logging, random, time, uuid, pickle
from pathlib import Path
from typing import List

import torch, numpy as np
from accelerate import notebook_launcher          # same multi-GPU helper
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSP jailbreak detector")
    p.add_argument("--cfg-path",  required=True, help="Path to jb_detect_nsp.yaml")
    p.add_argument("--dec-unique-id", required=True, help="RUN_ID_dec from training phase for the decomposer")
    p.add_argument("--cfg-unique-id", help="Inique ID from training phase for the config file",
                   default=None)
    p.add_argument("--detect-goal", action="store_true",
                   help="Detect using goal vectors instead of framing")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Optional override for the PCA variance thresholds.")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def load_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(l) for l in f if l.strip() and not l.strip().startswith("#")]

# ---------------------------------------------------------------------
# Global placeholders (will be set in main)
# ---------------------------------------------------------------------
CFG            = None
CFG_OUT        = None          # config dict (dict)
AR             = None          # all raw data (dict)
RUN_ID_dec     = ""
RUN_ID_cfg     = ""
TIMESTAMP      = ""
LOGGER_NAME    = "detect_nsp"
RESULTS = []          # will collect one dict per alpha / split


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    global CFG, CFG_OUT, AR, RUN_ID_dec, RUN_ID_cfg, TIMESTAMP

    args     = parse_args()
    RUN_ID_dec   = args.dec_unique_id
    TIMESTAMP= time.strftime("%Y%m%d_%H%M%S")
    
    if args.cfg_unique_id is None:
        args.cfg_unique_id = args.dec_unique_id
    RUN_ID_cfg = args.cfg_unique_id

    # ---------- load YAML ----------
    with open(args.cfg_path, "r") as f:
        CFG = yaml.safe_load(f)
    if args.alphas is not None:
        CFG["detector"]["alphas"] = args.alphas
    CFG["detector"]["detect_via_framing"] = not args.detect_goal

    # ---------- logging ----------
    Path("logs").mkdir(exist_ok=True)
    log_file = Path(f"logs/detect_{TIMESTAMP}_{RUN_ID_dec}.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s — %(levelname)s — %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_file, mode="w")])
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Log → %s", log_file)

    # ---------- seeds ----------
    seed = CFG["experiment"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ---------- GPU ----------
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s  (#GPUs=%d)", device, torch.cuda.device_count())

    # ---------- resolve paths ----------
    # ckpt_dir   = Path(f"checkpoints/decomposer_simple/decomposer_{RUN_ID_dec}")
    ckpt_dir   = Path(f"checkpoints/decomposer_simple/")
    cfg_out    = Path(f"output/config_{RUN_ID_cfg}.yaml")
    with open(cfg_out) as f:
        cfg_out_dct = yaml.safe_load(f)
    CFG_OUT = cfg_out_dct

    # merge layer settings into CFG so code below matches notebook
    for k in ["layers", "layer_combine"]:
        CFG.setdefault("model", {})[k] = cfg_out_dct["model"].get(k)
    ENC_LLM_NAME = cfg_out_dct["model"]["name"]

    # ---------- load datasets ----------
    paths = CFG["data"]
    AR = {
        "F_id":  load_jsonl(paths["input_path_varyFraming_id"]),
        "G_id":  load_jsonl(paths["input_path_varyGoal_id"]),
        "Fb_id": load_jsonl(paths["input_path_varyFraming_benign_id"]),
        "Gb_id": load_jsonl(paths["input_path_varyGoal_benign_id"]),
        "F_ood": load_jsonl(paths["input_path_varyFraming_ood"]),
        "G_ood": load_jsonl(paths["input_path_varyGoal_ood"]),
        "Fb_ood":load_jsonl(paths["input_path_varyFraming_benign_ood"]),
        "Gb_ood":load_jsonl(paths["input_path_varyGoal_benign_ood"]),
    }
    # combine & balance
    benign_id   = AR["Fb_id"] + AR["Gb_id"]
    jail_id     = AR["F_id"]  + AR["G_id"]
    m = min(len(benign_id), len(jail_id))
    benign_id, jail_id = random.sample(benign_id, m), random.sample(jail_id, m)
    benign_ood  = AR["Fb_ood"]+ AR["Gb_ood"]
    jail_ood    = AR["F_ood"] + AR["G_ood"]

    # ---------- launch worker (single-process; no DDP needed) ----------
    detect_worker(device, ENC_LLM_NAME, ckpt_dir, benign_id, jail_id,
                  benign_ood, jail_ood, args.batch_size, logger)

# # ---------------------------------------------------------------------
# # Worker (core notebook logic)
# # ---------------------------------------------------------------------
# def detect_worker(device, enc_name, ckpt_dir, ben_id, jb_id, ben_ood, jb_ood,
#                   batch_size, logger):
#     from utils.model_utils import load_model
#     from models.encoder   import HFEncoder_notPooled
#     from models.decomposer import NonlinearDecomposer, NonlinearDecomposer_tiny

#     # -------- load encoder & decomposer --------
#     model, tok = load_model(enc_name, device=device)
#     encoder = HFEncoder_notPooled(
#         model=model, tokenizer=tok, device=device,
#         layers=CFG["model"]["layers"],
#         layer_combine=CFG["model"]["layer_combine"],
#     )
#     ckpt = torch.load(Path(ckpt_dir) / "weights.pt", map_location=device)
#     enc_dim = ckpt["Wg.0.weight"].shape[1]
#     decomposer = NonlinearDecomposer_tiny(enc_dim=enc_dim).to(device)
#     decomposer.load_state_dict(ckpt)
#     encoder.eval(); decomposer.eval()

#     # -------- vector builders --------
#     @torch.no_grad()
#     def f_vec(txts: List[str]):
#         out=[]
#         for i in range(0,len(txts), batch_size):
#             rep = encoder(txts[i:i+batch_size])
#             _,v_f,_ = decomposer(rep)
#             if v_f.dim()==3: v_f=v_f.mean(1)
#             out.append(v_f.cpu())
#         return torch.cat(out)
#     @torch.no_grad()
#     def g_vec(txts: List[str]):
#         out=[]
#         for i in range(0,len(txts), batch_size):
#             rep = encoder(txts[i:i+batch_size])
#             v_g,_,_ = decomposer(rep)
#             if v_g.dim()==3: v_g=v_g.mean(1)
#             out.append(v_g.cpu())
#         return torch.cat(out)

#     get_vec = f_vec if CFG["detector"]["detect_via_framing"] else g_vec
#     to_np   = lambda x: x.numpy()

#     v_ben_ID  = get_vec([e["prompt"] for e in ben_id])
#     v_jb_ID   = get_vec([e["prompt"] for e in jb_id])
#     v_ben_OOD = get_vec([e["prompt"] for e in ben_ood])
#     v_jb_OOD  = get_vec([e["prompt"] for e in jb_ood])
#     logger.info("Built %s vectors.", "framing" if CFG["detector"]["detect_via_framing"] else "goal")

#     # -------- NSP helpers --------
#     def fit_whiten_pca(X, alpha):
#         mu=X.mean(0,keepdims=True); Z=(X-mu)
#         cov=np.cov(Z,rowvar=False)+1e-5*np.eye(X.shape[1])
#         vals,vecs=np.linalg.eigh(cov)
#         W=vecs@np.diag(vals**-0.5)@vecs.T
#         Z=Z@W.T
#         pca=PCA().fit(Z)
#         r=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),alpha))+1
#         P=pca.components_[:r].T
#         return {"mu":mu,"W":W,"P":P}
#     def resid(V,det):
#         z=(V-det["mu"])@det["W"].T
#         proj=det["P"]@(det["P"].T@z.T)
#         return z-proj.T
#     def nsp_l2(V,det): return np.linalg.norm(resid(V,det),axis=1)

#     # -------- with validation split --------
#     alphas=CFG["detector"]["alphas"]
#     ben_tr, ben_val = train_test_split(v_ben_ID, test_size=.2, random_state=42)
#     jb_tr , jb_val  = train_test_split(v_jb_ID , test_size=.2, random_state=42)
#     dets_val={a: fit_whiten_pca(ben_tr.numpy(), a) for a in alphas}
#     taus_val={a: chi2.ppf(.95, df=ben_tr.shape[1]-dets_val[a]["P"].shape[1])**.5
#               for a in alphas}
#     _evaluate("ID-val", ben_val, jb_val, dets_val, taus_val, nsp_l2, alphas, logger,
#               result_key="val_id")
#     _evaluate("OOD", v_ben_OOD, v_jb_OOD, dets_val, taus_val, nsp_l2, alphas, logger,
#               result_key="val_ood")

#     # -------- all ID (no val) -----------
#     dets={a: fit_whiten_pca(v_ben_ID.numpy(), a) for a in alphas}
#     taus={a: chi2.ppf(.95, df=v_ben_ID.shape[1]-dets[a]["P"].shape[1])**.5
#           for a in alphas}
#     _evaluate("ID", v_ben_ID, v_jb_ID, dets, taus, nsp_l2, alphas, logger,
#               result_key="noVal_id")
#     _evaluate("OOD", v_ben_OOD, v_jb_OOD, dets, taus, nsp_l2, alphas, logger,
#               result_key="noVal_ood")
#     # ------------ save metrics -------------
#     out_path = Path("output") / f"detect_metrics_{RUN_ID_dec}.json"
#     out_path.parent.mkdir(exist_ok=True)
#     with open(out_path, "w") as f:
#         json.dump(RESULTS, f, indent=2)
#     logger.info("Saved metrics → %s", out_path)



# ---------------------------------------------------------------------
# Worker (core notebook logic)  – REPLACE the old function with this one
# ---------------------------------------------------------------------
def detect_worker(device, enc_name, ckpt_root, ben_id, jb_id, ben_ood, jb_ood,
                  batch_size, logger):
    """
    • Performs *critical-layer selection* on-the-fly (Cohen-d on NSP scores).
    • Loads the encoder + the decomposer of the winning layer.
    • Runs the usual χ²-NSP detector and logs metrics.
    """
    from utils.model_utils import load_model
    from models.encoder import HFEncoder_notPooled
    from models.decomposer import NonlinearDecomposer
    from utils.critial_layer import find_critical_layers_dist

    # ---------------- checkpoint map: layer → path -------------------
    import re, glob, os

    pattern = re.compile(r"decomposer_layer(\d+)_")
    ckpt_by_layer = {}
    for p in glob.glob(f"./checkpoints/decomposer_simple/decomposer_layer*{RUN_ID_dec}*"):
        m = pattern.search(os.path.basename(p))
        if m: ckpt_by_layer[int(m.group(1))] = p
    if len(ckpt_by_layer) == 1: # a specific case which was messed up
        ckpt_by_layer = {}
        for p in glob.glob(f"./checkpoints/decomposer_simple/decomposer_layer*"):
            m = pattern.search(os.path.basename(p))
            if m: ckpt_by_layer[int(m.group(1))] = p
    if not ckpt_by_layer:
        raise FileNotFoundError(f"No per-layer checkpoints found for run {RUN_ID_dec}")

    # ---------------- load the frozen LLM once -----------------------
    model, tok = load_model(enc_name, device=device)
    n_layers   = model.config.num_hidden_layers
    cand_layers= list(range(n_layers))

    # ---------------- build calibration texts ------------------------
    N_CAL = 100
    cal_ben = random.sample([e["prompt"] for e in ben_id],  N_CAL // 2)
    cal_jb  = random.sample([e["prompt"] for e in jb_id],   N_CAL // 2)

    # ---------------- helper to load a decomposer --------------------
    def load_dec(layer_id: int):
        ckpt = torch.load(Path(ckpt_by_layer[layer_id]) / "weights.pt",
                          map_location=device)
        dec  = NonlinearDecomposer(
                 enc_dim=model.config.hidden_size,
                 d_g=CFG_OUT["d_g"], d_f=CFG_OUT["d_f"]).to(device).eval()
        dec.load_state_dict(ckpt)
        dec.half().eval()
        dec.eval()
        return dec


    # ---------------- critical-layer search --------------------------
    CRIT_METRIC = "cohen_d" 
    cl_outs = dict()
    best_g, best_f = dict(), dict()
    for layer in cand_layers:
        # ==== Cell: [Model & Decomposer Initialization] ====

        # load decomposer weights
        decomposer = load_dec(layer)
        
        # print(f"\n\n\nlayer {layer}:\n")
        with torch.no_grad():
            cl_outs[layer] = find_critical_layers_dist(
                HFEncoder_notPooled, model, tok, device,
                decomposer, cal_ben, cal_jb,
                candidate_layers=[layer],
                criterion=CRIT_METRIC,
                alpha=0.9,
                verbose=False,
            )
            best_g[layer] = {"encoder_l": cl_outs[layer]["best_layer_goal"], "score": cl_outs[layer]["score_goal"]}
            best_f[layer] = {"encoder_l": cl_outs[layer]["best_layer_framing"], "score": cl_outs[layer]["score_framing"]}
            # print(f"Best G: {best_g[layer]}, Best F: {best_f[layer]}")
    
    best_g_tups = sorted(best_g.items(), key=lambda x: x[1]['score'], reverse=True)
    best_f_tups = sorted(best_f.items(), key=lambda x: x[1]['score'], reverse=True)
    use_framing = CFG["detector"]["detect_via_framing"]
    if use_framing:
        enc_layer = [l for l, _ in best_f_tups if l > n_layers//2][0]
    else:
        enc_layer = [l for l, _ in best_g_tups if l > n_layers//2][0]
    logger.info("Selected layer %d for %s (metric=%s, score=%.3f)",
                enc_layer,
                "framing" if use_framing else "goal",
                CRIT_METRIC,
                best_f[enc_layer]["score"] 
                if use_framing 
                else best_g[enc_layer]["score"])
    # ----------------
    

    # ---------------- load encoder + decomposer for that layer -------
    encoder = HFEncoder_notPooled(
        model=model, tokenizer=tok, device=device,
        layers=[enc_layer], layer_combine="mean")

    decomposer = load_dec(enc_layer)        # already eval()

    # ---------------- vector builders --------------------------------
    @torch.no_grad()
    def f_vec(txts):
        out=[]
        for i in range(0, len(txts), batch_size):
            rep = encoder(txts[i:i+batch_size])
            _, v_f, _ = decomposer(rep)
            out.append(v_f.mean(1).cpu() if v_f.dim()==3 else v_f.cpu())
        return torch.cat(out)

    @torch.no_grad()
    def g_vec(txts):
        out=[]
        for i in range(0, len(txts), batch_size):
            rep = encoder(txts[i:i+batch_size])
            v_g, _, _ = decomposer(rep)
            out.append(v_g.mean(1).cpu() if v_g.dim()==3 else v_g.cpu())
        return torch.cat(out)

    get_vec = f_vec if use_framing else g_vec
    # ---------- build all splits (ID / OOD) ----------
    v_ben_ID  = get_vec([e["prompt"] for e in ben_id])
    v_jb_ID   = get_vec([e["prompt"] for e in jb_id])
    v_ben_OOD = get_vec([e["prompt"] for e in ben_ood])
    v_jb_OOD  = get_vec([e["prompt"] for e in jb_ood])
    logger.info("Built %s vectors on layer %d.",
                "framing" if use_framing else "goal", enc_layer)

    # ---------- rest of the original worker – UNCHANGED --------------
    _run_nsp_detection(v_ben_ID, v_jb_ID, v_ben_OOD, v_jb_OOD,
                       batch_size, logger)


# ---------------------------------------------------------------------
# helper: map every layer id to its checkpoint dir
# ---------------------------------------------------------------------
def _map_layer_to_ckpt(root_dir: str, run_id: str) -> dict[int, str]:
    """
    Returns {layer_id: path_to_ckpt_dir} for the given run-id.
    """
    import re, glob, os
    patt = re.compile(r"decomposer_layer(\d+)_.*" + re.escape(run_id))
    mapping = {}
    for p in glob.glob(os.path.join(root_dir, "decomposer_layer*")):
        m = patt.search(os.path.basename(p))
        if m:
            mapping[int(m.group(1))] = p
    return mapping


# ---------------------------------------------------------------------
# helper: run NSP detector on ID / OOD splits (identical to original)
# ---------------------------------------------------------------------
def _run_nsp_detection(v_ben_ID, v_jb_ID, v_ben_OOD, v_jb_OOD,
                       batch_size, logger):
    """
    """
    # -------- NSP helpers --------
    def fit_whiten_pca(X, alpha):
        mu=X.mean(0,keepdims=True); Z=(X-mu)
        cov=np.cov(Z,rowvar=False)+1e-5*np.eye(X.shape[1])
        vals,vecs=np.linalg.eigh(cov)
        W=vecs@np.diag(vals**-0.5)@vecs.T
        Z=Z@W.T
        pca=PCA().fit(Z)
        r=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),alpha))+1
        P=pca.components_[:r].T
        return {"mu":mu,"W":W,"P":P}
    def resid(V,det):
        z=(V-det["mu"])@det["W"].T
        proj=det["P"]@(det["P"].T@z.T)
        return z-proj.T
    def nsp_l2(V,det): return np.linalg.norm(resid(V,det),axis=1)

    # -------- with validation split --------
    alphas=CFG["detector"]["alphas"]
    ben_tr, ben_val = train_test_split(v_ben_ID, test_size=.2, random_state=42)
    jb_tr , jb_val  = train_test_split(v_jb_ID , test_size=.2, random_state=42)
    dets_val={a: fit_whiten_pca(ben_tr.numpy(), a) for a in alphas}
    taus_val={a: chi2.ppf(.95, df=ben_tr.shape[1]-dets_val[a]["P"].shape[1])**.5
              for a in alphas}
    _evaluate("ID-val", ben_val, jb_val, dets_val, taus_val, nsp_l2, alphas, logger,
              result_key="val_id")
    _evaluate("OOD", v_ben_OOD, v_jb_OOD, dets_val, taus_val, nsp_l2, alphas, logger,
              result_key="val_ood")

    # -------- all ID (no val) -----------
    dets={a: fit_whiten_pca(v_ben_ID.numpy(), a) for a in alphas}
    taus={a: chi2.ppf(.95, df=v_ben_ID.shape[1]-dets[a]["P"].shape[1])**.5
          for a in alphas}
    _evaluate("ID", v_ben_ID, v_jb_ID, dets, taus, nsp_l2, alphas, logger,
              result_key="noVal_id")
    _evaluate("OOD", v_ben_OOD, v_jb_OOD, dets, taus, nsp_l2, alphas, logger,
              result_key="noVal_ood")
    # ------------ save metrics -------------
    out_path = Path("output") / f"detect_metrics_{RUN_ID_dec}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    logger.info("Saved metrics → %s", out_path)

    pass





# ---------------------------------------------------------------------
# Small utility to print metrics exactly like the notebook
# ---------------------------------------------------------------------
def _evaluate(tag, v_ben, v_jb, dets, taus, scorer, alphas, logger,
              result_key):
    y = np.concatenate([np.zeros(len(v_ben)), np.ones(len(v_jb))])
    for a in alphas:
        s=np.concatenate([scorer(v_ben.numpy(), dets[a]),
                          scorer(v_jb.numpy(), dets[a])])
        y_hat=(s>taus[a]).astype(int)
        au=roc_auc_score(y,s)
        acc=accuracy_score(y,y_hat)
        prec,rec,f1,_=precision_recall_fscore_support(y,y_hat,average="binary",
                                                      pos_label=1,zero_division=0)
        tpr=y_hat[len(v_ben):].mean(); fpr=y_hat[:len(v_ben)].mean()
        RESULTS.append({result_key:
            {
                "run_id": RUN_ID_dec,
                "split": tag,
                "alpha": float(a),
                "acc":   acc,
                "f1":    f1,
                "auroc": au,
                "precision": prec,
                "recall":    rec,
                "tpr":   tpr,
                "fpr":   fpr
            }
        })
        logger.info(f"{result_key} ----  alpha=%0.2f , %-6s | Acc %.3f F1 %.3f AUROC %.3f "
                    "Prec %.3f Rec %.3f TPR %.3f FPR %.3f",
                    a, tag, acc, f1, au, prec, rec, tpr, fpr)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
