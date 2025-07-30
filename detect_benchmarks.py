#!/usr/bin/env python3
"""
detect_benchmarks.py
~~~~~~~~~~~~
Generic benchmark runner for jailbreak detection.

Usage
-----
python detect_benchmarks.py --cfg-path configs/jb_detect.yaml           \\
                    --dec-unique-id 20250717_…                 \\
                    --method jbshield   #  or: nsp

The YAML config format is **identical** to the one used by
detect_benchmarks_nsp.py.  The script re-uses all helper utils (data
loading, logging, metric aggregation) so you get the same JSON
report in output/.

Dependencies
------------
• jbshield_core.py     (must be import-able)
• your utils.model_utils / models.encoder / decomposer
"""

from __future__ import annotations
import argparse, logging, sys, time, json, random
from pathlib import Path
import yaml, numpy as np, torch
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_recall_fscore_support)
from scipy.stats import chi2

# our libs
from utils.model_utils import load_model
from models.encoder import HFEncoder_notPooled
from models.decomposer import NonlinearDecomposer_tiny
from benchmarks.jbshield_core import JBShieldDetector         # ← new
from jailbreak_detect_nsp import _evaluate                # reuse metric helper

LOGGER_NAME = "detect_benchmarks"
RESULTS = []

# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--dec-unique-id", required=False)
    p.add_argument("--cfg-unique-id")
    p.add_argument("--method", choices=["nsp", "jbshield"], default="jbshield")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()

# --------------------------------------------------------------------------- #
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip() and not l.startswith("#")]

# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    # ---------- logging ----------
    Path("logs").mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"logs/detect_{args.method}_{ts}_{args.dec_unique_id}.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s — %(levelname)s — %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_file, mode="w")])
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Log → %s", log_file)

    # ---------- cfg ----------
    with open(args.cfg_path) as f:
        CFG = yaml.safe_load(f)

    if args.method == "nsp":
        if args.cfg_unique_id is None:
            args.cfg_unique_id = args.dec_unique_id
        with open(f"output/config_{args.cfg_unique_id}.yaml") as f:
            CFG_OUT = yaml.safe_load(f)
        enc_name = CFG_OUT["model"]["name"]
    else:  # e.g., jbshield
        enc_name = CFG["model"]["name"]

    # ---------- data ----------
    paths = CFG["data"]
    AR = {k: load_jsonl(v) for k, v in paths.items()}
    benign_id   = AR["Fb_id"] + AR["Gb_id"]
    jail_id     = AR["F_id" ] + AR["G_id" ]
    m = min(len(benign_id), len(jail_id))
    benign_id, jail_id = random.sample(benign_id, m), random.sample(jail_id, m)
    benign_ood  = AR["Fb_ood"] + AR["Gb_ood"]
    jail_ood    = AR["F_ood" ] + AR["G_ood" ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_name = CFG_OUT["model"]["name"]
    logger.info("Method: %s   Encoder: %s", args.method, enc_name)

    if args.method == "jbshield":
        run_jbshield(enc_name, benign_id, jail_id, benign_ood, jail_ood,
                     args.batch_size, logger)
    else:
        from jailbreak_detect_nsp import detect_worker as nsp_worker
        ckpt_dir = Path("checkpoints/decomposer_simple/")
        nsp_worker(device, enc_name, ckpt_dir,
                   benign_id, jail_id, benign_ood, jail_ood,
                   args.batch_size, logger)

# --------------------------------------------------------------------------- #
def run_jbshield(enc_name, ben_id, jb_id, ben_ood, jb_ood,
                 batch_size, logger):
    # ------------- calibration -----------------
    N_CAL = 30
    cal_ben = random.sample([e["prompt"] for e in ben_id], N_CAL)
    cal_har = random.sample([e["prompt"] for e in jb_id ], N_CAL)  # harmful≈jail ID
    cal_jb  = random.sample([e["prompt"] for e in jb_id ], N_CAL)

    det = JBShieldDetector(enc_name, device="cuda" if torch.cuda.is_available() else "cpu",
                           batch_size=batch_size)
    det.calibrate(cal_ben, cal_har, cal_jb)

    # ------------- scoring splits --------------
    for split_tag, B, J in [("ID" , ben_id , jb_id ),
                           ("OOD", ben_ood, jb_ood)]:
        y_true = np.concatenate([np.zeros(len(B)), np.ones(len(J))])
        preds  = det.predict([e["prompt"] for e in B+J])
        auroc  = roc_auc_score(y_true, preds)   # binary predictions here
        acc    = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
                              y_true, preds, average="binary", zero_division=0)
        logger.info("%s  |  Acc %.3f  F1 %.3f  AUROC %.3f  Prec %.3f  Rec %.3f",
                    split_tag, acc, f1, auroc, prec, rec)
        RESULTS.append({"method": "jbshield", "split": split_tag,
                        "acc": acc, "f1": f1, "auroc": auroc,
                        "precision": prec, "recall": rec})

    out_path = Path("output") / f"detect_metrics_jbshield_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    logger.info("Saved metrics → %s", out_path)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
