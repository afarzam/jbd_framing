#!/usr/bin/env python3
"""
Example
-------
python train_decomposer.py --cfg-path configs/decomposer3.yaml --num-proc 4

# Resume training from a checkpoint using unique ID
python train_decomposer.py --cfg-path configs/decomposer3.yaml --num-proc 4 --checkpoint-unique-id 20250722_025458_074a110c-2578-4143-87f1-763b1c892868

# Resume training with optimizer state reconstruction (when only weights are available)
python train_decomposer.py --cfg-path configs/decomposer3.yaml --num-proc 4 --checkpoint-unique-id 20250722_025458_074a110c-2578-4143-87f1-763b1c892868 --original-config configs/decomposer_original.yaml

# Or specify checkpoint in config file
python train_decomposer.py --cfg-path configs/decomposer_with_checkpoint.yaml --num-proc 4
"""

from __future__ import annotations
import argparse, os, sys, time, uuid, yaml, logging, random, json
from pathlib import Path
unseeded_rng = random.Random() # this is to use for cases when we don't want to seed

import torch
import numpy as np
import torch.multiprocessing as mp
from accelerate import notebook_launcher            # spawns ranks

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decomposer multi-layer trainer")
    p.add_argument("--cfg-path", type=str, required=True,
                   help="YAML config file (updated version).")
    p.add_argument("--num-proc", type=int, default=4,
                   help="How many GPU ranks to launch.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Optional HF model name that overrides the one in the YAML.")
    p.add_argument("--sample-prop", type=float, default=None,
                   help="If set, randomly sample this proportion (0-1) of all_samples before training, with a minimum of 500.")
    p.add_argument("--checkpoint-unique-id", type=str, default=None,
                   help="Optional unique ID to resume training from checkpoints (e.g., 20250722_025458_074a110c-2578-4143-87f1-763b1c892868).")
    p.add_argument("--original-config", type=str, default=None,
                   help="Path to original training config used to create the checkpoint (for optimizer state reconstruction).")
    return p.parse_args()

# ---------------------------------------------------------------------
# Top-level helpers from notebook
# ---------------------------------------------------------------------
def load_jsonl(path: str):
    import json
    with open(path, "r") as f:
        return [json.loads(l) for l in f if l.strip() and not l.strip().startswith("#")]


# ------------------------------------------------------------------
# Global placeholders (seen by type-checkers & linters)
# They will be overwritten in main() before notebook_launcher starts.
# ------------------------------------------------------------------
CONFIG      = None
ALL_SAMPLES = None
RUN_ID      = ""
TIMESTAMP   = ""
LOGGER_NAME = "train_decomposer"
CHECKPOINT_UNIQUE_ID = None
ORIGINAL_CONFIG = None
MODEL_JOB_PREFIXES = {
    "meta-llama/Llama-3-8B-Instruct": "ll3",
    "meta-llama/Llama-2-7b-chat-hf": "ll2", 
    "lmsys/vicuna-13b-v1.5": "vic13",
    "lmsys/vicuna-7b-v1.5": "vic7",
    "mistralai/Mistral-7B-v0.1": "mis",
    "deepseek-ai/deepseek-llm-7b-chat": "dsk",
    "gpt2-medium": "gpt2",
    "google/gemma-2-9b": "gem9",
    "google/gemma-2-2b": "gem2"
}
# ------------------------------------------------------------------


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ---- basic env & logging ----
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    run_id = str(uuid.uuid4())
    ts     = time.strftime("%Y%m%d_%H%M%S")

    cfg_path = Path(args.cfg_path)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    # --- optional CLI override for the encoder model -----------------
    if args.model_name is not None:
        config["model"]["name"] = args.model_name

    # --- optional CLI override for checkpoint path -----------------
    if args.checkpoint_unique_id is not None:
        config["checkpoint_unique_id"] = args.checkpoint_unique_id
        
    # --- optional CLI override for checkpoint path -----------------
    if args.original_config is not None:
        config["original_config"] = args.original_config

    # log file
    log_root = Path(config["output"]["logs_dir"])
    log_root.mkdir(exist_ok=True)
    log_path = log_root / f"decomposer_{ts}_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_path, mode="w")]
    )
    logger = logging.getLogger("train_decomposer")
    logger.info("Run-ID %s  |  logs → %s", run_id, log_path)

    # ---- seeds ----
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ---- DATA (identical to notebook) ----
    data_cfg = config["data"]
    raw_F_harm  = load_jsonl(data_cfg["input_path_varyFraming"])
    raw_G_harm  = load_jsonl(data_cfg["input_path_varyGoal"])
    raw_F_ben   = load_jsonl(data_cfg["input_path_varyFraming_benign"])
    raw_G_ben   = load_jsonl(data_cfg["input_path_varyGoal_benign"])
    for e in raw_F_harm + raw_F_ben: e["split"] = "varyF"
    for e in raw_G_harm + raw_G_ben: e["split"] = "varyG"

    # --- preprocessing helper (unchanged) ---
    def _prep(entries, max_f_idx):
        out = []
        for ent in entries:
            if not all(k in ent for k in
                       ["prompt","goal","goal_index","framing_index","split"]):
                continue
            g,f = ent["goal_index"], ent["framing_index"]
            if ent["split"] == "varyF":
                f = g if f == 0 else max_f_idx + 1
                max_f_idx = max(max_f_idx, f)
            out.append({"text":ent["prompt"], "goal":ent["goal"],
                        "goal_index":g, "framing_index":f,
                        "label":ent.get("jailbroken", False),
                        "split":ent["split"]})
        return out, max_f_idx

    max_idx = max(e["framing_index"]
                  for e in raw_F_harm+raw_G_harm+raw_F_ben+raw_G_ben)
    P_F_harm, max_idx = _prep(raw_F_harm, max_idx)
    P_G_harm, max_idx = _prep(raw_G_harm, max_idx)
    P_F_ben , max_idx = _prep(raw_F_ben , max_idx)
    P_G_ben , max_idx = _prep(raw_G_ben , max_idx)
    all_samples = P_F_harm + P_G_harm + P_F_ben + P_G_ben
    logger.info("Total samples: %d", len(all_samples))

    # Optionally subsample all_samples if --sample-prop is set
    if args.sample_prop is not None:
        n_total = len(all_samples)
        n_sample = int(n_total * args.sample_prop)
        if n_sample < 500:
            logger.warning(f"Requested sample size {n_sample} is less than 500. Using 500 or all samples instead (which is smaller).")
            n_sample = 500
        if n_sample >= n_total:
            logger.warning(f"Requested sample size {n_sample} (from proportion {args.sample_prop}) is greater than or equal to total samples {n_total}. Using all samples.")
        else:
            all_samples = random.sample(all_samples, n_sample)
            logger.info(f"Randomly sampled {n_sample} samples (proportion {args.sample_prop}) from all_samples.")

    # # Hold big structures in global scope for workers
    # globals()["CONFIG"]       = config
    # globals()["ALL_SAMPLES"]  = all_samples
    # globals()["RUN_ID"]       = RUN_ID
    # globals()["TIMESTAMP"]    = ts
    # globals()["LOGGER_NAME"]  = "train_decomposer"
    global CONFIG, ALL_SAMPLES, RUN_ID, TIMESTAMP, LOGGER_NAME, MODEL_JOB_PREFIXES, CHECKPOINT_UNIQUE_ID, ORIGINAL_CONFIG
    CONFIG      = config
    ALL_SAMPLES = all_samples
    RUN_ID      = run_id
    TIMESTAMP   = ts
    LOGGER_NAME = "train_decomposer"
    CHECKPOINT_UNIQUE_ID = args.checkpoint_unique_id or config.get("checkpoint_unique_id")
    ORIGINAL_CONFIG = args.original_config or config.get("original_config")

    # ---- launch identical worker from notebook ----
    free_port = 10000 + unseeded_rng.randint(0, 50000)
    os.environ["MAIN_PROCESS_PORT"] = str(free_port) 
    os.environ.setdefault("MASTER_PORT", str(free_port))
    notebook_launcher(train_worker, num_processes=args.num_proc,
                      use_port=str(free_port))

# ---------------------------------------------------------------------
# Worker (verbatim from notebook minus cell magics)
# ---------------------------------------------------------------------
def train_worker():
    """
    This is *exactly* the body from the notebook’s worker cell, just
    transplanted.  Any change to logic will change results, so leave it
    be.
    """
    import gc, random, yaml, json, numpy as np, torch
    import torch.distributed as dist
    from pathlib import Path
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    from accelerate.utils import set_seed

    from collections import defaultdict
    # local imports identical to notebook
    from train_test.decomposer_training import train_decomposer
    from utils.model_utils import load_model_multiGPU
    from models.encoder import HFEncoder_notPooled
    from models.decomposer import NonlinearDecomposer
    from utils.misc import seed_worker, set_seed


    logger = logging.getLogger(LOGGER_NAME)
    config = CONFIG
    all_samples = ALL_SAMPLES
    checkpoint_unique_id = CHECKPOINT_UNIQUE_ID
    original_config_path = ORIGINAL_CONFIG

    # ——— build dataset helpers (identical) —
    class DualPairDataset(Dataset):
        def __init__(self, samples, stratified_capping=True):
            self.samples = samples
            self.goal_pairs, self.frame_pairs = [], []
            by_goal_F, by_frame_G = defaultdict(list), defaultdict(list)
            for idx,s in enumerate(samples):
                if s["split"]=="varyF": by_goal_F [s["goal_index"]].append(idx)
                else:                   by_frame_G[s["framing_index"]].append(idx)
            for lst in by_goal_F.values():
                self.goal_pairs += [(a,b,0) for a in lst for b in lst if a<b]
            for lst in by_frame_G.values():
                self.frame_pairs += [(a,b,1) for a in lst for b in lst if a<b]
            if stratified_capping:           # same heuristic
                cap = int(np.median([len(v) for v in by_goal_F.values()]))
                for g,lst in by_goal_F.items():
                    if len(lst) > cap:
                        by_goal_F[g] = random.sample(lst, cap)
            self.all_pairs = self.goal_pairs + self.frame_pairs
        def __len__(self):  return len(self.all_pairs)
        def __getitem__(self, k): return self.all_pairs[k]

    def collate_dual(batch):
        texts,gid,fid,ptype=[],[],[],[]
        for a,b,t in batch:
            sa,sb = all_samples[a], all_samples[b]
            texts.extend([sa["text"], sb["text"]])
            gid.extend([sa["goal_cid"], sb["goal_cid"]])
            fid.extend([sa["framing_index"], sb["framing_index"]])
            ptype.append(t)
        return texts, torch.tensor(gid), torch.tensor(fid), torch.tensor(ptype)

    # contiguous goal-ids
    unique_goals = sorted({s["goal_index"] for s in all_samples})
    goal2cid = {g:i for i,g in enumerate(unique_goals)}
    for s in all_samples: s["goal_cid"] = goal2cid[s["goal_index"]]
    train_ds = DualPairDataset(all_samples)

    # ------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = int(world_size) > 1
    if distributed:
        dist.init_process_group("nccl", init_method="env://",
                                rank=local_rank, world_size=int(world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed = config["experiment"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- load LLM & tokenizer ----
    model, tok = load_model_multiGPU(
        model_name=config["model"]["name"],
        local_rank=local_rank,
        load_in_8bit=False, load_in_4bit=False
    )
    if config["model"]["layers"] == 'all':
        num_layers = model.config.num_hidden_layers
        layers = list(range(num_layers))
    else:
        layers = config["model"]["layers"]
        if isinstance(layers, int): layers = [layers]
    logger.info("Loaded model %s ", config["model"]["name"])

    # notebook hard-coded overrides
    cfg = config
    # cfg["training"]["num_epochs"] = 3
    # cfg["training"]["batch_size"] = 8
    # cfg["training"]["grad_accum_steps"]       = 8
    # cfg["model"]["layers"]        = layers
    # cfg["model"]["last_token"]    = False
    # cfg["model"]["layer_combine"] = cfg["model"].get("layer_combine","mean")
    # cfg["lambda_Worth"]           = cfg.get("lambda_Worth",0.05)

    # one decomposer per layer
    for layer in layers:
        logger.info("Rank %d  |  training layer %d", local_rank, layer)
        encoder = HFEncoder_notPooled(
            model=model, tokenizer=tok, device=device,
            layers=[layer],
            layer_combine=cfg["model"]["layer_combine"],
            last_token=cfg["model"]["last_token"]
        ); encoder.eval()

        dec = NonlinearDecomposer(
            enc_dim=model.config.hidden_size,
            d_g=cfg["d_g"], d_f=cfg["d_f"],
            hidden_dim=cfg.get("hidden_dim",1024),
            dropout=cfg.get("dropout",0.1)
        ).to(device)
        
        # Load checkpoint if specified
        need_optimizer_reconstruction = False
        if checkpoint_unique_id is not None:
            # Construct checkpoint path for this specific layer
            model_short = MODEL_JOB_PREFIXES.get(config["model"]["name"], "unknownModel")
            checkpoint_dir = Path(cfg["output"]["checkpoints_root"]) / f"{model_short}_decomposer_layer{layer}_{checkpoint_unique_id}"
            checkpoint_path = checkpoint_dir / "weights.pt"
            
            logger.info("Looking for checkpoint at: %s", checkpoint_path)
            
            if checkpoint_path.exists():
                logger.info("Loading checkpoint from: %s", checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Check if this is a full checkpoint with optimizer state or just weights
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint with optimizer state
                    state_dict = checkpoint['model_state_dict']
                    logger.info("Found full checkpoint with optimizer state")
                    # Note: We could also load optimizer state here if needed
                    # opt.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    # Just weights - we'll need to reconstruct optimizer state
                    state_dict = checkpoint
                    need_optimizer_reconstruction = True
                    logger.info("Found weights-only checkpoint, will reconstruct optimizer state")
                
                # Handle both regular state dict and DDP state dict formats
                if all(key.startswith('module.') for key in state_dict.keys()):
                    # DDP format - remove 'module.' prefix
                    state_dict = {key[7:]: value for key, value in state_dict.items()}
                
                dec.load_state_dict(state_dict)
                logger.info("Successfully loaded checkpoint weights")
            else:
                logger.warning("Checkpoint path does not exist: %s", checkpoint_path)
        else:
            logger.info("No checkpoint specified, starting training from scratch")
        
        d_f = dec.Wf(torch.randn(model.config.hidden_size,device=device)).shape[0]

        gen = torch.Generator()
        gen.manual_seed(config['experiment']['seed']) 
        if distributed:
            dec = DDP(dec, device_ids=[local_rank])
            sampler = DistributedSampler(train_ds, rank=local_rank,
                                         num_replicas=world_size, shuffle=True, seed=config['experiment']['seed'])
            loader  = DataLoader(train_ds,
                                 batch_size=cfg["training"]["batch_size"],
                                 sampler=sampler, collate_fn=collate_dual,
                                 num_workers=8, pin_memory=True, shuffle=False,
                                 worker_init_fn=lambda wid: set_seed(seed=config['experiment']['seed'] + wid),#seed_worker, 
                                 generator=gen)
        else:
            dec.__dict__["module"] = dec
            sampler = None
            loader  = DataLoader(train_ds,
                                 batch_size=cfg["training"]["batch_size"],
                                 shuffle=True, collate_fn=collate_dual,
                                 num_workers=8, pin_memory=True,
                                 worker_init_fn=lambda wid: set_seed(seed=config['experiment']['seed'] + wid),#seed_worker, 
                                 generator=gen)

        opt   = torch.optim.AdamW(dec.parameters(), lr=cfg["lr"])
        sched = CosineAnnealingLR(opt, T_max=len(train_ds)*cfg["training"]["num_epochs"])
        
        # Try to load optimizer and scheduler state if available
        if checkpoint_unique_id is not None:
            # Use the same checkpoint directory we constructed above
            opt_path = checkpoint_dir / "opt.pt"
            sched_path = checkpoint_dir / "sched.pt"
            
            logger.info("Looking for optimizer state at: %s", opt_path)
            logger.info("Looking for scheduler state at: %s", sched_path)
            
            if opt_path.exists():
                logger.info("Loading optimizer state from: %s", opt_path)
                opt.load_state_dict(torch.load(opt_path, map_location=device))
                logger.info("Successfully loaded optimizer state")
            else:
                need_optimizer_reconstruction = True
                
            if sched_path.exists():
                logger.info("Loading scheduler state from: %s", sched_path)
                sched.load_state_dict(torch.load(sched_path, map_location=device))
                logger.info("Successfully loaded scheduler state")
        
        # Reconstruct optimizer state if needed
        if need_optimizer_reconstruction and original_config_path is not None:
            from utils.misc import reconstruct_optimizer_state
            try:
                reconstruct_optimizer_state(
                    optimizer=opt,
                    scheduler=sched,
                    original_config_path=original_config_path,
                    current_config=cfg,
                    dataset_size=len(train_ds),
                    logger=logger
                )
            except Exception as e:
                logger.warning("Failed to reconstruct optimizer state: %s", str(e))
                logger.info("Continuing with fresh optimizer state")
        
        n_goals = len(unique_goals)
        adv_clf = torch.nn.Linear(d_f, n_goals).to(device)
        if distributed:
            adv_clf = DDP(adv_clf, device_ids=[local_rank])
        adv_opt = torch.optim.AdamW(adv_clf.parameters(), lr=1e-4)

        stats, scaler = train_decomposer(
            encoder = encoder, 
            decomposer = dec, 
            dataloader=loader, 
            optimizer=opt,
            adv_clf=adv_clf, 
            adv_opt=adv_opt,
            lambda_adv=cfg["lambda_adv"],
            scheduler=sched, device=device,
            epochs=cfg["training"]["num_epochs"],
            starting_epoch=cfg["training"].get("starting_epoch", 0),
            lambda_g=cfg["lambda_g"], lambda_f=cfg["lambda_f"],
            lambda_repulse=cfg["lambda_repulse"],
            lambda_orth=cfg["lambda_orth"]*10,
            lambda_recon=cfg["lambda_recon"],
            lambda_Worth=cfg["lambda_Worth"],
            grad_clip=cfg["grad_clip"],
            grad_accum_steps=cfg["training"]["grad_accum_steps"],
            log_every=50, info=logger.info,
            layer_str=str(layer),
            cfg = config,  # config dict for logging
            layer = layer,  # layer number for logging
            local_rank = local_rank,  # for distributed training, if applicable
            timestamp = TIMESTAMP,  # for logging
            run_id = RUN_ID,  # for logging
        )

        if local_rank == 0:
            ck_root = Path(cfg["output"]["checkpoints_root"])
            model_short = MODEL_JOB_PREFIXES.get(config["model"]["name"], "unknownModel")
            ckpt_dir = ck_root / f"{model_short}_decomposer_layer{layer}_{TIMESTAMP}_{RUN_ID}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({k:v.cpu() for k,v in dec.module.state_dict().items()},
                       ckpt_dir/"weights.pt")
            torch.save(opt.state_dict(),  ckpt_dir / "opt.pt")
            torch.save(adv_opt.state_dict(),  ckpt_dir / "adv_opt.pt")
            torch.save(sched.state_dict(),  ckpt_dir / "sched.pt")
            if scaler is not None:
                torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")
            with open(ckpt_dir/"train_stats.json","w") as f: json.dump(stats,f)
            if layer == 0:        # save config snapshot once
                snap = Path(cfg["output"]["config_snapshot_dir"])
                snap.mkdir(exist_ok=True)
                with open(snap/f"config_{model_short}_{TIMESTAMP}_{RUN_ID}.yaml","w") as f:
                    yaml.safe_dump(cfg,f)
            logger.info("Checkpoint layer %d → %s", layer, ckpt_dir)
            
            
            # ---- update run_info.json ------------------------------------------------
            info_path = Path("output/run_info.json")
            info_path.parent.mkdir(exist_ok=True)
            # 1) load or create the nested dict
            if info_path.exists():
                with open(info_path, "r") as f:
                    run_info = json.load(f)
            else:
                run_info = {}
            # 2) ensure outer key for this RUN_ID
            run_rec = run_info.setdefault(RUN_ID, {
                "timestamp": TIMESTAMP,
                "encoder_model": config["model"]["name"],
                "config_snapshot": str((Path(config["output"]["config_snapshot_dir"])
                                        / f"config_{TIMESTAMP}_{RUN_ID}.yaml").resolve()),
                "layers": {}
            })
            # 3) record this layer’s checkpoint directory
            run_rec["layers"][f"layer{layer}"] = str(ckpt_dir.resolve())
            # 4) write back
            with open(info_path, "w") as f:
                json.dump(run_info, f, indent=2)
            # -------------------------------------------------------------------------


    if distributed: dist.destroy_process_group()

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
