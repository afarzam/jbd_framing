"""
"""

from __future__ import annotations
import torch, torch.nn.functional as F
from typing import Optional, Callable, Dict
from tqdm import tqdm
import random
import inspect

from pathlib import Path


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


# --------------------------------------------------------------------------- #
# helpers (kept local; feel free to centralise in decomp.models.losses later)
# --------------------------------------------------------------------------- #
def _l2_recon(v: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v_hat, v)

def _orth_penalty(vg: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
    # minimise squared cosine similarity
    return ((vg * vf).sum(-1) ** 2).mean()

def _info_nce(anchor: torch.Tensor,
              positive: torch.Tensor,
              negatives: torch.Tensor,
              T: float = 0.1) -> torch.Tensor:
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negatives, dim=-1)

    logits_pos = (a * p).sum(-1, keepdim=True) / T          # (B,1)
    logits_neg = a @ n.t() / T                              # (B,B*K)
    logits     = torch.cat([logits_pos, logits_neg], dim=1) # (B,1+B*K)
    labels     = torch.zeros(len(a), dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, labels)


# --- this version excludes negative pairs that 
#               are not actually negative pairs
def _safe_info_nce(anchor: torch.Tensor,
                   positive: torch.Tensor,
                   all_vecs: torch.Tensor,
                   neg_mask: torch.Tensor,           # 1 = forbidden, 0 = allowed
                   T: float = 0.1) -> torch.Tensor:
    """
    • anchor, positive: (B, D) already sliced into pairs
    • all_vecs        : (N, D) all samples in current batch (N = 2B)
    • neg_mask        : (B, N) True where a vector may NOT be used as a negative
    """
    B, D = anchor.size()
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(all_vecs, dim=-1)                # (N,D)

    logits_pos = (a * p).sum(-1, keepdim=True) / T   # (B,1)

    logits_neg = a @ n.t() / T                      # (B,N)
    logits_neg = logits_neg.masked_fill(neg_mask, -1e4)

    logits = torch.cat([logits_pos, logits_neg], dim=1)  # (B,1+N)
    labels = torch.zeros(B, dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, labels)



# put this once, e.g. top of the file
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_in):
        ctx.lambda_in = lambda_in
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambda_in * g, None



# --------------------------------------------------------------------------- #
# main function
# --------------------------------------------------------------------------- #
def train_decomposer(
    *,
    encoder: torch.nn.Module,                      # frozen LLM encoder
    decomposer: torch.nn.Module,                   # LinearDecomposer or similar
    dataloader: torch.utils.data.DataLoader,       # yields texts (or tuples)
    optimizer: torch.optim.Optimizer,
    adv_clf,
    adv_opt,
    lambda_adv: float = 1.0,
    device: torch.device | str = "cuda",
    epochs: int = 10,
    starting_epoch: int = 0,
    lambda_g: float = 1.0,
    lambda_f: float = 1.0,
    lambda_repulse: float = 3.0,  # same weight as pulls
    lambda_orth: float = 0.1,
    lambda_recon: float = 1.0,
    lambda_Worth: float = 0.0,
    use_negative_queue: bool = False,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,     # accumulate gradients for this many steps
    layer_norm: bool = False,  # whether to apply layer norm to encoder output
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_every: int = 100,
    info: Callable[[str], None] = print,           # logger hook
    layer_str: str=' Unspecified',  # which layer is training
    cfg: Optional[Dict] = None,  # config dict for logging
    layer: Optional[int] = None,  # layer number for logging
    local_rank: int = -1,  # for distributed training, if applicable
    timestamp: Optional[str] = None,  # for logging
    run_id: Optional[str] = None,  # for logging
) -> None:
    """
    Minimal self-supervised training loop.

    *Expects each batch to contain (anchor, positive) pairs next to each other.*
    """
    device = torch.device(device)
    encoder.eval()                # should already be frozen/no-grad
    decomposer.to(device).train()

    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    stats = {"loss": []}
    if use_negative_queue:
        queue = None
        q_ptr, K = 0, 2048 # config['training']['neg_queue_size']  
    for epoch in range(starting_epoch, epochs):
        if hasattr(dataloader.dataset, "all_pairs"):
            random.shuffle(dataloader.dataset.all_pairs) # this is unnecessary because the shuffling is already happening elsewhere
        # dataloader.sampler.set_epoch(epoch) # this takes care of the shuffling at each epoch

        pbar = tqdm(dataloader, desc=f"epoch {epoch}, layer {layer_str}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            global_step += 1

            # Support either a pure text batch or (text, goal_id, …)
            texts = batch[0] if isinstance(batch, (tuple, list)) else batch

            # ---------------------------------------------------------------
            # forward pass
            # ---------------------------------------------------------------
            with torch.no_grad():
                reps = encoder(texts)                  # (2B, enc_dim)
                if layer_norm:
                    reps = torch.nn.functional.layer_norm(reps, reps.shape[-1:])


            with torch.cuda.amp.autocast():
                vg_all, vf_all, vhat_all = decomposer(reps)
                if len(vg_all.shape) == 3: # if not pooled across tokens already
                    vg_all = vg_all.mean(dim=1)  # (2B, d_g)
                    vf_all = vf_all.mean(dim=1)  # (2B, d_g)
                    
                    
                # ------------------ MoCo negative queue ------------------
                if use_negative_queue:
                    if K > 0:                               # queue enabled
                        if queue is None:                   # create on first batch
                            queue = torch.zeros(K, vg_all.size(-1), device=device)
                        # concat current batch + queue as negative bank
                        neg_bank = torch.cat([vg_all.detach(), queue], dim=0)
                    else:
                        neg_bank = vg_all.detach()
                # ---------------------------------------------------------

                
                # split back into pairs
                vg_a, vg_p = vg_all[0::2], vg_all[1::2]
                vf_a, vf_p = vf_all[0::2], vf_all[1::2]
                vhat_a     = vhat_all[0::2]

                pair_types = batch[3].to(device)       # 0 = goal, 1 = frame
                
                # ---------------------------------------------------------
                # build “forbidden-negative” masks
                gids_all = batch[1].to(device)         # (2B,)
                fids_all = batch[2].to(device)         # (2B,)

                same_goal  = gids_all.unsqueeze(0) == gids_all.unsqueeze(1)   # (2B,2B)
                same_frame = fids_all.unsqueeze(0) == fids_all.unsqueeze(1)   # (2B,2B)

                # keep only rows that correspond to anchors (0,2,4,…)
                mask_goal  = same_goal [0::2]   # shape (B, 2B)
                mask_frame = same_frame[0::2]   # shape (B, 2B)
                # ---------------------------------------------------------

                # --------------------------------------------------------
                if use_negative_queue:
                    neg_bank = torch.cat([vg_all.detach(), queue], dim=0)   # (2B+K, D) 
                    # use the same 'neg_bank' in _safe_info_nce() calls below
                    # --- make mask_goal and mask_frame shapes match neg_bank
                    extra = neg_bank.size(0) - mask_goal.size(1)   # = K
                    if extra > 0:
                        pad = torch.zeros(mask_goal.size(0), extra,
                                        dtype=torch.bool, device=device)
                        mask_goal  = torch.cat([mask_goal,  pad], dim=1)
                        mask_frame = torch.cat([mask_frame, pad], dim=1)
                    # ---  
                else:
                    neg_bank = vg_all.detach()
                # --------------------------------------------------------

                # ---- build losses for every pair --------------------------
                L_recon = _l2_recon(reps[0::2], vhat_a)
                L_goal, L_frame, L_push = 0, 0, 0

                # mask for goal-pairs
                m_goal   = (pair_types == 0)
                # if m_goal.any():
                #     L_goal  = _info_nce(
                #                 vg_a[m_goal], vg_p[m_goal], vg_all.detach())
                #     # *repel* vf for same-goal pairs
                #     L_push  = _info_nce(
                #                 vf_a[m_goal], vf_p[m_goal], vf_all.detach()) * (-1)

                # # mask for frame-pairs
                m_frame  = (pair_types == 1)
                # if m_frame.any():
                #     L_frame = _info_nce(
                #                 vf_a[m_frame], vf_p[m_frame], vf_all.detach())
                #     # *repel* vg for same-frame pairs
                #     L_push += _info_nce(
                #                 vg_a[m_frame], vg_p[m_frame], vg_all.detach()) * (-1)
                if m_goal.any():
                    L_goal = _safe_info_nce(
                        vg_a[m_goal], vg_p[m_goal], neg_bank, 
                        neg_mask = mask_goal[m_goal]         # forbid same-goal negatives
                    )
                    L_push = -_safe_info_nce(
                        vf_a[m_goal], vf_p[m_goal], neg_bank, 
                        neg_mask = mask_goal[m_goal]
                    )

                if m_frame.any():
                    L_frame = _safe_info_nce(
                        vf_a[m_frame], vf_p[m_frame], neg_bank, 
                        neg_mask = mask_frame[m_frame]       # forbid same-frame negatives
                    )
                    L_push += -_safe_info_nce(
                        vg_a[m_frame], vg_p[m_frame], neg_bank, # TODO: replaced vg_all.detach() with neg_bank
                        neg_mask = mask_frame[m_frame]
                    )

                    
                    
                # ----- adversarial classifier
                gids = batch[1].to(device)          

                logits_clf = adv_clf(vf_all.detach())          # train classifier
                loss_clf   = F.cross_entropy(logits_clf, gids)

                logits_adv = adv_clf(GradReverse.apply(vf_all, 1.0))  # reverse gradient
                loss_adv   = F.cross_entropy(logits_adv, gids)
                # -----

                loss = (lambda_recon * L_recon +
                        lambda_g   * L_goal +
                        lambda_f   * L_frame +
                        lambda_repulse   * L_push +          # same weight as pulls
                        lambda_orth* _orth_penalty(vg_a, vf_a)
                        + loss_clf
                        + lambda_adv * loss_adv)
                if lambda_Worth>0.0:
                    core = getattr(decomposer, "module", decomposer)
                    Wg = core.Wg[0].weight
                    Wf = core.Wf[0].weight
                    L_Worth = torch.norm(Wg.T @ Wf, p='fro')**2
                    loss += lambda_Worth * L_Worth

                
                stats["loss"].append(loss.item())


            loss = loss / grad_accum_steps 
            scaler.scale(loss).backward()
            # # -- alternative 1
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(decomposer.parameters(), grad_clip)
            # scaler.step(optimizer)
            # scaler.update() # decomposer step
            # adv_opt.zero_grad(set_to_none=True)     # clear earlier grads
            # loss_clf.backward(retain_graph=True)    # only classifier params have grads
            # adv_opt.step() # adversarial classifier step
            # optimizer.zero_grad()
            # -- alternative 2
            if global_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                scaler.unscale_(adv_opt)
                torch.nn.utils.clip_grad_norm_(decomposer.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(adv_clf.parameters(), grad_clip)
                scaler.step(optimizer)    # decomposer
                scaler.step(adv_opt)      # adversarial classifier
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                adv_opt.zero_grad(set_to_none=True)
            
                if scheduler is not None:
                    scheduler.step()
            
            if use_negative_queue:
                if K > 0 and global_step % grad_accum_steps == 0: # TODO: Uncomment this when all is fixed
                    with torch.no_grad():
                        k = vg_all.size(0)             # usually 2B
                        if k >= K:                     # batch larger than queue
                            queue[:] = vg_all[-K:].detach()
                            q_ptr = 0
                        else:
                            end = q_ptr + k
                            if end <= K:               # fits contiguously
                                queue[q_ptr:end] = vg_all.detach()
                            else:                      # wrap around
                                first = K - q_ptr
                                queue[q_ptr:]   = vg_all[:first].detach()
                                queue[:end-K]   = vg_all[first:].detach()
                            q_ptr = (q_ptr + k) % K



            if global_step % log_every == 0:
                info(f"layer {layer_str}, [{epoch}:{batch_idx}], loss={loss.item():.4f}")

            pbar.set_postfix(loss=float(loss))
            
        if local_rank == 0:
            ck_root = Path(cfg["output"]["checkpoints_root"])
            model_short = MODEL_JOB_PREFIXES.get(cfg["model"]["name"], "unknownModel")
            ckpt_dir = ck_root / f"{model_short}_decomposer_layer{layer}_{timestamp}_{run_id}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / f"epoch_{epoch}").mkdir(parents=True, exist_ok=True)
            torch.save({k:v.cpu() for k,v in decomposer.module.state_dict().items()},
                       ckpt_dir / f"epoch_{epoch}" /"weights.pt")
            torch.save(optimizer.state_dict(),  ckpt_dir / f"epoch_{epoch}"  / "opt_epoch.pt")
            torch.save(adv_opt.state_dict(),  ckpt_dir / f"epoch_{epoch}"  / "adv_opt.pt")
            torch.save(scheduler.state_dict(),  ckpt_dir / f"epoch_{epoch}"  / "sched.pt")
            if scaler is not None:
                torch.save(scaler.state_dict(), ckpt_dir / f"epoch_{epoch}"  / "scaler.pt")

    # leave weights on whichever device caller wants next

    # if len(inspect.signature(train_decomposer).return_annotation) == 0:
    #     # No explicit return annotation, so check how function is called
    #     frame = inspect.currentframe()
    #     outer_frames = inspect.getouterframes(frame)
    #     # Try to detect if caller expects two outputs
    #     # This is a best-effort heuristic
    #     try:
    #         caller_locals = outer_frames[1].frame.f_locals
    #         caller_code = outer_frames[1].code_context[0] if outer_frames[1].code_context else ""
    #         if caller_code.count(",") >= 1:
    #             return stats, scaler
    #         else:
    #             return stats
    #     except Exception:
    #         return stats
    # else:
    #     return stats
    return stats, scaler




# --------------------------------------------------------------------------- #
# main function
# --------------------------------------------------------------------------- #
def train_decomposer_withSAE(
    *,
    encoder: torch.nn.Module,                      # frozen LLM encoder
    decomposer: torch.nn.Module,                   # LinearDecomposer or similar
    dataloader: torch.utils.data.DataLoader,       # yields texts (or tuples)
    optimizer: torch.optim.Optimizer,
    adv_clf,
    adv_opt,
    lambda_adv: float = 1.0,
    code_dim: int = 16384,  # sparse code dimension (Gemma-Scope SAE)
    device: torch.device | str = "cuda",
    epochs: int = 10,
    lambda_g: float = 1.0,
    lambda_f: float = 1.0,
    lambda_repulse: float = 3.0,  # same weight as pulls
    lambda_orth: float = 0.1,
    lambda_sparse: float = 0.1,  # sparsity penalty for SAE
    lambda_recon: float = 1.0,
    lambda_Worth: float = 0.0,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,     # accumulate gradients for this many steps
    layer_norm: bool = False,  # whether to apply layer norm to encoder output
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_every: int = 100,
    info: Callable[[str], None] = print,           # logger hook
) -> None:
    """
    Minimal self-supervised training loop.

    *Expects each batch to contain (anchor, positive) pairs next to each other.*
    """
    device = torch.device(device)
    encoder.eval()                # should already be frozen/no-grad
    decomposer.to(device).train()

    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    stats = {"loss": []}
    # queue = torch.zeros(4096,#config['training']['neg_queue_size'],
    #                     vg_all.size(-1), device=device) # TODO: fix this!! vg_all is not defined yet here!!
    # q_ptr = 0 # TODO: Uncomment this when all is fixed

    for epoch in range(epochs):
        if hasattr(dataloader.dataset, "all_pairs"):
            random.shuffle(dataloader.dataset.all_pairs) # this is unnecessary because the shuffling is already happening elsewhere
        # dataloader.sampler.set_epoch(epoch) # this takes care of the shuffling at each epoch

        pbar = tqdm(dataloader, desc=f"epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            global_step += 1

            # Support either a pure text batch or (text, goal_id, …)
            texts = batch[0] if isinstance(batch, (tuple, list)) else batch

            # ---------------------------------------------------------------
            # forward pass
            # ---------------------------------------------------------------
            with torch.no_grad():
                reps = encoder(texts)                  # (2B, enc_dim)
                if layer_norm:
                    reps = torch.nn.functional.layer_norm(reps, reps.shape[-1:])


            with torch.cuda.amp.autocast():
                vg_all, vf_all, vhat_all = decomposer(reps)

                # split back into pairs
                vg_a, vg_p = vg_all[0::2], vg_all[1::2]
                vf_a, vf_p = vf_all[0::2], vf_all[1::2]
                vhat_a     = vhat_all[0::2]

                pair_types = batch[3].to(device)       # 0 = goal, 1 = frame
                
                # ---------------------------------------------------------
                # build “forbidden-negative” masks
                gids_all = batch[1].to(device)         # (2B,)
                fids_all = batch[2].to(device)         # (2B,)

                same_goal  = gids_all.unsqueeze(0) == gids_all.unsqueeze(1)   # (2B,2B)
                same_frame = fids_all.unsqueeze(0) == fids_all.unsqueeze(1)   # (2B,2B)

                # keep only rows that correspond to anchors (0,2,4,…)
                mask_goal  = same_goal [0::2]   # shape (B, 2B)
                mask_frame = same_frame[0::2]   # shape (B, 2B)
                
                # ---
                if config['training'].get('use_neg_queue', False):
                    extra = neg_bank.size(0) - mask_goal.size(1)   # = K
                    if extra > 0:
                        pad = torch.zeros(mask_goal.size(0), extra,
                                        dtype=torch.bool, device=device)
                        mask_goal  = torch.cat([mask_goal,  pad], dim=1)
                        mask_frame = torch.cat([mask_frame, pad], dim=1)
                # ----------------------------------------------------------

                # ---------------------------------------------------------


                # ---- build losses for every pair --------------------------
                code = reps[:, -code_dim:]            # (2B, K)
                L_recon  = _l2_recon(reps[0::2], vhat_a)
                L_sparse = code.abs().mean()
                L_goal, L_frame, L_push = 0, 0, 0

                # mask for goal-pairs
                m_goal   = (pair_types == 0)

                # # mask for frame-pairs
                m_frame  = (pair_types == 1)
                if m_goal.any():
                    L_goal = _safe_info_nce(
                        vg_a[m_goal], vg_p[m_goal], vg_all.detach(),
                        neg_mask = mask_goal[m_goal]         # forbid same-goal negatives
                    )
                    L_push = -_safe_info_nce(
                        vf_a[m_goal], vf_p[m_goal], vf_all.detach(),
                        neg_mask = mask_goal[m_goal]
                    )

                if m_frame.any():
                    L_frame = _safe_info_nce(
                        vf_a[m_frame], vf_p[m_frame], vf_all.detach(),
                        neg_mask = mask_frame[m_frame]       # forbid same-frame negatives
                    )
                    L_push += -_safe_info_nce(
                        vg_a[m_frame], vg_p[m_frame], vg_all.detach(),
                        neg_mask = mask_frame[m_frame]
                    )

                    
                    
                # ----- adversarial classifier
                gids = batch[1].to(device)          

                logits_clf = adv_clf(vf_all.detach())          # train classifier
                loss_clf   = F.cross_entropy(logits_clf, gids)

                logits_adv = adv_clf(GradReverse.apply(vf_all, 1.0))  # reverse gradient
                loss_adv   = F.cross_entropy(logits_adv, gids)
                # -----

                loss = (lambda_recon * L_recon +
                        lambda_sparse* L_sparse +
                        lambda_g   * L_goal +
                        lambda_f   * L_frame +
                        lambda_repulse   * L_push +          # same weight as pulls
                        lambda_orth* _orth_penalty(vg_a, vf_a)
                        + loss_clf
                        + lambda_adv * loss_adv)
                
                if lambda_Worth>0.0:
                    core = getattr(decomposer, "module", decomposer)
                    Wg = core.Wg[0].weight
                    Wf = core.Wf[0].weight
                    L_Worth = torch.norm(Wg.T @ Wf, p='fro')**2
                    loss += lambda_Worth * L_Worth
                
                stats["loss"].append(loss.item())


            loss = loss / grad_accum_steps 
            scaler.scale(loss).backward()
            
            # torch.nn.utils.clip_grad_norm_(decomposer.parameters(), grad_clip)
            # torch.nn.utils.clip_grad_norm_(adv_clf.parameters(), grad_clip)
            if global_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                scaler.unscale_(adv_opt)
                torch.nn.utils.clip_grad_norm_(decomposer.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(adv_clf.parameters(), grad_clip)
                scaler.step(optimizer)    # decomposer
                scaler.step(adv_opt)      # adversarial classifier
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                adv_opt.zero_grad(set_to_none=True)
            
                if scheduler is not None:
                    scheduler.step()

            if global_step % log_every == 0:
                info(f"[{epoch}:{batch_idx}] loss={loss.item():.4f}")

            pbar.set_postfix(loss=float(loss))


    return stats


