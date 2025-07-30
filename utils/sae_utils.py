
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import torch
from utils.model_utils import load_model
import torch.nn as nn
import warnings
import re

HF_token = "YOUR_HUGGING_FACE_TOKEN_HERE"



from huggingface_hub import hf_hub_download
import numpy as np, torch, torch.nn as nn

__all__ = ["JumpReLUSAE", "get_gemma_sae"]

# --------------------------------------------------------------------- #
# basic Jump-ReLU SAE (same shapes as in Gemma-Scope release)
# --------------------------------------------------------------------- #
class JumpReLUSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    # @torch.no_grad()
    def encode(self, resid: torch.Tensor) -> torch.Tensor:          # resid (B, D)
        pre = resid @ self.W_enc + self.b_enc                       # (B, K)
        return (pre > self.threshold) * torch.relu(pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

# --------------------------------------------------------------------- #
# loader – returns a frozen SAE living on <device>
# --------------------------------------------------------------------- #
# def get_gemma_sae(device: torch.device,
#                   repo: str = "google/gemma-scope-2b-pt-res",
#                   npz:  str = "layer_20/width_16k/average_l0_71/params.npz"
# ) -> JumpReLUSAE:
def get_gemma_sae(device: torch.device,
                  repo="google/gemma-scope-2b-pt-res",
                  npz="layer_20/width_16k/average_l0_71/params.npz",
                  trainable: bool = False) -> JumpReLUSAE:
    path = hf_hub_download(repo_id=repo, filename=npz, force_download=False)
    params = np.load(path)
    sae = JumpReLUSAE(params["W_enc"].shape[0], params["W_enc"].shape[1]).to(device)
    sae.load_state_dict({k: torch.from_numpy(v).to(device) for k, v in params.items()})
    # sae.eval() ; sae.requires_grad_(False)
    if not trainable:
        sae.eval().requires_grad_(False)
    else:
        sae.train().requires_grad_(True)
    return sae


# def load_multiple_saes(layers, device, trainable=False):
#     """
#     Returns {layer: JumpReLUSAE} ordered dict.
#     All SAEs share the same repo/npz naming scheme as Gemma-Scope.
#     """
#     saes = {}
#     for l in layers:
#         npz = f"layer_{l}/width_16k/average_l0_71/params.npz"
#         saes[l] = get_gemma_sae(device, npz=npz, trainable=trainable)
#     return saes


# utils/sae_utils.py
from huggingface_hub import list_repo_files
_L0_RE = re.compile(r"average_l0_(\d+)")
_TARGET_L0 = 70                      # aim for sparsity ≈ 7 %

def _pick_npz_for_layer(layer: int, files):
    """
    Among repo files, choose the average_l0_* sub-folder closest to TARGET_L0
    for this layer.  Returns the NPZ path string or None if layer absent.
    """
    candidates = []
    for f in files:
        if f.startswith(f"layer_{layer}/width_16k/average_l0_") and f.endswith("/params.npz"):
            m = _L0_RE.search(f)
            if m:
                l0 = int(m.group(1))
                candidates.append((abs(l0 - _TARGET_L0), l0, f))
    if not candidates:
        return None
    candidates.sort()        # smallest |Δ| first, ties resolved by lower l0
    return candidates[0][2]  # best-path


from huggingface_hub.utils import EntryNotFoundError
def load_multiple_saes(layers, device, trainable=False,
                       repo="google/gemma-scope-2b-pt-res"):
    """
    Return {layer: SAE}.  Picks, for each requested layer, the average_l0_*
    checkpoint whose sparsity is closest to 70 if several exist.
    Falls back to the *canonical* release if the whole layer is missing.
    """
    all_files = set(list_repo_files(repo))
    saes = {}
    for layer in layers:
        npz = _pick_npz_for_layer(layer, all_files)

        # ------ fall back to canonical release if layer missing -------------
        if npz is None:
            print(f"\n\nCould not find repo for layer {layer} \n\n")
            canon_repo = f"{repo}-canonical"
            try:
                npz = f"layer_{layer}/width_16k/canonical/params.npz"
                hf_hub_download(repo_id=canon_repo, filename=npz, 
                                force_download=False, token=HF_token)
                repo_to_use = canon_repo
            except EntryNotFoundError:
                print(f"[warn] No SAE found for layer {layer}; skipping.")
                continue
        else:
            repo_to_use = repo

        saes[layer] = get_gemma_sae(
            device,
            repo=repo_to_use,
            npz=npz,
            trainable=trainable,
        )
        print(f"Loaded layer {layer} SAE  ({repo_to_use}/{npz})")

    if not saes:
        raise ValueError("None of the requested SAE layers were found.")
    return saes



# from huggingface_hub.utils import EntryNotFoundError
# def load_multiple_saes(layers, device, trainable=False):
#     """
#     Return {layer: JumpReLUSAE}.  Tries the "average_l0_71" files first,
#     then falls back to the canonical release if a layer is missing there.
#     """
#     saes = {}
#     for l in layers:
#         loaded = False

#         # --- 1) average_l0_71 file in the original repo ------------------
#         try:
#             npz = f"layer_{l}/width_16k/average_l0_71/params.npz"
#             saes[l] = get_gemma_sae(
#                 device,
#                 repo="google/gemma-scope-2b-pt-res",
#                 npz=npz,
#                 trainable=trainable,
#             )
#             loaded = True
#         except EntryNotFoundError:
#             pass   # silently fall through

#         # --- 2) canonical file in the *-canonical repo -------------------
#         if not loaded:
#             try:
#                 npz = f"layer_{l}/width_16k/canonical/params.npz"
#                 saes[l] = get_gemma_sae(
#                     device,
#                     repo="google/gemma-scope-2b-pt-res-canonical",
#                     npz=npz,
#                     trainable=trainable,
#                 )
#                 loaded = True
#             except EntryNotFoundError:
#                 pass

#         # --- 3) still not found → warn ----------------------------------
#         if not loaded:
#             print(f"[warn] SAE for layer {l} not found in either release; skipping.")

#     if not saes:
#         raise ValueError("None of the requested SAE layers were found.")
#     return saes




# ===================================
# class GemmaSAEWrapper(nn.Module):
#     def __init__(self, 
#                  sae, model, device,
#                  max_new_tokens=50):
#         super().__init__()
#         self.sae = sae.eval()  # keep frozen for now
#         self.model = model.eval()
#         self.device = device
#         self.max_new_tokens=max_new_tokens
#         # torch.set_grad_enabled(False) # avoid blowing up mem
#         # warnings.filterwarnings("ignore", category=UserWarning, message="setting torch's float32_matmul_precision to 'high'")
#         # torch.set_float32_matmul_precision('high')
        
#     @torch.no_grad()
#     def gather_residual_activations(model, target_layer, inputs):
#         target_act = None
#         def gather_target_act_hook(mod, inputs, outputs):
#             nonlocal target_act # make sure we can modify the target_act from the outer scope
#             target_act = outputs[0]
#             return outputs
#         handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
#         _ = model.forward(inputs)
#         handle.remove()
#         return target_act
    
    
#     @torch.no_grad()
#     def forward(self, prompt=None, resids=None, target_layer=20):
#         # Pass it in to the model and generate text
#         if resids is None:
#             inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
#             resids = self.gather_residual_activations(self.model, target_layer, inputs)
            
#         sae_acts = self.sae.encode(resids.to(torch.float32))
#         # recon = self.sae.decode(sae_acts)
        
#         return sae_acts#, recon  

    
    
# class JumpReLUSAE(nn.Module):
#     def __init__(self, d_model, d_sae):
#         # Initialize model and tokenizer
        
#         # Note that we initialise these to zeros because we're loading in pre-trained weights.
#         # If you want to train your own SAEs then we recommend using blah
#         super().__init__()
#         self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
#         self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
#         self.threshold = nn.Parameter(torch.zeros(d_sae))
#         self.b_enc = nn.Parameter(torch.zeros(d_sae))
#         self.b_dec = nn.Parameter(torch.zeros(d_model))

#     def encode(self, input_acts):
#         pre_acts = input_acts @ self.W_enc + self.b_enc
#         mask = (pre_acts > self.threshold)
#         acts = mask * torch.nn.functional.relu(pre_acts)
#         return acts

#     def decode(self, acts):
#         return acts @ self.W_dec + self.b_dec

#     def forward(self, acts):
#         acts = self.encode(acts)
#         recon = self.decode(acts)
#         return recon



# def get_GemmaSAE(tokenizer, model, device):
#     torch.set_grad_enabled(False) # avoid blowing up mem

#     torch.set_float32_matmul_precision('high')

#     # # The input text
#     # prompt = "Would you be able to travel through time using a wormhole?"

#     # # Use the tokenizer to convert it to tokens. Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
#     # inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
#     # print(inputs)

#     # # Pass it in to the model and generate text
#     # outputs = model.generate(input_ids=inputs, max_new_tokens=50)
#     # print(tokenizer.decode(outputs[0]))


#     path_to_params = hf_hub_download(
#         repo_id="google/gemma-scope-2b-pt-res",
#         filename="layer_20/width_16k/average_l0_71/params.npz",
#         force_download=False,
#     )

#     params = np.load(path_to_params)
#     pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

#     sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
#     sae.load_state_dict(pt_params)
#     sae.to(device)
    
#     return sae

