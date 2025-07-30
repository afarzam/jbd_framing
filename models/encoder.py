import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.attn_pool import AttnPool1D
import warnings


class FrozenHFEncoder(torch.nn.Module):
    """
    Wraps any causal-LM and returns one 4096-d sentence vector.
    • mean-pool last hidden state  (works for Llama, Vicuna, Yi, …)
    • no_grad  → still fully GPU-accelerated but parameters are frozen.
    """
    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=True, truncation=True, padding=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, use_auth_token=True)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        out = self.model(**toks, output_hidden_states=True)
        h = out.hidden_states[-1]          # (B, T, D)
        return h.mean(dim=1)               # (B, D)
    


# models/encoder.py   (NEW VERSION OF HFEncoder)
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.attn_pool import AttnPool1D

class HFEncoder(nn.Module):
    """
    Generic frozen encoder for any HF causal-LM (Llama-2/3, Yi, Vicuna, …)

    • adds a pad-token automatically if the tokenizer lacks one  
    • passes only the arguments every model accepts
    • optional AttnPool1D instead of simple mean-pool
    """
    def __init__(self,
                 model_name       : str | None = None,
                 model            = None,
                 tokenizer        = None,
                 device           : str = "cuda",
                 dtype            = torch.float16,
                 attn_pool        : bool = False,
                 layers           = "last",          
                 layer_combine    : str = "mean",
                 last_token       : bool = False,):   
        super().__init__()

        # --- load or re-use -------------------------------------------------
        if model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model     = AutoModelForCausalLM.from_pretrained(
                                model_name, torch_dtype=dtype,
                                low_cpu_mem_usage=True, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
            self.model     = model

        self.model.to(device).eval()
        self.device = device
        self.use_attn = attn_pool

        # Llama & friends ship without a pad token → patch it here
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token     = self.tokenizer.eos_token
            self.tokenizer.pad_token_id  = self.tokenizer.eos_token_id

        if self.use_attn:
            self.pool = AttnPool1D(self.model.config.hidden_size).to(device)
        
        
        # ---- select the layers to use
        if layers == "last":
            self.layer_ids = [-1]                    # last layer only
        else:                                        # e.g. [ -1, -3, -5 ]
            self.layer_ids = list(layers)

        self.layer_combine = layer_combine.lower()
        assert self.layer_combine in ("mean","attn")

        # nL      = len(self.layer_ids)
        self.out_dim   = self.model.config.hidden_size
        if self.layer_combine=="attn":
            self.layer_pool = AttnPool1D(self.out_dim).to(device)
            
        self.last_token = last_token
        

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            max_length=getattr(self.model.config, "max_position_embeddings", None)
        ).to(self.device)
        overflow = getattr(toks, "overflowing_tokens", None)
        if overflow is not None and overflow.numel() > 0:
            n_trunc = toks.overflowing_tokens.shape[0]
            max_len = getattr(self.model.config, "max_position_embeddings", None)
            warnings.warn(f"[HFEncoder] {n_trunc} prompts truncated to "
                          f"{self.tokenizer.model_max_length} tokens."
                          "\nRaising it to {max_len}...")
            self.tokenizer.model_max_length = max_len
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,          
                max_length=getattr(self.model.config, "max_position_embeddings", None)
            ).to(self.device)
            overflow = getattr(toks, "overflowing_tokens", None)
            if overflow is not None and overflow.numel() > 0:
                warnings.warn(f"STILL {n_trunc} prompts truncated to "
                            f"{max_len} tokens!")


        # Llama-3 forward() only accepts input_ids & attention_mask
        inputs = {"input_ids": toks.input_ids,
                  "attention_mask": toks.attention_mask}

        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        # last = out.hidden_states[-1]                    # (B, T, D)

        # if self.use_attn:
        #     return self.pool(last)                      # (B, D)
        # return last.mean(dim=1)                         # (B, D)
    
        hs   = out.hidden_states                     # tuple(len=L+1) (B,T,D)
        picked = [hs[i] for i in self.layer_ids]     # list[(B,T,D)]
        layer_stack = torch.stack(picked, 0)         # (nL, B, T, D)
    
        # First, pool across tokens
        if self.last_token:
            pooled_tokens = layer_stack[:, :, -1, :]
        elif self.use_attn:
            nL, B, T, D = layer_stack.shape
            # Reshape to (nL * B, T, D) to apply attention pooling
            reshaped_stack = layer_stack.reshape(nL * B, T, D)
            # The output will be (nL * B, D), which is then reshaped back
            pooled_tokens = self.pool(reshaped_stack).view(nL, B, D)
        else:
            pooled_tokens = layer_stack.mean(dim=2)  # (nL, B, D)

        # Now, combine across layers
        # Transpose to (B, nL, D) for layer combination
        layer_combined = pooled_tokens.permute(1, 0, 2)

        if self.layer_combine == "attn":
            return self.layer_pool(layer_combined)  # (B, D)
        elif self.layer_combine == "mean":
            return layer_combined.mean(dim=1)  # (B, D)




class HFEncoder_notPooled(nn.Module):
    """
    Generic frozen encoder for any HF causal-LM (Llama-2/3, Yi, Vicuna, …)

    • adds a pad-token automatically if the tokenizer lacks one  
    • passes only the arguments every model accepts
    • optional AttnPool1D instead of simple mean-pool
    """
    def __init__(self,
                 model_name       : str | None = None,
                 model            = None,
                 tokenizer        = None,
                 device           : str = "cuda",
                 dtype            = torch.float16,
                 attn_pool        : bool = False,
                 layers           = "last",         
                 layer_combine    : str = "mean",
                 last_token       : bool = False):  
        super().__init__()

        # --- load or re-use -------------------------------------------------
        if model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model     = AutoModelForCausalLM.from_pretrained(
                                model_name, torch_dtype=dtype,
                                low_cpu_mem_usage=True, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
            self.model     = model

        self.model.to(device).eval()
        self.device = device

        # Llama & friends ship without a pad token → patch it here
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token     = self.tokenizer.eos_token
            self.tokenizer.pad_token_id  = self.tokenizer.eos_token_id
            
        self.use_attn = attn_pool

        # ---- select the layers to use
        if layers == "last":
            self.layer_ids = [-1]                    # last layer only
        else:                                        # e.g. [ -1, -3, -5 ]
            self.layer_ids = list(layers)

        self.layer_combine = layer_combine.lower()
        assert self.layer_combine in ("mean","attn")

        # nL      = len(self.layer_ids)
        self.out_dim   = self.model.config.hidden_size
        if self.layer_combine=="attn":
            self.layer_pool = AttnPool1D(self.out_dim).to(device)
        self.last_token = last_token


    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            max_length=getattr(self.model.config, "max_position_embeddings", None)
        ).to(self.device)
        overflow = getattr(toks, "overflowing_tokens", None)
        if overflow is not None and overflow.numel() > 0:
            n_trunc = toks.overflowing_tokens.shape[0]
            max_len = getattr(self.model.config, "max_position_embeddings", None)
            warnings.warn(f"[HFEncoder] {n_trunc} prompts truncated to "
                          f"{self.tokenizer.model_max_length} tokens."
                          "\nRaising it to {max_len}...")
            self.tokenizer.model_max_length = max_len
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,          
                max_length=getattr(self.model.config, "max_position_embeddings", None)
            ).to(self.device)
            overflow = getattr(toks, "overflowing_tokens", None)
            if overflow is not None and overflow.numel() > 0:
                warnings.warn(f"STILL {n_trunc} prompts truncated to "
                            f"{max_len} tokens!")

        # Llama-3 forward() only accepts input_ids & attention_mask
        inputs = {"input_ids": toks.input_ids,
                  "attention_mask": toks.attention_mask}

        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        # last = out.hidden_states[-1]                    # (B, T, D)
        # return last
        hs   = out.hidden_states                     # tuple(len=L+1) (B,T,D)

        picked = [hs[i] for i in self.layer_ids]     # list[(B,T,D)]
        layer_stack = torch.stack(picked, dim=0)         # (nL, B, T, D)
        if self.last_token:
            layer_stack = layer_stack[:, :, -1:, :]
        # print(f"\n\n layer_stack.shape: {layer_stack.shape} \n\n")
        
        # ---- !!# TODO: This is purely experimental! Uncomment it or update when done!
        # transpose to (B, nL, T, D) for pooling
        layer_stack = layer_stack.permute(1, 0, 2, 3)

        if self.layer_combine=='attn':
            return self.layer_pool(layer_stack)                      # (B, T, D)
        return layer_stack.mean(dim=1)                         # (B, T, D)
        # ----
        # return layer_stack
    

    



class GemmaHFEncoder_notPooled(torch.nn.Module):
    """
    Same interface as HFEncoder but works around Gemma’s generate-only docs.
    We just do a standard forward pass with `output_hidden_states=True`.
    """
    def __init__(self, model, tokenizer, device="cuda",
                 dtype: torch.dtype = torch.float16,
                 attn_pool: bool = False,
                 last_token: bool = False):
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.last_token = last_token

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            max_length=getattr(self.model.config, "max_position_embeddings", None)
        ).to(self.device)
        overflow = getattr(toks, "overflowing_tokens", None)
        if overflow is not None and overflow.numel() > 0:
            n_trunc = toks.overflowing_tokens.shape[0]
            max_len = getattr(self.model.config, "max_position_embeddings", None)
            warnings.warn(f"[HFEncoder] {n_trunc} prompts truncated to "
                          f"{self.tokenizer.model_max_length} tokens."
                          "\nRaising it to {max_len}...")
            self.tokenizer.model_max_length = max_len
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                max_length=getattr(self.model.config, "max_position_embeddings", None)
            ).to(self.device)
            overflow = getattr(toks, "overflowing_tokens", None)
            if overflow is not None and overflow.numel() > 0:
                warnings.warn(f"STILL {n_trunc} prompts truncated to "
                            f"{max_len} tokens!")

        out = self.model(**toks, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]
        if self.last_token:        
            return h[:, -1:, :]
        else:
            return h
    
    
    
class GemmaHFEncoder(torch.nn.Module):
    """
    Same interface as HFEncoder but works around Gemma’s generate-only docs.
    We just do a standard forward pass with `output_hidden_states=True`.
    """
    def __init__(self, model, tokenizer, device="cuda",
                 dtype: torch.dtype = torch.float16,
                 attn_pool: bool = False):
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.use_attn = attn_pool
        if self.use_attn:
            self.pool = AttnPool1D(self.model.config.hidden_size).to(device)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            max_length=getattr(self.model.config, "max_position_embeddings", None)
        ).to(self.device)
        overflow = getattr(toks, "overflowing_tokens", None)
        if overflow is not None and overflow.numel() > 0:
            n_trunc = toks.overflowing_tokens.shape[0]
            max_len = getattr(self.model.config, "max_position_embeddings", None)
            warnings.warn(f"[HFEncoder] {n_trunc} prompts truncated to "
                          f"{self.tokenizer.model_max_length} tokens."
                          "\nRaising it to {max_len}...")
            self.tokenizer.model_max_length = max_len
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                max_length=getattr(self.model.config, "max_position_embeddings", None)
            ).to(self.device)
            overflow = getattr(toks, "overflowing_tokens", None)
            if overflow is not None and overflow.numel() > 0:
                warnings.warn(f"STILL {n_trunc} prompts truncated to "
                            f"{max_len} tokens!")

        out = self.model(**toks, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]          # (B,T,D)
        if self.use_attn:
            pooled = self.pool(h)          # attention-pool
        else:
            pooled = h.mean(dim=1)         # mean-pool (old behaviour)
        return pooled
    
    

class GemmaHFEncoder_withLogits(torch.nn.Module):
    """
    Same interface as HFEncoder but works around Gemma’s generate-only docs.
    We just do a standard forward pass with `output_hidden_states=True`.
    """
    def __init__(self, model, tokenizer, device="cuda",
                 dtype: torch.dtype = torch.float16,
                 attn_pool: bool = False):
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.use_attn = attn_pool
        if self.use_attn:
            self.pool = AttnPool1D(self.model.config.hidden_size).to(device)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        out = self.model(**toks, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]          # (B,T,D)
        logits = out.logits                     # (B, T, V)  →  (B, V)
        if self.use_attn:
            pooled = self.pool(h)          # attention-pool
            logits_pooled = self.pool(logits)  # attention-pool logits
        else:
            pooled = h.mean(dim=1)         # mean-pool (old behaviour)
            logits_pooled = logits.mean(dim=1)  # mean-pool logits
        return pooled, logits_pooled

        
        

# ↓ add at the end of file (keep the existing classes)
class GemmaHFEncoderSAE(torch.nn.Module):
    """
    Gemma encoder that concatenates:
      • mean-pooled final hidden state  (4096 d)
      • SAE sparse code from residual of <target_layer>  (16 k d, mostly zeros)
    """
    def __init__(self,
                 model,
                 tokenizer,
                 sae,
                 target_layer: int = 20,
                 device: str = "cuda",
                 attn_pool: bool = False):
        super().__init__()
        self.model, self.tokenizer, self.sae = model, tokenizer, sae
        self.device, self.target_layer = device, target_layer
        self.model.to(device).eval()
        self.sae.eval()
        self.use_attn = attn_pool
        if self.use_attn:
            self.pool = AttnPool1D(self.model.config.hidden_size).to(device)

        # register one forward-hook that caches residual activations
        self._cached_resid = None
        def _hook(_, __, outputs):
            # Gemma-Scope hook convention: outputs[0] is residual stream (B, T, D)
            self._cached_resid = outputs[0]
        self.model.model.layers[self.target_layer].register_forward_hook(_hook)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        self._cached_resid = None                         # reset cache
        toks = self.tokenizer(texts,
                              return_tensors="pt",
                              padding=True,
                              truncation=True).to(self.device)

        out = self.model(**toks,
                         output_hidden_states=True,
                         use_cache=False)

        h = out.hidden_states[-1]          # (B,T,D)
        if self.use_attn:
            dense = self.pool(h)          # attention-pool
        else:
            dense = h.mean(dim=1)         # mean-pool (old behaviour)

        resid = self._cached_resid                        # (B, T, D)
        if resid is None:
            raise RuntimeError("SAE hook captured nothing.")
        resid_vec = resid.mean(dim=1)                     # (B, D)

        code = self.sae.encode(resid_vec.float())         # (B, 16384)
        code = code.to(dense.dtype)                       # cast back to fp16

        return torch.cat([dense, code], dim=-1)           # (B, 4096+16384)





# -------------------------------------------------------------------------- #
# Multi-layer SAE encoder: concatenate sparse codes from several residual
# layers and return  dense(4096)  ⊕  code_layer1  ⊕  …  ⊕  code_layerL
# -------------------------------------------------------------------------- #
from utils.sae_utils import JumpReLUSAE
class GemmaHFEncoderMultiSAE_notPooled(torch.nn.Module):
    def __init__(self,
                 model,
                 tokenizer,
                 sae_dict: dict[int, JumpReLUSAE],   # layer → SAE
                 device: str = "cuda",
                 attn_pool: bool = False):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        self.sae_dict   = sae_dict                    # frozen or trainable
        self.device     = device
        self.layers     = sorted(sae_dict.keys())     # e.g. [14, 20, 26]
        self.use_attn = attn_pool
        if self.use_attn:
            self.pool = AttnPool1D(self.model.config.hidden_size).to(device)

        self.model.to(device).eval()

        # ---- register hooks, one per layer --------------------------------
        self._cache = {l: None for l in self.layers}

        def make_hook(layer_id):
            def _hook(_, __, out):
                self._cache[layer_id] = out[0]   # (B, T, D)
                return out
            return _hook

        for l in self.layers:
            self.model.model.layers[l].register_forward_hook(make_hook(l))

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        # reset caches
        for l in self.layers:
            self._cache[l] = None

        toks = self.tokenizer(texts,
                              return_tensors="pt",
                              padding=True,
                              truncation=True).to(self.device)

        out = self.model(**toks, output_hidden_states=True, use_cache=False)
        dense = out.hidden_states[-1]          # (B,T,D)

        # build sparse codes in layer order
        code_parts = []
        for l in self.layers:
            resid_vec = self._cache[l]
            if resid_vec is None:
                raise RuntimeError(f"Hook for layer {l} did not fire.")
            code = self.sae_dict[l].encode(resid_vec.float())         # (B, 16k)
            code_parts.append(code.to(dense.dtype))

        full_code = torch.cat(code_parts, dim=-1)                     # (B, 16k*L)
        return torch.cat([dense, full_code], dim=-1)                  # (B, 4096+…)
    
    

# -------------------------------------------------------------------------- #
# Multi-layer SAE encoder: concatenate sparse codes from several residual
# layers and return  dense(4096)  ⊕  code_layer1  ⊕  …  ⊕  code_layerL
# -------------------------------------------------------------------------- #
from utils.sae_utils import JumpReLUSAE
class GemmaHFEncoderMultiSAE(torch.nn.Module):
    def __init__(self,
                 model,
                 tokenizer,
                 sae_dict: dict[int, JumpReLUSAE],   # layer → SAE
                 device: str = "cuda",
                 attn_pool: bool = False):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        self.sae_dict   = sae_dict                    # frozen or trainable
        self.device     = device
        self.layers     = sorted(sae_dict.keys())     # e.g. [14, 20, 26]
        self.use_attn = attn_pool
        if self.use_attn:
            self.pool = AttnPool1D(self.model.config.hidden_size).to(device)

        self.model.to(device).eval()

        # ---- register hooks, one per layer --------------------------------
        self._cache = {l: None for l in self.layers}

        def make_hook(layer_id):
            def _hook(_, __, out):
                self._cache[layer_id] = out[0].mean(1)   # (B, D)
                return out
            return _hook

        for l in self.layers:
            self.model.model.layers[l].register_forward_hook(make_hook(l))

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        # reset caches
        for l in self.layers:
            self._cache[l] = None

        toks = self.tokenizer(texts,
                              return_tensors="pt",
                              padding=True,
                              truncation=True).to(self.device)

        out = self.model(**toks, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]          # (B,T,D)
        if self.use_attn:
            dense = self.pool(h)          # attention-pool
        else:
            dense = h.mean(dim=1)         # mean-pool (old behaviour)


        # build sparse codes in layer order
        code_parts = []
        for l in self.layers:
            resid_vec = self._cache[l]
            if resid_vec is None:
                raise RuntimeError(f"Hook for layer {l} did not fire.")
            code = self.sae_dict[l].encode(resid_vec.float())         # (B, 16k)
            code_parts.append(code.to(dense.dtype))

        full_code = torch.cat(code_parts, dim=-1)                     # (B, 16k*L)
        return torch.cat([dense, full_code], dim=-1)                  # (B, 4096+…)

