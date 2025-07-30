import torch, torch.nn as nn, torch.nn.functional as F

# class AttnPool1D(nn.Module):
#     """
#     One-headed self-attention that returns a single pooled vector.
#     • input  : (B, T, D)
#     • output : (B, D)
#     """
#     def __init__(self, d_model: int):
#         super().__init__()
#         # learnable “query’’ which attends to the token sequence
#         self.q = nn.Parameter(torch.randn(1, 1, d_model))  # (1,1,D)
#         self.k = nn.Linear(d_model, d_model, bias=False)   # key / value share proj
#         self.v = nn.Linear(d_model, d_model, bias=False)

#     def forward(self, h: torch.Tensor) -> torch.Tensor:     # h : (B,T,D)
#         B, T, D = h.shape
#         q = self.q.expand(B, 1, D)                          # (B,1,D)
#         k = self.k(h)                                       # (B,T,D)
#         v = self.v(h)                                       # (B,T,D)

#         attn_logits = (q @ k.transpose(1,2)) / (D ** 0.5)   # (B,1,T)
#         w = F.softmax(attn_logits, dim=-1)                  # (B,1,T)
#         pooled = (w @ v).squeeze(1)                         # (B,D)
#         return pooled                        # same shape the mean-pool returned



class AttnPool1D(nn.Module):
    """
    One-headed self-attention that returns a single pooled vector.
    • input  : (B, T, D) or (B, nL, T, D)
    • output : (B, D) or (B, T, D)
    If input is 4D, pools along nL for each token position.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))  # (1,1,D)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3:
            # h: (B, T, D) -- pool along T
            B, T, D = h.shape
            if h.dtype != self.k.weight.dtype:
                h = h.to(self.k.weight.dtype)
            q = self.q.expand(B, 1, D).to(h.dtype)              # (B,1,D)
            k = self.k(h)                                       # (B,T,D)
            v = self.v(h)                                       # (B,T,D)
            attn_logits = torch.matmul(q, k.transpose(1,2)) / (D ** 0.5)  # (B,1,T)
            w = F.softmax(attn_logits, dim=-1)                  # (B,1,T)
            pooled = torch.matmul(w, v).squeeze(1)              # (B,D)
            return pooled
        elif h.dim() == 4:
            # h: (B, nL, T, D) -- pool along nL for each token
            B, nL, T, D = h.shape
            h_perm = h.permute(0, 2, 1, 3)  # (B, T, nL, D)
            if h_perm.dtype != self.k.weight.dtype:
                h_perm = h_perm.to(self.k.weight.dtype)
            q = self.q.expand(B, T, 1, D).to(h_perm.dtype)      # (B, T, 1, D)
            k = self.k(h_perm)                                  # (B, T, nL, D)
            v = self.v(h_perm)                                  # (B, T, nL, D)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, T, 1, nL)
            w = F.softmax(attn_logits, dim=-1)                               # (B, T, 1, nL)
            pooled = torch.matmul(w, v).squeeze(2)                           # (B, T, D)
            return pooled
        else:
            raise ValueError("Input must be of shape (B, T, D) or (B, nL, T, D)")