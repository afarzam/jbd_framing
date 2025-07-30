import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearDecomposer(nn.Module):
    """
    Improved LinearDecomposer.
    Splits the input vector v into two components using two linear maps,
    then reconstructs v using a single linear layer.
    """
    def __init__(self, enc_dim: int = 4096, d_g: int = 512, d_f: int = 512):
        super().__init__()
        self.Wg = nn.Linear(enc_dim, d_g, bias=False)
        self.Wf = nn.Linear(enc_dim, d_f, bias=False)
        self.recon = nn.Linear(d_g + d_f, enc_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Wf.weight)
        nn.init.xavier_uniform_(self.recon.weight)
        if self.recon.bias is not None:
            nn.init.zeros_(self.recon.bias)

    def forward(self, v):
        v_goal  = self.Wg(v)
        v_frame = self.Wf(v)
        v_hat   = self.recon(torch.cat([v_goal, v_frame], dim=-1))
        return v_goal, v_frame, v_hat


class NonlinearDecomposer(nn.Module):
    """
    NonlinearDecomposer with a deeper reconstruction network, ELU activations,
    and MLPs for both self.Wg and self.Wf.
    Splits the input vector v into two components,
    applies nonlinearity in the mapping stages and in the reconstruction head.
    """
    def __init__(self, enc_dim: int = 4096, d_g: int = 512, d_f: int = 512,
                 hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        # Replace simple linear maps with MLPs for both components
        self.Wg = nn.Sequential(
            nn.Linear(enc_dim, d_g),
            nn.ELU(),
            nn.Linear(d_g, d_g)
        )
        self.Wf = nn.Sequential(
            nn.Linear(enc_dim, d_f),
            nn.ELU(),
            nn.Linear(d_f, d_f)
        )
        
        # Deeper reconstruction network with ELU activations and dropout
        self.recon = nn.Sequential(
            nn.Linear(d_g + d_f, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, enc_dim),
        )
        self._init_weights()
        

    def _init_weights(self):
        # Initialize weights in the MLP for v_goal (Wg)
        for layer in self.Wg:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize weights in the MLP for v_frame (Wf)
        for layer in self.Wf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize weights in the reconstruction network
        for layer in self.recon:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, v):
        v_goal  = self.Wg(v)
        v_frame = self.Wf(v)
        combined = torch.cat([v_goal, v_frame], dim=-1)
        v_hat = self.recon(combined)
        return v_goal, v_frame, v_hat
    
    


class NonlinearDecomposer_tiny(nn.Module):
    """
    NonlinearDecomposer with a deeper reconstruction network, ELU activations,
    and MLPs for both self.Wg and self.Wf.
    Splits the input vector v into two components,
    applies nonlinearity in the mapping stages and in the reconstruction head.
    """
    def __init__(self, enc_dim: int = 4096, d_g: int = 128, d_f: int = 128,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        # Replace simple linear maps with MLPs for both components
        self.Wg = nn.Sequential(
            nn.Linear(enc_dim, d_g),
            nn.ELU(),
            nn.Linear(d_g, d_g)
        )
        self.Wf = nn.Sequential(
            nn.Linear(enc_dim, d_f),
            nn.ELU(),
            nn.Linear(d_f, d_f)
        )
        
        # Deeper reconstruction network with ELU activations and dropout
        self.recon = nn.Sequential(
            nn.Linear(d_g + d_f, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, enc_dim),
        )
        self._init_weights()
        

    def _init_weights(self):
        # Initialize weights in the MLP for v_goal (Wg)
        for layer in self.Wg:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize weights in the MLP for v_frame (Wf)
        for layer in self.Wf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize weights in the reconstruction network
        for layer in self.recon:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, v):
        v_goal  = self.Wg(v)
        v_frame = self.Wf(v)
        combined = torch.cat([v_goal, v_frame], dim=-1)
        v_hat = self.recon(combined)
        return v_goal, v_frame, v_hat
    
    


import torch, torch.nn as nn, torch.nn.functional as F
class TinyDecomposer(nn.Module):
    """
    Much smaller than the default MLP above.
    """
    def __init__(self, enc_dim, d_g=128, d_f=128, hidden=512):
        super().__init__()
        self.Wg = nn.Sequential(nn.Linear(enc_dim, hidden),
                                nn.ELU(),
                                nn.Linear(hidden, d_g))
        self.Wf = nn.Sequential(nn.Linear(enc_dim, hidden),
                                nn.ELU(),
                                nn.Linear(hidden, d_f))
        self.recon = nn.Sequential(nn.Linear(d_g + d_f, hidden),
                                   nn.ELU(),
                                   nn.Linear(hidden, enc_dim))

    def forward(self, x):
        vg = self.Wg(x)
        vf = self.Wf(x)
        vhat = self.recon(torch.cat([vg, vf], dim=-1))
        return vg, vf, vhat

    
    
    
class LinearDecomposer_tokenwise(nn.Module):
    """
    Improved LinearDecomposer.
    Splits the input vector v into two components using two linear maps,
    then reconstructs v using a single linear layer.
    """
    def __init__(self, enc_dim: int = 4096, seq_len: int = 8192, d_g: int = 512, d_f: int = 512):
        super().__init__()
        self.Wg = nn.Linear(enc_dim, d_g, bias=False)
        self.Wf = nn.Linear(enc_dim, d_f, bias=False)
        self.recon = nn.Linear(d_g + d_f, enc_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Wf.weight)
        nn.init.xavier_uniform_(self.recon.weight)
        if self.recon.bias is not None:
            nn.init.zeros_(self.recon.bias)

    def forward(self, v):
        v_goal  = self.Wg(v)
        v_frame = self.Wf(v)
        v_hat   = self.recon(torch.cat([v_goal, v_frame], dim=-1))
        return v_goal, v_frame, v_hat


class NonlinearDecomposer_tokenwise(nn.Module):
    """
    NonlinearDecomposer with a deeper reconstruction network, ELU activations,
    and MLPs for both self.Wg and self.Wf.
    Splits the input vector v into two components,
    applies nonlinearity in the mapping stages and in the reconstruction head.
    """
    def __init__(self, enc_dim: int = 4096, seq_len: int = 8192, d_g: int = 512, d_f: int = 512,
                 hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        # Replace simple linear maps with MLPs for both components
        self.Wg = nn.Sequential(
            nn.Linear(enc_dim, d_g),
            nn.ELU(),
            nn.Linear(d_g, d_g)
        )
        self.Wf = nn.Sequential(
            nn.Linear(enc_dim, d_f),
            nn.ELU(),
            nn.Linear(d_f, d_f)
        )
        
        # Deeper reconstruction network with ELU activations and dropout
        self.recon = nn.Sequential(
            nn.Linear(d_g + d_f, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, enc_dim),
        )
        self._init_weights()
        

    def _init_weights(self):
        # Initialize weights in the MLP for v_goal (Wg)
        for layer in self.Wg:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize weights in the MLP for v_frame (Wf)
        for layer in self.Wf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize weights in the reconstruction network
        for layer in self.recon:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, v):
        v_goal  = self.Wg(v)
        v_frame = self.Wf(v)
        combined = torch.cat([v_goal, v_frame], dim=-1)
        v_hat = self.recon(combined)
        return v_goal, v_frame, v_hat

    

import torch, torch.nn as nn, torch.nn.functional as F
class TinyDecomposer_tokenwise(nn.Module):
    """
    Much smaller than the default MLP above.
    """
    def __init__(self, enc_dim, seq_len: int = 8192, d_g=128, d_f=128, hidden=512):
        super().__init__()
        self.Wg = nn.Sequential(nn.Linear(enc_dim, hidden),
                                nn.ELU(),
                                nn.Linear(hidden, d_g))
        self.Wf = nn.Sequential(nn.Linear(enc_dim, hidden),
                                nn.ELU(),
                                nn.Linear(hidden, d_f))
        self.recon = nn.Sequential(nn.Linear(d_g + d_f, hidden),
                                   nn.ELU(),
                                   nn.Linear(hidden, enc_dim))

    def forward(self, x):
        vg = self.Wg(x)
        vf = self.Wf(x)
        vhat = self.recon(torch.cat([vg, vf], dim=-1))
        return vg, vf, vhat


