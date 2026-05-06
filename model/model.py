import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.fft as fft
import math
import torch.nn.functional as F
# from lstm import CudnnLstmModels
# from fno import FNO2d, FNO1d
from .tcn import TemporalTCN
import pandas as pd
import numpy as np



class MonotoneMap1D(nn.Module):
    """
    Monotone map:
      f(x) = alpha*x + sum_k w_k * softplus(s_k*(x - b_k)) + c
    with alpha, w_k, s_k >= 0 (via softplus), b_k, c free.
    """
    def __init__(self, n_bumps: int = 8, eps: float = 1e-6):
        super().__init__()
        self.n_bumps = n_bumps
        self.eps = eps

    def forward(self, x, packed_params):
        """
        x: (...,)
        packed_params: (..., P) where P = 2 + 3K   (alpha, c, [w_k, s_k, b_k]_k)
        Layout we expect:
          [ alpha, c, w_1..w_K, s_1..s_K, b_1..b_K ]
        """
        K = (packed_params.shape[-1] - 2) // 3
        assert K > 0

        alpha_raw = packed_params[..., 0]
        c         = packed_params[..., 1]
        w_raw     = packed_params[..., 2:2+K]
        s_raw     = packed_params[..., 2+K:2+2*K]
        b         = packed_params[..., 2+2*K:2+3*K]

        # constrain to >= 0
        alpha = F.softplus(alpha_raw) + self.eps
        w     = F.softplus(w_raw)     + self.eps
        s     = F.softplus(s_raw)     + self.eps

        # broadcast over K
        z = s * (x[..., None] - b)     # (..., K)
        bumps = F.softplus(z)
        y = alpha * x + (w * bumps).sum(dim=-1) + c
        return y


class SpatioTemporalQM(nn.Module):
    """
    Spatio-Temporal QM with smooth temporal parameterization via Fourier basis.

    Shapes:
        inps:          (B, P, T, F_in)
        patches_latlon: whatever STBlock expects
        x_target:      (B, P, T)
        t_idx (optional): (T,) time index or DOY, used for Fourier basis.

    If t_idx is None:
        -> uses normalized t in [0,1] over the length T.

    If t_idx is provided:
        - If float: assumed already normalized to [0,1].
        - If integer (e.g., 0..T-1 or 1..365):
            -> normalized to [0,1] by dividing by max(t_idx).
    """

    def __init__(
        self,
        f_in,
        f_model=64,
        heads=4,
        t_blocks=3,
        st_layers=2,
        degree=8,
        dropout=0.1,
        transform_type="monotone",   # "monotone" or "poly"
        temp_enc="Conv1d",
        n_harmonics=2,               # number of Fourier harmonics
        enforce_nonneg=True,         # final ReLU on yhat
    ):
        super().__init__()

        self.f_in = f_in
        self.f_model = f_model
        self.degree = degree
        self.transform_type = transform_type
        self.enforce_nonneg = enforce_nonneg
        self.n_harmonics = n_harmonics

        # Input embedding
        self.embed = nn.Linear(f_in, f_model)

        # Spatio-temporal stacks
        self.stacks = nn.ModuleList([
            STBlock(
                f_model,
                heads=heads,
                t_hidden=2 * f_model,
                t_blocks=t_blocks,
                dropout=dropout,
                tempModel=temp_enc,
            )
            for _ in range(st_layers)
        ])

        # Output dimension (ny) of parameter vector
        if self.transform_type == "monotone":
            # alpha, c, (w_k, s_k, b_k) for k = 1..degree
            self.ny = 2 + 3 * degree
        else:
            # polynomial: sum_{i=1..degree} a_i x^i + b
            self.ny = degree + 1

        if n_harmonics ==0 :
            self.to_coeffs = nn.Linear(f_model, self.ny)
        else:
            # Fourier basis size: 1 (constant) + 2 * n_harmonics (sin, cos)
            self.n_basis = 1 + 2 * n_harmonics
            # Map pooled hidden state -> Fourier coefficients for parameters
            # Coeffs shape will be (B, P, n_basis, ny).
            self.to_coeffs = nn.Linear(f_model, self.n_basis * self.ny)

        self.monotone = MonotoneMap1D(n_bumps=degree)

    # ---------------------------------------------------------------------
    def _fourier_basis(self, T, t_idx, device):
        """
        Build Fourier basis of shape (T, n_basis).

        t_idx:
            - None      -> uses linspace(0,1,T)
            - tensor    -> if float, assumed in [0,1]; if int, normalized by max.
        """
        if t_idx is None:
            # uniform time grid in [0,1]
            t = torch.linspace(0.0, 1.0, T, device=device)
        else:
            t = t_idx.to(device)
            if t.dtype.is_floating_point:
                # assume already normalized to [0,1]
                pass
            else:
                # integer time index or DOY -> normalize to [0,1]
                max_t = t.max()
                if max_t == 0:
                    # degenerate, but avoid div-by-zero
                    t = torch.zeros_like(t, dtype=torch.float32)
                else:
                    t = t.to(torch.float32) / max_t

        # t in [0,1], shape (T,)
        basis = [torch.ones(T, device=device)]  # constant term

        for k in range(1, self.n_harmonics + 1):
            basis.append(torch.sin(2 * math.pi * k * t[0])) ## t is same across batch and patches
            basis.append(torch.cos(2 * math.pi * k * t[0]))

        basis = torch.stack(basis, dim=1)  # (T, n_basis)
        return basis

    # ---------------------------------------------------------------------
    def forward(self, inps, patches_latlon, x_target, t_idx=None):
        """
        inps:          (B, P, T, F_in)
        patches_latlon: as before
        x_target:      (B, P, T)
        t_idx:         (T,) optional time index / DOY.

        Returns:
            yhat:   (B, P, T)
            params: (B, P, T, ny)
        """
        B, P, T, _ = inps.shape

        # Embed inputs
        h = self.embed(inps)  # (B, P, T, f_model)

        # Spatio-temporal mixing
        for blk in self.stacks:
            h = blk(h, patches_latlon)  # (B, P, T, f_model)
        
        if self.n_harmonics == 0 :
            params = self.to_coeffs(h)  # (B, P, T, ny)
        else:
            # Temporal pooling over T to get a single representation per (B,P)
            # Simple mean pooling; if you want fancier, change here.
            h_pool = h.mean(dim=2)  # (B, P, f_model)

            # Map pooled hidden state to Fourier coefficients for parameters
            coeffs = self.to_coeffs(h_pool)  # (B, P, n_basis * ny)
            coeffs = coeffs.view(B, P, self.n_basis, self.ny)  # (B, P, n_basis, ny)

            # Build Fourier basis over time: (T, n_basis)
            basis = self._fourier_basis(T, t_idx, device=h.device)  # (T, n_basis)

            # Combine basis and coefficients to get time-varying params
            # params[b,p,t,n] = sum_k coeffs[b,p,k,n] * basis[t,k]
            # -> einsum: 'bpkn,tk -> bptn'
            params = torch.einsum("bpkn,tk->bptn", coeffs, basis)  # (B, P, T, ny)

        # Apply transform
        if self.transform_type == "monotone":
            yhat = self.monotone(x_target, params)  # (B, P, T)
        else:
            # Polynomial branch:
            # params: (..., ny) = [a1..a_degree, b]
            poly_coeffs = params[..., :-1]  # (B, P, T, degree)
            shift = params[..., -1]         # (B, P, T)

            # Positive scales, as in your earlier design
            poly_coeffs = torch.exp(poly_coeffs)

            degree = poly_coeffs.shape[-1]
            powers = torch.stack(
                [x_target ** (i + 1) for i in range(degree)],
                dim=-1,
            )  # (B, P, T, degree)

            yhat = (powers * poly_coeffs).sum(dim=-1) + shift  # (B, P, T)

        if self.enforce_nonneg:
            yhat = F.relu(yhat)

        return yhat, params



class STBlock(nn.Module):
    """
    Interleaved Spatio-Temporal Block:
      x -> TemporalConv1d -> SpatialAttention -> (residuals + norm)
    """
    def __init__(self, dim, heads=4, t_hidden=128, t_blocks=3, dropout=0.1, tempModel='Conv1d'):
        super().__init__()

        self.tempModel = tempModel
        if self.tempModel == 'Conv1d':
            self.tenc = TemporalConv1d(dim, hidden=t_hidden, n_blocks=t_blocks, dropout=dropout)
        elif self.tempModel == 'TCN':
            self.tenc = TemporalTCN(dim, hidden=t_hidden, n_blocks=t_blocks, dropout=dropout)
        elif self.tempModel == 'LSTM':
            self.tenc = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=t_blocks, dropout=dropout, batch_first=True)
        elif self.tempModel == 'MLP':
            self.tenc = build_transform_generator(dim, t_hidden, dim, t_blocks)
        elif self.tempModel == 'MLP+LSTM':
            self.tenc_mlp = build_transform_generator(dim, t_hidden, dim, t_blocks)
            self.tenc_lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=t_blocks, dropout=dropout, batch_first=True)
        elif self.tempModel == 'Transformer':
            self.tenc = TemporalSelfAttention(dim, heads=heads, ff_mult=2, dropout=dropout, causal=False)
        elif self.tempModel == 'Conv1d+MLP':
            self.tenc_conv = TemporalConv1d(dim, hidden=t_hidden, n_blocks=t_blocks, dropout=dropout)
            self.tenc_mlp = build_transform_generator(dim, t_hidden, dim, t_blocks)
        else:
            raise ValueError(f"Unknown tempModel type: {self.tempModel}")
        
        self.sattn = PatchSpatialAttention(dim, n_heads=heads, ff_mult=2, dropout=dropout)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x, pos):            # x: (B,P,T,F); pos: (B,P,2)
        B, P, T, F_ = x.size()
        if self.tempModel in ['LSTM', 'MLP', 'MLP+LSTM', 'Transformer', 'Conv1d+MLP']:
            x_ = x.view(B * P, T, F_) # (BP,T,F)
            if self.tempModel == 'LSTM':
                y_ = self.tenc(x_)[0] # (BP,T,F)
            elif self.tempModel == 'MLP':
                y_ = self.tenc(self.n1(x_)) + x_ # (BP,T,F)
            elif self.tempModel == 'MLP+LSTM':
                y_ = self.tenc_mlp(self.n1(x_)) + x_
                y_ = self.tenc_lstm(y_)[0] + y_
            elif self.tempModel == 'Transformer':
                y_ = self.tenc(self.n1(x_)) + x_
            elif self.tempModel == 'Conv1d+MLP':
                y_ = self.tenc_conv(self.n1(x_.view(B, P, T, F_))).view(B*P, T, F_) + x_
                y_ = self.tenc_mlp(self.n1(y_)) + y_
            y = y_.view(B, P, T, F_)  # (B,P,T,F)
        elif self.tempModel in ['TCN']:
            y = self.tenc(self.n1(x))
        else:
            y = self.tenc(self.n1(x)) + x     # temporal residual

       
        z = self.sattn(self.n2(y), pos)   # spatial attn (already residual inside)
        return z


    

# MLP builder
def build_transform_generator(nx, hidden_dim, ny, num_layers):
    layers = []
    layers.append(nn.Linear(nx, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.2))
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(hidden_dim, ny))
    return nn.Sequential(*layers)



class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        pad = (kernel_size - 1) * dilation
        left_pad = pad if causal else pad // 2

        self.pad1 = (left_pad, 0) if causal else (pad // 2, pad - pad // 2)
        self.pad2 = (left_pad, 0) if causal else (pad // 2, pad - pad // 2)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        y = F.pad(x, self.pad1) if self.causal else F.pad(x, self.pad1, mode='constant', value=0)
        y = self.conv1(y)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)

        y = F.pad(y, self.pad2) if self.causal else F.pad(y, self.pad2, mode='constant', value=0)
        y = self.conv2(y)
        y = self.norm2(y)

        return F.gelu(x + self.dropout(y))  # residual

class TemporalCNN(nn.Module):
    """
    Temporal 1D CNN over rho (sequence length).
    Accepts [B, T, nx] and returns [B, T, ny].
    """
    def __init__(
        self,
        nx,
        ny,
        hidden=64,
        num_blocks=4,
        kernel_size=3,
        base_dilation=1,
        dropout=0.1,
        causal=False
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(nx, hidden, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            dilation = (base_dilation ** i) if base_dilation > 1 else (2 ** i)
            blocks.append(DilatedResBlock(hidden, kernel_size, dilation, dropout, causal))
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Conv1d(hidden, ny, kernel_size=1)

    def forward(self, x_b_t_nx):
        # x_b_t_nx: [B, T, nx]
        x = x_b_t_nx.transpose(1, 2)        # -> [B, nx, T]
        x = self.input_proj(x)              # -> [B, H, T]
        x = self.blocks(x)                  # -> [B, H, T]
        y = self.output_proj(x)             # -> [B, ny, T]
        return y.transpose(1, 2)
    



class TemporalConv1d(nn.Module):
    def __init__(self, dim, hidden=128, kernel_size=3, base_dilation=1, n_blocks=3, dropout=0.1):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            dil = base_dilation * (2**i)
            blocks += [
                nn.Conv1d(dim, dim, kernel_size, padding=dil*(kernel_size-1)//2, dilation=dil, groups=dim),
                nn.GELU(),
                nn.GroupNorm(1, dim),            # <- channel norm on (N,C,T)
                nn.Conv1d(dim, hidden, 1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, dim, 1),
                nn.GroupNorm(1, dim),            # <- channel norm
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):  # x: (B,P,T,F)
        B, P, T, F = x.shape
        y = x.reshape(B*P, T, F).transpose(1, 2)  # (BP, F, T)
        k = 0
        for _ in range(len(self.blocks)//8):
            residual = y
            y = self.blocks[k+0](y); y = self.blocks[k+1](y); y = self.blocks[k+2](y)
            y = self.blocks[k+3](y); y = self.blocks[k+4](y); y = self.blocks[k+5](y)
            y = self.blocks[k+6](y); y = self.blocks[k+7](y)
            y = y + residual
            k += 8
        y = y.transpose(1, 2).reshape(B, P, T, F)  # back to (B,P,T,F)
        return y



def pairwise_relpos(latlon):  # latlon: (B, P, 2) [lat, lon] in degrees
    # returns rel: (B, P, P, 4): [dx, dy, great_circle_dist_km, bearing_sin]
    lat = torch.deg2rad(latlon[..., 0])
    lon = torch.deg2rad(latlon[..., 1])

    dlat = lat[:, :, None] - lat[:, None, :]
    dlon = lon[:, :, None] - lon[:, None, :]

    dx = dlon * torch.cos((lat[:, :, None] + lat[:, None, :]) / 2.0)
    dy = dlat

    # haversine distance (km)
    a = torch.sin(dlat/2)**2 + torch.cos(lat[:, :, None]) * torch.cos(lat[:, None, :]) * torch.sin(dlon/2)**2
    dist = 2 * 6371.0 * torch.arcsin(torch.clamp(torch.sqrt(a), 0, 1-1e-7))

    bearing = torch.atan2(
        torch.sin(dlon) * torch.cos(lat[:, None, :]),
        torch.cos(lat[:, :, None]) * torch.sin(lat[:, None, :]) - torch.sin(lat[:, :, None]) * torch.cos(lat[:, None, :]) * torch.cos(dlon)
    )
    rel = torch.stack([dx, dy, dist/500.0, torch.sin(bearing)], dim=-1)  # mild scaling for dist
    return rel  # (B, P, P, 4)

class PatchSpatialAttention(nn.Module):
    """
    Spatial self-attention over the K+1 nodes in each patch, per time step.
    Input:  x  (B, P, T, F)
            pos (B, P, 2) with [lat, lon] for each node in the patch (deg)
    Output: (B, P, T, F)
    """
    def __init__(self, dim, n_heads=4, ff_mult=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.h = n_heads
        self.dk = dim // n_heads
        assert dim % n_heads == 0

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult*dim),
            nn.GELU(),
            nn.Linear(ff_mult*dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        # Rel-pos -> bias per head
        self.relproj = nn.Linear(4, n_heads, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x, pos):
        """
        x:   (B, P, T, F)
        pos: (B, P, 2) lat/lon degrees for nodes in each patch
        """
        B, P, T, F_ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, P, F)
        x_ = self.norm1(x)

        # QKV along spatial nodes for each time slice
        qkv = self.qkv(x_)  # (B, T, P, 3F)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape for heads
        def split_heads(t):
            return t.view(B, T, P, self.h, self.dk).permute(0,1,3,2,4)  # (B,T,H,P,dk)
        q, k, v = map(split_heads, (q, k, v))

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,T,H,P,P)

        # add relative positional bias per head
        rel = pairwise_relpos(pos)             # (B,P,P,4)
        rel_h = self.relproj(rel).permute(0,3,1,2)  # (B,H,P,P)
        attn = attn + rel_h[:, None, ...]      # broadcast to (B,T,H,P,P)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)            # (B,T,H,P,dk)

        # merge heads
        out = out.permute(0,1,3,2,4).contiguous().view(B, T, P, F_)  # (B,T,P,F)
        out = self.out(out)
        x = x + self.dropout(out)              # residual
        y = self.norm2(x)
        y = y + self.dropout(self.ff(y))       # feed-forward + residual
        y = y.permute(0,2,1,3).contiguous()    # (B,P,T,F)
        return y


class TemporalSelfAttention(nn.Module):
    """
    Temporal Transformer block (per patch):
      Input:  (B*P, T, F)
      Output: (B*P, T, F)
    """
    def __init__(self, dim, heads=4, ff_mult=2, dropout=0.1, causal=False, max_T=6000):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.causal = causal

        # Learned positional embeddings over time steps
        # self.pos_emb = nn.Embedding(max_T, dim)

        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def sinusoidal_pos_emb(T, dim, device='cpu'):
        """
        Returns sinusoidal positional encodings (T, dim)
        for positions [0, T-1].
        """
        pos = torch.arange(T, device=device).unsqueeze(1)           # (T, 1)
        i = torch.arange(dim // 2, device=device).unsqueeze(0)      # (1, dim/2)
        denom = torch.pow(10000, (2 * i) / dim)
        angles = pos / denom                                        # (T, dim/2)
        pe = torch.zeros(T, dim, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def _causal_mask(self, T, device):
        # [T, T] upper-triangular mask: True = mask out
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x):  # x: (B*P, T, F)
        BP, T, F = x.shape
        # pos_ids = torch.arange(T, device=x.device)
        # x = x + self.pos_emb(pos_ids)[None, :, :]  # broadcast over batch
        x = x + self.sinusoidal_pos_emb(T, F, x.device)

        # Pre-norm
        h = self.ln1(x)

        attn_mask = self._causal_mask(T, x.device) if self.causal else None
        # Self-attention
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        y = y + x  # residual

        # FFN
        z = self.ln2(y)
        z = self.ff(z)
        z = z + y  # residual
        return z

