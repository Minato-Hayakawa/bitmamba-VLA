import torch
import torch.nn as nn
import torch.nn.functional as F
from .bit_linear import BitLinear, RMSNorm


class BitMambaBlock(nn.Module):
    """
    BitMamba-2 Block integrating BitLinear layers and Mamba-2 Selective SSM logic.
    """
    def __init__(self, d_model: int, n_heads: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_conv = d_conv
        self.expand = expand
        
        self.d_inner = int(self.expand * self.d_model)
        self.head_dim = self.d_inner // self.n_heads
        
        # Linear projection dimension: z (d_inner) + x (d_inner) + B (n_heads) + C (n_heads) + dt (n_heads)
        self.dim_proj = 2 * self.d_inner + 3 * self.n_heads
        
        # 1. Input Projector using BitLinear
        self.in_proj = BitLinear(self.d_model, self.dim_proj)
        
        # 2. 1D Causal Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1  # Causal padding handeled in forward
        )
        
        # 3. Mamba Parameters
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 4. Output Projector using BitLinear
        self.out_proj = BitLinear(self.d_inner, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to concatenated [z, x_in, B, C, dt]
        zxbcdt = self.in_proj(x)
        
        # Split projections
        splits = [
            self.d_inner, 
            self.d_inner, 
            self.n_heads, 
            self.n_heads, 
            self.n_heads
        ]
        z, x_in, B, C, dt = torch.split(zxbcdt, splits, dim=-1)
        
        # 2. 1D Causal Convolution over x_in
        # Reshape to (Batch, Channels, SeqLen) for Conv1d
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim causal padding
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 3. SSM Parameters calculation
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log).view(1, 1, self.n_heads, 1)
        
        # Reshape for multi-head SSM computation
        x_r = x_conv.view(batch_size, seq_len, self.n_heads, self.head_dim)
        delta = torch.exp(A * dt.unsqueeze(-1))
        u = x_r * (B.unsqueeze(-1) * dt.unsqueeze(-1))
        
        # 4. Recurrent Scan / State Space Transition Loop
        h = torch.zeros(batch_size, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        h_states = []
        for t in range(seq_len):
            h = h * delta[:, t] + u[:, t]
            h_states.append(h)
            
        h_states = torch.stack(h_states, dim=1)  # (Batch, SeqLen, Heads, HeadDim)
        
        # 5. Output Projection
        y = h_states * C.unsqueeze(-1)
        y = y.view(batch_size, seq_len, self.d_inner)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        
        return self.out_proj(y)


class BitMamba2LM(nn.Module):
    """
    Complete BitMamba-2 Language Model Architecture.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_layers: int, 
        n_heads: int, 
        d_conv: int = 4, 
        expand: int = 2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            BitMambaBlock(d_model=d_model, n_heads=n_heads, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits