import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class BitLinear(nn.Module):
    """
    1.58-bit (Ternary {-1, 0, 1}) Weight & 8-bit Activation Linear Layer using STE.
    Based on BitNet b1.58 implementation.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.norm = RMSNorm(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 1. Activation Normalization & 8-bit Quantization ---
        x_norm = self.norm(x)
        # Scale to [-128, 127]
        max_val = torch.max(torch.abs(x_norm), dim=-1, keepdim=True).values.clamp(min=1e-5)
        scale_x = 127.0 / max_val
        x_quant = torch.clamp(torch.round(x_norm * scale_x), -128, 127) / scale_x
        
        # STE for Activation
        x_eff = x_norm + (x_quant - x_norm).detach()

        # --- 2. Weight 1.58-bit (Ternary {-1, 0, 1}) Quantization ---
        scale_w = 1.0 / torch.mean(torch.abs(self.weight)).clamp(min=1e-5)
        w_quant = torch.clamp(torch.round(self.weight * scale_w), -1, 1) / scale_w
        
        # STE for Weight
        w_eff = self.weight + (w_quant - self.weight).detach()

        # --- 3. Linear Transformation ---
        return F.linear(x_eff, w_eff, self.bias)