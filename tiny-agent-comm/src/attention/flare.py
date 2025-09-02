"""
FLARE (Fast Low-rank Attention Routing Engine) Implementation
Based on arXiv:2508.12594
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


class FLAREAttention(nn.Module):
    """
    FLARE attention mechanism with linear complexity O(NM) instead of O(N²).
    Routes attention through fixed-length latent sequences.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        latent_tokens: int = 32,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_deep_mlp: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.latent_tokens = latent_tokens
        self.head_dim = head_dim or (dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # Learnable latent sequence
        self.latent_queries = nn.Parameter(
            torch.randn(1, latent_tokens, dim)
        )
        
        # Input projections
        self.q_proj = nn.Linear(dim, self.head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * num_heads, bias=False)
        
        # Deep MLP for key/value projections (as per paper)
        if use_deep_mlp:
            self.latent_k_mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, self.head_dim * num_heads),
            )
            self.latent_v_mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, self.head_dim * num_heads),
            )
        else:
            self.latent_k_mlp = nn.Linear(dim, self.head_dim * num_heads)
            self.latent_v_mlp = nn.Linear(dim, self.head_dim * num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(self.head_dim * num_heads, dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand latent queries for batch
        latent_queries = repeat(
            self.latent_queries, '1 m d -> b m d', b=batch_size
        )
        
        # Project inputs
        q = self.q_proj(x)  # (B, N, H*D)
        k = self.k_proj(x)  # (B, N, H*D)
        v = self.v_proj(x)  # (B, N, H*D)
        
        # Project latent sequences
        latent_k = self.latent_k_mlp(latent_queries)  # (B, M, H*D)
        latent_v = self.latent_v_mlp(latent_queries)  # (B, M, H*D)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        latent_k = rearrange(latent_k, 'b m (h d) -> b h m d', h=self.num_heads)
        latent_v = rearrange(latent_v, 'b m (h d) -> b h m d', h=self.num_heads)
        
        # Step 1: Encode - Project input to latent sequence
        # Attention from latent queries to input keys/values
        encode_scores = torch.matmul(latent_k, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            encode_scores = encode_scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        encode_attn = F.softmax(encode_scores, dim=-1)
        encode_attn = self.dropout(encode_attn)
        
        # Aggregate input values into latent space
        latent_representation = torch.matmul(encode_attn, v)  # (B, H, M, D)
        
        # Step 2: Decode - Project latent sequence back to input sequence
        # Attention from input queries to latent keys/values
        decode_scores = torch.matmul(q, latent_v.transpose(-2, -1)) * self.scale
        decode_attn = F.softmax(decode_scores, dim=-1)
        decode_attn = self.dropout(decode_attn)
        
        # Aggregate latent values back to input space
        out = torch.matmul(decode_attn, latent_representation)  # (B, H, N, D)
        
        # Reshape and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        if return_attention:
            # Return both encode and decode attention patterns
            return out, (encode_attn, decode_attn)
        return out, None


class FLARETransformerBlock(nn.Module):
    """
    Transformer block using FLARE attention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        latent_tokens: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = FLAREAttention(
            dim=dim,
            num_heads=num_heads,
            latent_tokens=latent_tokens,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention with residual
        attn_out, _ = self.attn(self.norm1(x), mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class FLAREEncoder(nn.Module):
    """
    Stack of FLARE transformer blocks for encoding
    """
    
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int = 8,
        latent_tokens: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FLARETransformerBlock(
                dim=dim,
                num_heads=num_heads,
                latent_tokens=latent_tokens,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x