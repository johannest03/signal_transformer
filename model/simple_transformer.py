import jax
import jax.numpy as jnp
from flax import linen as nn 
from deep_learning.positional_encoding import PositionalEncoding
from deep_learning.cls_prep import CLSPrep
from deep_learning.low_rank_tensor_fusion import LowRankTensorFusion

class SimpleTransformer(nn.Module):
    input_dim: int
    max_len: int
    num_classes: int
    d_model: int = 32
    n_heads: int = 2
    dropout_rate: float = 0.3
    d_ff: int = 128
    rank: int = 16
    
    @nn.compact
    def __call__(self, x_bp, x_ecg, train=True):
        
        x_bp = nn.Conv(features=self.d_model, kernel_size=(5,), strides=(1,), padding="SAME")(x_bp)
        x_ecg = nn.Conv(features=self.d_model, kernel_size=(5,), strides=(1,), padding="SAME")(x_ecg)

        x_bp = nn.Dropout(rate=self.dropout_rate)(x_bp, deterministic=not train)
        x_ecg = nn.Dropout(rate=self.dropout_rate)(x_ecg, deterministic=not train)
      
        bp_emb = PositionalEncoding(self.max_len, self.d_model)(x_bp)
        ecg_emb = PositionalEncoding(self.max_len, self.d_model)(x_ecg)

        bp_emb = nn.Dense(self.d_model)(bp_emb)
        ecg_emb = nn.Dense(self.d_model)(ecg_emb)
        
        # CLS
        bp_emb = CLSPrep(self.d_model)(bp_emb)
        ecg_emb = CLSPrep(self.d_model)(ecg_emb)

        # Self attention block 1
        bp_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(bp_emb, train=train)

        ecg_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(ecg_emb, train=train)

        # Self attention block 2
        bp_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(bp_attn, train=train)

        ecg_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(ecg_attn, train=train)
        
        # Self attention block 3
        bp_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(bp_attn, train=train)

        ecg_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(ecg_attn, train=train)
        
        # Self attention block 4
        bp_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(bp_attn, train=train)

        ecg_attn = SelfAttentionTransFormerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(ecg_attn, train=train)

        # Cross Attention
        bp_co = CrossAttentionTransformerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(bp_attn, ecg_attn, train=train)
        
        ecg_co = CrossAttentionTransformerBlock(
            d_model=self.d_model,
            d_ff = self.d_ff,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
            )(ecg_attn, bp_attn, train=train)
        
        # Fusion        
        bp_cls = bp_co[:, 0, :]
        ecg_cls = ecg_co[:, 0, :]

        fused_bp = bp_cls + LowRankTensorFusion(output_dim = self.d_model, rank=self.rank)(bp_cls, ecg_cls)
        fused_ecg = ecg_cls + LowRankTensorFusion(output_dim = self.d_model, rank=self.rank)(ecg_cls, bp_cls)
        
        gate = nn.Dense(2)(jnp.concatenate([fused_bp, fused_ecg], axis=-1))
        gate = nn.softmax(gate, axis=-1)
        
        combined = gate[:, 0:1] * fused_bp + gate[:, 1:2] * fused_ecg
        combined = nn.Dropout(rate=self.dropout_rate)(combined, deterministic=not train)

        logits = nn.Dense(self.num_classes)(combined)
        return logits
    
    
class SelfAttentionTransFormerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float 
    
    @nn.compact
    def __call__(self, x, train=True):
        x_norm = nn.LayerNorm()(x)
        attn_out = nn.SelfAttention(
            num_heads=self.n_heads, 
            qkv_features=self.d_model, 
            dropout_rate=self.dropout_rate, 
        )(x_norm, deterministic=not train)
        
        x = x + nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=not train)
        
        ff_out = nn.LayerNorm()(x)
        ff_out = nn.Dense(self.d_ff * 2)(ff_out)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate)(ff_out, deterministic=not train)
        x = x + ff_out
        return x
    
class CrossAttentionTransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float 
    
    @nn.compact
    def __call__(self, x, y, train=True):
        x_norm = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, 
            dropout_rate=self.dropout_rate, 
        )(x_norm, y, deterministic=not train)
        
        x = x + nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=not train)

        ff_out = nn.LayerNorm()(x)
        ff_out = nn.Dense(self.d_ff * 2)(ff_out)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate)(ff_out, deterministic=not train)
        x = x + ff_out
        return x