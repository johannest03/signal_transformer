from flax import linen as nn
import jax.numpy as jnp 

class LowRankTensorFusion(nn.Module):
    
    output_dim:int
    rank:int
    
    @nn.compact
    def __call__(self, cls_a, cls_b):
        U = nn.Dense(self.rank)(cls_a)
        V = nn.Dense(self.rank * self.output_dim)(cls_b)
        V = V.reshape(cls_b.shape[0], self.rank, self.output_dim)
        
        fused = jnp.einsum('bi,bij->bj', U, V)
        
        fused = nn.gelu(fused)
        return fused