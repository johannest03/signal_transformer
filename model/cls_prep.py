from flax import linen as nn
import jax.numpy as jnp

class CLSPrep(nn.Module):
    d_model:int
    
    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        cls_token = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, self.d_model))
        cls_tokens = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        return x