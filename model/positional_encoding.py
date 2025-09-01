from flax import linen as nn

class PositionalEncoding(nn.Module):
    max_len: int
    d_model: int
    
    @nn.compact
    def __call__(self, x):
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        )
        seq_len = x.shape[1]
        return x + pos_emb[:seq_len][None, :, :]

