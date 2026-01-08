
from max.nn import (
    Module,
    Linear,
)

# NOTE 
# These methods are adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py

def get_timestep_embedding(
    timesteps: TensorValue,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> TensorValue:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (TensorValue):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        TensorValue: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = ops.mul(-math.log(max_period), ops.range(
        start=0, stop=half_dim, dtype=torch.float32, device=timesteps.device
    ))
    exponent = ops.div(exponent, (half_dim - downscale_freq_shift))

    emb = ops.exp(exponent)
    emb = ops.mul(timesteps[:, None].float(), emb[None, :])

    # scale embeddings
    emb = ops.mul(scale, emb)

    # concat sine and cosine embeddings
    emb = ops.concat([ops.sin(emb), ops.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = ops.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: TensorValue) -> TensorValue:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

# TODO: add more general support (e.g. bias parameter unfixed, cond_proj_dim, post_act_fn)
class TimestepEmbedding(Module): 
    def __init__(self,
                 in_dim: int
                 latent_dim: int,
                 dtype,
                 device,
                 out_dim: int = None):
        super()__init__()

        self.linear_1 = Linear(
            in_dim=in_dim,
            out_dim=latent_dim,
            dtype=dtype,
            device=device,
            name="linear_1",
            has_bias=True
        )
        self.act = ops.silu
        self.linear_2 = Linear(
            in_dim=latent_dim,
            out_dim=latent_dim if out_dim is None else out_dim,
            dtype=dtype,
            device=device,
            name="linear_2",
            has_bias=True
        )
    
    def __call__(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)