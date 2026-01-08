# NOTE: WARNING
# The following code has been in part written by Grok code, and has not been tested yet. 
# USE AT YOUR OWN RISK.


from max.nn import (
    Module,
    Conv2d,
    ConvTranspose2d,
    LayerList,
    LayerNorm,
    GroupNorm,
    Linear,
)
from max.graph import TensorValue, ops
from .attention import UNetAttention

from .timesteps import Timesteps

class Downsample2D(Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 device,
                 dtype):
        super().__init__()
        
        self.conv = Conv2d(
            kernel_size=kernel_size,
            in_channels=dim,
            out_channels=dim,
            dtype=dtype,
            stride=(2,2),
            padding=(1,1),
            device=device
        )

    def __call__(self, x):
        return self.conv(x)

class Upsample2D(Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 device,
                 dtype):
        super().__init__()

        self.conv = ConvTranspose2d( # TODO
            kernel_size=kernel_size,
            in_channels=dim,
            out_channels=dim,
            dtype=dtype,
            stride=(2, 2),
            padding=(0, 0),
            device=device
        )

    def __call__(self, x):
        return self.conv(x)

class GEGLU(Module):
    def __init__(self,
                 device,
                 dtype,
                 hidden_dim,
                 latent_dim,
    ) -> None:
        super().__init__()
        
        self.proj = Linear(
            in_dim=hidden_dim,
            out_dim=latent_dim,
            dtype=dtype,
            device=device,
            name="proj",
            has_bias=True
        )
    
    def __call__(self, x):
        x = self.proj(x)
        chunks = ops.chunk(x, 2, axis=-1)
        x = chunks[0]
        g = chunks[1]
        return ops.mul(x, ops.gelu(g))

# Modified from https://github.com/huggingface/diffusers/blob/8600b4c10d67b0ce200f664204358747bd53c775/src/diffusers/models/attention.py#L752
class UNetBasicTransformerBlock(Module):
    r"""
    A basic Transformer block.

    Parameters:
        hidden_dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
    """

    def __init__(self,
                 device,
                 dtype,
                 hidden_dim: int,
                 num_attention_heads: int,
                 attention_head_dim: int,
                 cross_attention_dim: Optional[int] = None,
                 activation_fn: str = 'geglu',
                 attention_bias: bool = False,
                 attention_out_bias: bool = True,
                 norm_type: str = 'layer_norm',
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        
        if norm_type != 'layer_norm': raise NotImplementedError()
        if activation_fn != 'geglu': raise NotImplementedError()

        self.norm1 = LayerNorm(hidden_dim, [device], dtype)
        self.attn1 = UNetAttention( # first self attention
            device,
            dtype,
            query_dim=hidden_dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_has_bias=attention_bias,
            o_proj_has_bias=attention_out_bias,
        )
        self.norm2 = LayerNorm(hidden_dim, [device], dtype)
        self.attn2 = UNetAttention( # then cross attention
            device,
            dtype,
            query_dim=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_has_bias=attention_bias,
            o_proj_has_bias=attention_out_bias,
        )
        self.norm3 = LayerNorm(hidden_dim, [device], dtype)
        self.ff = [
            GEGLU(
                device,
                dtype,
                hidden_dim, 
                hidden_dim*4, # TODO: Aath did this for convenience, check if it holds everywhere
            ),
            Linear(
                in_dim=hidden_dim,
                out_dim=hidden_dim*2, # TODO: Aath did this for convenience, check if it holds everywhere
                dtype=dtype,
                device=device,
                name="fc1",
                has_bias=True
            )
        ]
        self.ff = LayerList(ff)
        
    def __call__(self, 
                 x: TensorValue,
                 encoder_hidden_states: Optional[TensorValue] = None):
        h = self.norm1(x)
        h = self.attn1(h)
        x = h + x
        h = self.norm2(x)
        h = self.attn2(h, encoder_hidden_states=encoder_hidden_states)
        x = h + x
        h = self.norm3(x)
        for idx, layer in enumerate(self.ff):
            h = layer(h)
        x = h + x
        return x        

class Transformer2DModel(Module):
    def __init__(self,
                 num_attention_heads,
                 attention_head_dim,
                 in_channels,
                 num_layers,
                 cross_attention_dim,
                 device,
                 dtype,
                 num_groups=32):
        super().__init__()

        self.norm = GroupNorm(num_groups, in_channels, device=device, dtype=dtype)
        self.proj_in = Conv2d(
            kernel_size=(1, 1),
            in_channels=in_channels,
            out_channels=in_channels,
            dtype=dtype,
            stride=(1, 1),
            padding=(0, 0),
            device=device
        )

        self.transformer_blocks = LayerList([
            UNetBasicTransformerBlock(
                device=device,
                dtype=dtype,
                hidden_dim=in_channels,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ])

        self.proj_out = Conv2d(
            kernel_size=(1, 1),
            in_channels=in_channels,
            out_channels=in_channels,
            dtype=dtype,
            stride=(1, 1),
            padding=(0, 0),
            device=device
        )

    def __call__(self, x, encoder_hidden_states=None):
        batch, height, width, channels = x.shape
        residual = x

        x = self.norm(x)
        x = self.proj_in(x)

        # Flatten spatial dimensions for transformer
        x = ops.reshape(x, (batch, height * width, channels))

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, encoder_hidden_states=encoder_hidden_states)

        # Reshape back to spatial dimensions
        x = ops.reshape(x, (batch, height, width, channels))
        x = self.proj_out(x)

        return x + residual

class ResnetBlock2D(Module):
    def __init__(self,
                 num_groups,
                 hidden_dim,
                 kernel_size,
                 device,
                 dtype):
        super().__init__()
        self.norm1 = GroupNorm(num_groups, hidden_dim)
        self.conv1 = Conv2d(
            kernel_size=kernel_size,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dtype=dtype,
            stride=(1,1),
            padding=(1,1),
            device=device
        )
        self.time_emb_proj = Linear(
            in_dim=hidden_dim*4,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            name="time_emb_proj",
            has_bias=True
        )
        self.norm2 = GroupNorm(num_groups, hidden_dim)
        self.conv2 = Conv2d(
            kernel_size=kernel_size,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dtype=dtype,
            stride=(1,1),
            padding=(1,1),
            device=device
        )
        self.nonlinearity = ops.silu

    def __call__(self, x, temb=None):
        residual = x

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        if temb is not None:
            temb = self.time_emb_proj(temb)
            x = x + temb

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        return x + residual

class UpBlock2D(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 layers_per_block,
                 num_groups,
                 device,
                 dtype):
        super().__init__()

        self.resnets = LayerList([
            ResnetBlock2D(
                num_groups=num_groups,
                hidden_dim=in_channels if i == 0 else out_channels,
                kernel_size=kernel_size,
                device=device,
                dtype=dtype
            )
            for i in range(layers_per_block)
        ])

        self.upsamplers = LayerList([
            Upsample2D(
                dim=out_channels,
                device=device,
                dtype=dtype
            )
        ])

    def __call__(self, hidden_states, res_hidden_states_tuple=None, temb=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class CrossAttnUpBlock2D(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 layers_per_block,
                 transformer_layers_per_block,
                 num_attention_heads,
                 attention_head_dim,
                 cross_attention_dim,
                 num_groups,
                 device,
                 dtype):
        super().__init__()

        self.attentions = LayerList([
            Transformer2DModel(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                in_channels=in_channels if i == 0 else out_channels,
                num_layers=transformer_layers_per_block,
                cross_attention_dim=cross_attention_dim,
                device=device,
                dtype=dtype,
                num_groups=num_groups
            )
            for i in range(layers_per_block)
        ])

        self.resnets = LayerList([
            ResnetBlock2D(
                num_groups=num_groups,
                hidden_dim=in_channels if i == 0 else out_channels,
                kernel_size=kernel_size,
                device=device,
                dtype=dtype
            )
            for i in range(layers_per_block)
        ])

        self.upsamplers = LayerList([
            Upsample2D(
                dim=out_channels,
                device=device,
                dtype=dtype
            )
        ])

    def __call__(self, hidden_states, res_hidden_states_tuple=None, temb=None, encoder_hidden_states=None):
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states, temb=temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class DownBlock2D(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 layers_per_block,
                 num_groups,
                 device,
                 dtype):
        super().__init__()

        self.resnets = LayerList([
            ResnetBlock2D(
                num_groups=num_groups,
                hidden_dim=in_channels if i == 0 else out_channels,
                kernel_size=kernel_size,
                device=device,
                dtype=dtype
            )
            for i in range(layers_per_block)
        ])

        self.downsamplers = LayerList([
            Downsample2D(
                dim=out_channels,
                kernel_size=kernel_size,
                device=device,
                dtype=dtype
            )
        ])

    def __call__(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class CrossAttnDownBlock2D(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 layers_per_block,
                 transformer_layers_per_block,
                 num_attention_heads,
                 attention_head_dim,
                 cross_attention_dim,
                 num_groups,
                 device,
                 dtype):
        super().__init__()

        self.attentions = LayerList([
            Transformer2DModel(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                in_channels=in_channels if i == 0 else out_channels,
                num_layers=transformer_layers_per_block,
                cross_attention_dim=cross_attention_dim,
                device=device,
                dtype=dtype,
                num_groups=num_groups
            )
            for i in range(layers_per_block)
        ])

        self.resnets = LayerList([
            ResnetBlock2D(
                num_groups=num_groups,
                hidden_dim=in_channels if i == 0 else out_channels,
                kernel_size=kernel_size,
                device=device,
                dtype=dtype
            )
            for i in range(layers_per_block)
        ])

        self.downsamplers = LayerList([
            Downsample2D(
                dim=out_channels,
                kernel_size=kernel_size,
                device=device,
                dtype=dtype
            )
        ])

    def __call__(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()

        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states, temb=temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class UNetMidBlock2DCrossAttn(Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 num_layers,
                 num_attention_heads,
                 attention_head_dim,
                 cross_attention_dim,
                 num_groups,
                 device,
                 dtype):
        super().__init__()

        self.attentions = LayerList([
            Transformer2DModel(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                in_channels=in_channels,
                num_layers=1,
                cross_attention_dim=cross_attention_dim,
                device=device,
                dtype=dtype,
                num_groups=num_groups
            )
            for _ in range(num_layers)
        ])

        self.resnets = LayerList([
            ResnetBlock2D(
                num_groups=num_groups,
                hidden_dim=in_channels,
                kernel_size=kernel_size,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers + 1)
        ])

    def __call__(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb=temb)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb=temb)

        return hidden_states


class UNet2DConditionModel(Module):
    def __init__(self,
                 device,
                 dtype,
                 # spatial/channel dims
                 sample_size=None,
                 in_channels=4,
                 out_channels=4,
                 # blocks
                 block_out_channels=(320, 640, 1280, 1280),
                 layers_per_block=2,
                 transformer_layers_per_block=1,
                 # normalization/activations
                 norm_num_groups=32,
                 act_fn="silu",
                 # cross attention
                 cross_attention_dim=768,
                 attention_head_dim=8,
                 num_attention_heads=None, # if none, will be precomputed
                 # downsample/upsample
                 kernel_size=(3,3),
                 padding=(1,1),
                 # time embedding
                 time_embedding_type="positional",
                 time_embedding_dim=None,
                 flip_sin_to_cos=True,
                 freq_shift=0):
        super().__init__()

        if cross_attention_dim % attention_head_dim != 0: raise ValueError()
        if num_attention_heads is None: num_attention_heads = cross_attention_dim // attention_head_dim

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.transformer_layers_per_block = transformer_layers_per_block
        self.norm_num_groups = norm_num_groups
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads

        # Time embedding
        if time_embedding_dim is None:
            time_embedding_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(
            num_channels=block_out_channels[0] * 4,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift
        )

        self.time_embedding = Linear(
            in_dim=block_out_channels[0] * 4,
            out_dim=time_embedding_dim,
            dtype=dtype,
            device=device
        )

        # Input convolution
        self.conv_in = Conv2d(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            dtype=dtype,
            stride=(1,1),
            padding=padding,
            device=device
        )

        # Down blocks
        self.down_blocks = LayerList([])
        output_channel = block_out_channels[0]

        for i, out_channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = out_channel

            if i < len(block_out_channels) - 1:
                # Cross attention down blocks
                down_block = CrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    layers_per_block=layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    num_groups=norm_num_groups,
                    device=device,
                    dtype=dtype
                )
            else:
                # Final down block without cross attention
                down_block = DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    layers_per_block=layers_per_block,
                    num_groups=norm_num_groups,
                    device=device,
                    dtype=dtype
                )

            self.down_blocks.append(down_block)

        # Mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            kernel_size=kernel_size,
            num_layers=transformer_layers_per_block,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            num_groups=norm_num_groups,
            device=device,
            dtype=dtype
        )

        # Up blocks
        self.up_blocks = LayerList([])
        reversed_block_out_channels = list(reversed(block_out_channels))

        output_channel = reversed_block_out_channels[0]
        for i, out_channel in enumerate(reversed_block_out_channels):
            input_channel = output_channel
            output_channel = out_channel

            if i == 0:
                # First up block without cross attention
                up_block = UpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    layers_per_block=layers_per_block,
                    num_groups=norm_num_groups,
                    device=device,
                    dtype=dtype
                )
            else:
                # Cross attention up blocks
                up_block = CrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    layers_per_block=layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    num_groups=norm_num_groups,
                    device=device,
                    dtype=dtype
                )

            self.up_blocks.append(up_block)

        # Output layers
        self.conv_norm_out = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=block_out_channels[0],
            device=device,
            dtype=dtype
        )
        self.conv_act = ops.silu
        self.conv_out = Conv2d(
            kernel_size=kernel_size,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            dtype=dtype,
            stride=(1,1),
            padding=padding,
            device=device
        )


    def __call__(self,
                 sample,
                 timestep,
                 encoder_hidden_states,
                 added_cond_kwargs=None,
                 return_dict=True):
        # 1. Time embedding
        timesteps = timestep
        if not ops.is_tensor(timestep):
            timesteps = ops.constant(timestep, dtype=sample.dtype)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # 2. Pre-process
        sample = self.conv_in(sample)

        # 3. Down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=t_emb)

            down_block_res_samples += res_samples

        # 4. Mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                temb=t_emb,
                encoder_hidden_states=encoder_hidden_states
            )

        # 5. Up
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                sample = up_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample = up_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=t_emb
                )

        # 6. Post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return {"sample": sample}