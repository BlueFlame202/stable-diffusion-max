
from max.nn import (
    Module,
    Linear,
)
from max.graph import TensorValue, ops

class UNetAttention(Module):
    def __init__(self,
                 device,
                 dtype,
                 query_dim,
                 cross_attention_dim=None,
                 heads=8,
                 dim_head=64,
                 qkv_has_bias=False,
                 o_proj_has_bias=True):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_q = Linear(
            in_dim=query_dim,
            out_dim=inner_dim,
            has_bias=qkv_has_bias,
            dtype=dtype,
            device=device
        )

        if cross_attention_dim is not None:
            self.to_k = Linear(
                in_dim=cross_attention_dim,
                out_dim=inner_dim,
                has_bias=qkv_has_bias,
                dtype=dtype,
                device=device
            )
            self.to_v = Linear(
                in_dim=cross_attention_dim,
                out_dim=inner_dim,
                has_bias=qkv_has_bias,
                dtype=dtype,
                device=device
            )
        else:
            self.to_k = Linear(
                in_dim=query_dim,
                out_dim=inner_dim,
                has_bias=qkv_has_bias,
                dtype=dtype,
                device=device
            )
            self.to_v = Linear(
                in_dim=query_dim,
                out_dim=inner_dim,
                has_bias=qkv_has_bias,
                dtype=dtype,
                device=device
            )

        self.to_out = Linear(
            in_dim=inner_dim,
            out_dim=query_dim,
            has_bias=o_proj_has_bias,
            dtype=dtype,
            device=device
        )

    def __call__(self, x, encoder_hidden_states=None):
        batch, seq_len, dim = x.shape

        q = self.to_q(x)
        q = ops.reshape(q, (batch, seq_len, self.heads, self.dim_head))
        q = ops.transpose(q, (0, 2, 1, 3))  # (batch, heads, seq_len, dim_head)

        if encoder_hidden_states is not None:
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            _, encoder_seq_len, _ = encoder_hidden_states.shape
        else:
            k = self.to_k(x)
            v = self.to_v(x)
            encoder_seq_len = seq_len

        k = ops.reshape(k, (batch, encoder_seq_len, self.heads, self.dim_head))
        k = ops.transpose(k, (0, 2, 1, 3))  # (batch, heads, encoder_seq_len, dim_head)

        v = ops.reshape(v, (batch, encoder_seq_len, self.heads, self.dim_head))
        v = ops.transpose(v, (0, 2, 1, 3))  # (batch, heads, encoder_seq_len, dim_head)

        # Attention
        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)

        out = ops.matmul(attn_weights, v)
        out = ops.transpose(out, (0, 2, 1, 3))  # (batch, seq_len, heads, dim_head)
        out = ops.reshape(out, (batch, seq_len, self.heads * self.dim_head))

        return self.to_out(out)