# clip.py
# author: Aathreya Kadambi
# description: Implement CLIPTextEncoder with MAX Graph API analogously to the implementation in the transformers package.

import numpy as np
from numpy.typing import NDArray

from max.nn import (
    MLP,
    Embedding,
    Linear,
    Module,
    LayerNorm,
    Layer,
    LayerList,
    TransformerBlock,
)
from max.nn.attention import MultiheadAttention, MHAMaskVariant
from max.dtype import DType

from max.graph import TensorValue, ops, Graph, TensorType, DeviceRef
from max.graph.weights import WeightData
from max.driver import Tensor
from max.engine import InferenceSession
from max.driver import CPU, Accelerator

import math

class CLIPMLP(Module):
    def __init__(self,
                 hidden_dim,
                 latent_dim,
                 dtype=DType.float32,
                 device=None # TODO: extend to allow for multiple devices
    ):
        self.fc1 = Linear(
            in_dim=hidden_dim,
            out_dim=latent_dim,
            dtype=dtype,
            device=device,
            name="fc1",
            has_bias=True
        )
        self.fc2 = Linear(
            in_dim=latent_dim,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            name="fc2",
            has_bias=True
        )
        self.activation_fn = ops.gelu

        super().__init__()
    
    def __call__(self, x: TensorValue):
        x = self.fc1(x)
        x = self.activation_fn(x, approximate='none')
        return self.fc2(x)

# Based on TransformerBlock # https://github.com/modular/modular/blob/main/max/python/max/nn/transformer/transformer.py#L29
class CLIPEncoderLayer(Module):
    def __init__(
        self,
        attention: Module,
        mlp: Layer,
        attention_norm: Layer,
        mlp_norm: Layer,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm

    def __call__(
        self,
        x: TensorValue
    ) -> TensorValue:
        il = self.input_layernorm(x)
        attn_out = self.self_attn(
            il, # MAIN CHANGE! (aside from now removing residual multiplier)
            mask_variant=MHAMaskVariant.CAUSAL_MASK
        )
        h = x + attn_out
        mlp = self.mlp(self.post_attention_layernorm(h))
        return h + mlp

class CLIPTextEncoder(Module): # if we extend Transformer, we would have to rewrite the Transformer piece to 
    def __init__(self, 
                 vocab_size=49408,
                 hidden_dim=768,
                 max_len=77, 
                 n_heads=8, 
                 n_layers=12,
                 devices=None,
                 dtype=DType.float32
    ):
        super().__init__()
        self.dtype=dtype
        self.devices=devices
        self.max_len=max_len

        # compare to transformers package implementation in CLIPTextEmbeddings
        self.token_embedding = Embedding(vocab_size=vocab_size, hidden_dim=hidden_dim, dtype=dtype, device=devices[0])
        self.pos_embedding = Embedding(vocab_size=max_len, hidden_dim=hidden_dim, dtype=dtype, device=devices[0])

        self.layers = [
            CLIPEncoderLayer(
                attention=MultiheadAttention(
                    n_heads,
                    hidden_dim,
                    devices=devices,
                    dtype=dtype,
                    qkv_has_bias=True,
                    o_proj_has_bias=True
                ),
                mlp=CLIPMLP(
                    hidden_dim,
                    4*hidden_dim, # TODO: check this, I just defined this to work with my case
                    dtype=dtype,
                    device=devices[0]
                ),
                attention_norm=LayerNorm(hidden_dim, devices, dtype),
                mlp_norm=LayerNorm(hidden_dim, devices, dtype),
            )
            for _ in range(n_layers)
        ]
        self.layers = LayerList(self.layers)
        self.final_norm = LayerNorm(hidden_dim, devices, dtype)

        self.g = None

    def __call__(self,
                 tokens: TensorValue
    ) -> TensorValue:
        pos_ids = ops.range(0, tokens.shape[1], dtype=DType.int64, device=self.devices[0])
        pos_emb = self.pos_embedding(pos_ids)
        pos_emb = ops.reshape(pos_emb, (1, pos_emb.shape[0], pos_emb.shape[1]))
        tok_emb = self.token_embedding(tokens)
        h = tok_emb + pos_emb # TODO: might need to be careful about batching
        for idx, layer in enumerate(self.layers):
            h = layer(h)
        h = self.final_norm(h)
        return h

    def load_clip_state(self, hf_state: dict[str, np.ndarray]):
        """
        Map HuggingFace CLIPTextModel weights to this module.
        """
        new_sd = CLIPTextEncoder._convert_clip_state_dict(hf_state)
        self.load_state_dict(new_sd)

    # for loading state dict from a huggingface model
    def _convert_clip_state_dict(hf_state: dict[str, np.ndarray]) -> dict[str, WeightData]:
        new_sd = {}
        for hf_name, arr in hf_state.items():
            arr = arr.cpu().numpy()
            if hf_name.startswith("text_model.embeddings.token_embedding.weight"):
                new_sd["token_embedding.weight"] = WeightData.from_numpy(arr, name="token_embedding.weight")
            elif hf_name.startswith("text_model.embeddings.position_embedding.weight"):
                new_sd["pos_embedding.weight"] = WeightData.from_numpy(arr, name="pos_embedding.weight")
            elif hf_name.startswith(f"text_model.encoder.layers"): 
                for i in range(23):
                    if hf_name.startswith(f"text_model.encoder.layers.{i}.self_attn"):
                        for piece in [ "q_proj", "k_proj", "v_proj", "out_proj" ]:
                            if hf_name.startswith(f"text_model.encoder.layers.{i}.self_attn.{piece}.weight"):
                                if piece == "out_proj": piece = "o_proj"
                                new_sd[f"layers.{i}.self_attn.{piece}.weight"] = WeightData.from_numpy(arr, name=f"layers.{i}.self_attn.{piece}.weight")
                                break
                            elif hf_name.startswith(f"text_model.encoder.layers.{i}.self_attn.{piece}.bias"):
                                if piece == "out_proj": piece = "o_proj"
                                new_sd[f"layers.{i}.self_attn.{piece}.bias"] = WeightData.from_numpy(arr, name=f"layers.{i}.self_attn.{piece}.bias")
                                break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.layer_norm1.weight"):
                        new_sd[f"layers.{i}.input_layernorm.weight"] = WeightData.from_numpy(arr, name=f"layers.{i}.input_layernorm.weight")
                        break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.layer_norm1.bias"):
                        new_sd[f"layers.{i}.input_layernorm.bias"] = WeightData.from_numpy(arr, name=f"layers.{i}.input_layernorm.bias")
                        break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.mlp.fc1.weight"):
                        new_sd[f"layers.{i}.mlp.fc1.fc1.weight"] = WeightData.from_numpy(arr, name=f"layers.{i}.mlp.fc1.fc1.weight")
                        break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.mlp.fc1.bias"):
                        new_sd[f"layers.{i}.mlp.fc1.fc1.bias"] = WeightData.from_numpy(arr, name=f"layers.{i}.mlp.fc1.fc1.bias")
                        break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.mlp.fc2.weight"):
                        new_sd[f"layers.{i}.mlp.fc2.fc2.weight"] = WeightData.from_numpy(arr, name=f"layers.{i}.mlp.fc2.fc2.weight")
                        break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.mlp.fc2.bias"):
                        new_sd[f"layers.{i}.mlp.fc2.fc2.bias"] = WeightData.from_numpy(arr, name=f"layers.{i}.mlp.fc2.fc2.bias")
                        break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.layer_norm2.weight"):
                        new_sd[f"layers.{i}.post_attention_layernorm.weight"] = WeightData.from_numpy(arr, name=f"layers.{i}.post_attention_layernorm.weight")
                        break
                    elif hf_name.startswith(f"text_model.encoder.layers.{i}.layer_norm2.bias"):
                        new_sd[f"layers.{i}.post_attention_layernorm.bias"] = WeightData.from_numpy(arr, name=f"layers.{i}.post_attention_layernorm.bias")
                        break
            elif hf_name.startswith(f"text_model.final_layer_norm.weight"):
                new_sd["final_norm.weight"] =  WeightData.from_numpy(arr, name="final_norm.weight")
            elif hf_name.startswith(f"text_model.final_layer_norm.bias"):
                new_sd["final_norm.bias"] =  WeightData.from_numpy(arr, name="final_norm.bias")
        return new_sd

    def _build_graph(self):
        self.g = Graph(
            "clip_text_encoder",
            self,
            input_types=[TensorType(DType.int64, 
                                    (1,self.max_len), 
                                    device=DeviceRef.from_device(self.devices[0])
                        )]
        )

    def execute(self,
                tokens: NDArray[np.long],
    ) -> Tensor:
        if self.g is None:
            self._build_graph()

        tokens_tensor = Tensor.from_numpy(tokens).to(self.devices[0])
        session = InferenceSession(devices=self.devices)
        model = session.load(self.g, weights_registry=self.state_dict())
        result = model.execute(tokens_tensor)[0]
        assert isinstance(result, Tensor)
        return result.to(CPU()) if self.devices[0] == Accelerator() else result