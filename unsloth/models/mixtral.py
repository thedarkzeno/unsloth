# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .llama import *
from ._utils import __version__

from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralBLockSparseTop2MLP,
    MixtralSparseMoeBlock,
    MixtralModel,
    MixtralForCausalLM,
)
# For Pytorch 2.1.1
try:
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralSdpaAttention,
        MixtralFlashAttention2,
    )
except:
    MixtralSdpaAttention   = MixtralAttention
    MixtralFlashAttention2 = MixtralAttention
pass


def MixtralAttention_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    bool = False,
    use_cache:            bool = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    Q, K, V = self.apply_qkv(self, hidden_states)

    # Check for inference
    if use_cache and past_key_value is not None and q_len == 1:
        A, past_key_value = LlamaAttention_fast_forward_inference(
            self,
            hidden_states,
            past_key_value,
            position_ids,
        )
        return A, None, past_key_value
    pass

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is None:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, seq_len = kv_seq_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)
    pass

    if past_key_value is not None:
        # reuse k, v, self_attention
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (not HAS_FLASH_ATTENTION):
        # Xformers memory efficient attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        M = bsz * q_len

        has_swa = isinstance(causal_mask, xformers.attn_bias.BlockDiagonalCausalMask)

        # Group query attention
        K = K  .view(bsz, q_len, n_kv_heads,        1, head_dim)
        V = V  .view(bsz, q_len, n_kv_heads,        1, head_dim)
        K = K.expand(bsz, q_len, n_kv_heads, n_groups, head_dim)
        V = V.expand(bsz, q_len, n_kv_heads, n_groups, head_dim)
        if hidden_states.requires_grad:
            K = K.reshape(bsz, q_len, n_heads, head_dim)
            V = V.reshape(bsz, q_len, n_heads, head_dim)

            if has_swa:
                Q = Q.view(1, M, n_heads, head_dim)
                K = K.view(1, M, n_heads, head_dim)
                V = V.view(1, M, n_heads, head_dim)
            pass
        else:
            # Xformers does support the forward pass though
            Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)

            if has_swa:
                Q = Q.view(1, M, n_kv_heads, n_groups, head_dim)
                K = K.view(1, M, n_kv_heads, n_groups, head_dim)
                V = V.view(1, M, n_kv_heads, n_groups, head_dim)
            pass
        pass

        A = xformers_attention(Q, K, V, attn_bias = causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        sw = getattr(self.config, "sliding_window")
        sw = q_len if sw is None else sw
        window = (-1, -1) if (q_len <= sw) else (sw, sw)
        A = flash_attn_func(Q, K, V, causal = True, window_size = window)
    else:
        # Grouped query attention
        # if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
        K = K.reshape(bsz, n_heads, q_len, head_dim)
        V = V.reshape(bsz, n_heads, q_len, head_dim)
        # pass
        # Needs (batch_size, n_heads, seq_len, head_dim)
        # is_casual and attention_mask must not be both set!
        A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False)
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2)
    pass
    
    attn_output = A.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


def MixtralForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    if causal_mask is None:
        bsz, q_len = input_ids.shape
        sliding_window = getattr(self.config, "sliding_window")
        if sliding_window is None or sliding_window <= 0:
            causal_mask = xformers.attn_bias.LowerTriangularMask()
        elif q_len <= sliding_window:
            causal_mask = xformers.attn_bias.LowerTriangularMask()
        else:
            # Fix from https://github.com/Rypo
            causal_mask = xformers.attn_bias.BlockDiagonalCausalMask\
                .from_seqlens([q_len]*bsz)\
                .make_local_attention(window_size = sliding_window)
    pass

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        shift_logits = logits
        if not hasattr(self, "extra_ignored_labels"):
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda")
        pass
        
        shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )
    pass

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
pass

def MixtralDecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,

    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    residual = hidden_states

    # hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
    hidden_states = rms_norm_fn(
                hidden_states,
                self.input_layernorm.weight,
                None,
                eps=self.input_layernorm.variance_epsilon,
                residual=None,
                prenorm=False,
                residual_in_fp32=False,
            )

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
    

    hidden_states, router_logits = self.block_sparse_moe(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs
pass

def MixtralMLP_fast_forward(self, x):
    return FAST_MLP.apply(x, self.w1.weight, self.w3.weight, self.w2.weight)

class FastMixtralModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        MixtralAttention      .forward = MixtralAttention_fast_forward
        MixtralSdpaAttention  .forward = MixtralAttention_fast_forward
        MixtralFlashAttention2.forward = MixtralAttention_fast_forward
        MixtralDecoderLayer   .forward = MixtralDecoderLayer_fast_forward
        # MixtralBLockSparseTop2MLP            .forward = MixtralMLP_fast_forward
        MixtralModel          .forward = LlamaModel_fast_forward
        MixtralForCausalLM    .forward = MixtralForCausalLM_fast_forward
        PeftModelForCausalLM  .forward = PeftModelForCausalLM_fast_forward
        return
    pass


    @staticmethod
    def from_pretrained(
        model_name = "unsloth/mistral-7b-bnb-4bit",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = False,
        token = None,
        device_map = "sequential",
        rope_scaling = None, # Mistral does not support RoPE scaling
        fix_tokenizer = True,
    ): 
        if rope_scaling is not None:
            logger.warning_once("Unsloth: Mistral models do not support RoPE scaling.")

        SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
           f"==((====))==  Unsloth: Fast Mistral patching release {__version__}\n"\
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB\n"\
           f"O^O/ \_/ \\    CUDA capability = {gpu_stats.major}.{gpu_stats.minor}. Xformers = {xformers_version}. FA = {HAS_FLASH_ATTENTION}.\n"\
           f"\        /    Pytorch version: {torch.__version__}. CUDA Toolkit = {torch.version.cuda}\n"\
           f' "-____-"     bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. Platform = {platform_system}\n'
        logger.warning_once(statistics)
        FastMixtralModel.pre_patch()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = device_map,
            torch_dtype = dtype,
            quantization_config = bnb_config,
            token = token,
            # rope_scaling = rope_scaling,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length = max_seq_length,
            padding_side = "right",
            token = token,
        )

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = FastMixtralModel.post_patch(model)

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o   = original_apply_o
        pass

        # Save max_seq_length
        max_position_embeddings = max(max_seq_length, model.config.max_position_embeddings)
        model.max_seq_length = max_position_embeddings
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_position_embeddings
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_position_embeddings

        # We check the tokenizer first for errors
        if fix_tokenizer:
            tokenizer = check_tokenizer(
                model = model,
                tokenizer = tokenizer,
                model_name = model_name,
                model_max_length = max_seq_length,
                padding_side = "right",
                token = token,
            )
        pass
        return model, tokenizer
    pass
pass
