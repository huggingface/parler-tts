# coding=utf-8
# Copyright 2024 and The HuggingFace Inc. team. All rights reserved.
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
""" Parler-TTS model configuration"""

from transformers import AutoConfig, logging
from transformers.configuration_utils import PretrainedConfig

from importlib.metadata import version
from packaging.version import Version

use_dac_on_the_hub = Version(version("transformers")) > Version("4.44.2dev")

logger = logging.get_logger(__name__)

PARLER_TTS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "parler-tts/parler-tts-mini-v1": "https://huggingface.co/parler-tts/parler-tts-mini-v1/resolve/main/config.json",
    # See all ParlerTTS models at https://huggingface.co/models?filter=parler_tts
}


class ParlerTTSDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`ParlerTTSDecoder`]. It is used to instantiate a
    Parler-TTS decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Parler-TTS
    [parler-tts/parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2049):
            Vocabulary size of the ParlerTTSDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`ParlerTTSDecoder`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        num_cross_attention_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention in the cross-attention layers.
            If it is not specified, will default to `num_key_value_heads`.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(hidden_size).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        num_codebooks (`int`, *optional*, defaults to 4):
            The number of parallel codebooks forwarded to the model.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether input and output word embeddings should be tied.
        rope_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to use ROPE or absolute positional embeddings.
        rope_theta (`float`, *optional*, defaults to 100000.0):
            The base period of the RoPE embeddings.
        cross_attention_implementation_strategy (`str`, *optional*):
            If not specified, the cross-attention implementation will be the same as `_attn_implementation`. If `always_eager`, it will always be the eager implementation. If `always_sdpa`, it will always be the sdpa implementation.
        use_fused_lm_heads(`bool`, *optional*, defaults to `False`):
            Whether to fuse audio LM heads instead of applying them sequentially.
        codebook_weights(`List[int]`, *optional*):
            Weights applied to each codebook when computing the loss.
    """

    model_type = "parler_tts_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=2049,  # vocab size = 2048 (encodec vocab size) + 1 (eos)
        max_position_embeddings=2048,
        num_hidden_layers=24,
        ffn_dim=4096,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_cross_attention_key_value_heads=None,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        scale_embedding=False,
        num_codebooks=4,
        pad_token_id=2048,
        bos_token_id=2049,
        eos_token_id=2048,
        tie_word_embeddings=False,
        rope_embeddings=False,
        rope_theta=10_000.0,
        cross_attention_implementation_strategy=None,
        use_fused_lm_heads=False,
        codebook_weights=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if num_cross_attention_key_value_heads is None:
            num_cross_attention_key_value_heads = num_key_value_heads
        self.num_cross_attention_key_value_heads = num_cross_attention_key_value_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_codebooks = num_codebooks
        self.rope_embeddings = rope_embeddings
        self.rope_theta = rope_theta
        self.cross_attention_implementation_strategy = cross_attention_implementation_strategy
        self.use_fused_lm_heads = use_fused_lm_heads
        self.codebook_weights = codebook_weights

        if codebook_weights is not None and len(codebook_weights) != num_codebooks:
            raise ValueError(f"`codebook_weights` has length {len(codebook_weights)} when it should be of length {num_codebooks}.")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ParlerTTSConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParlerTTSModel`]. It is used to instantiate a
    Parler-TTS model according to the specified arguments, defining the text encoder, audio encoder and Parler-TTS decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 1024):
            Vocabulary size of the prompt token ids. Defines the number of different tokens that can be
            represented by the `prompt_inputs_ids`.
        prompt_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether to use cross-attention conditioning for the prompt (as well as the description).
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **text_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the text encoder config.
                - **audio_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Example:

    ```python
    >>> from transformers import (
    ...     ParlerTTSConfig,
    ...     ParlerTTSDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     ParlerTTSForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = ParlerTTSDecoderConfig()

    >>> configuration = ParlerTTSConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a ParlerTTSForConditionalGeneration (with random weights) from the parler-tts/parler-tts-mini-v1 style configuration
    >>> model = ParlerTTSForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("parler_tts-model")

    >>> # loading model and config from pretrained folder
    >>> parler_tts_config = ParlerTTSConfig.from_pretrained("parler_tts-model")
    >>> model = ParlerTTSForConditionalGeneration.from_pretrained("parler_tts-model", config=parler_tts_config)
    ```"""

    model_type = "parler_tts"
    is_composition = True

    def __init__(self, vocab_size=1024, prompt_cross_attention=False, **kwargs):
        super().__init__(**kwargs)
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")

        text_encoder_config = kwargs.pop("text_encoder")
        text_encoder_model_type = text_encoder_config.pop("model_type")

        audio_encoder_config = kwargs.pop("audio_encoder")
        audio_encoder_model_type = audio_encoder_config.pop("model_type")

        model_version = kwargs.get("transformers_version", None)
        if model_version is not None and Version(model_version) <= Version("4.44.2dev") and use_dac_on_the_hub and audio_encoder_model_type=="dac":
            # here we have to manually change model type if DAC based on transformers version
            audio_encoder_model_type = "dac_on_the_hub"

        decoder_config = kwargs.pop("decoder")

        self.vocab_size = vocab_size
        self.prompt_cross_attention = prompt_cross_attention
        self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **text_encoder_config)
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        self.decoder = ParlerTTSDecoderConfig(**decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        audio_encoder_config: PretrainedConfig,
        decoder_config: ParlerTTSDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`ParlerTTSConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`ParlerTTSConfig`]: An instance of a configuration object
        """

        return cls(
            text_encoder=text_encoder_config.to_dict(),
            audio_encoder=audio_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs,
        )

    @property
    # This is a property because you might want to change the codec model on the fly
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate