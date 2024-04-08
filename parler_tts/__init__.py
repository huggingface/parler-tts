from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .modeling_parler_tts import (
    ParlerTTSForCausalLM,
    ParlerTTSForConditionalGeneration,
    apply_delay_pattern_mask,
    build_delay_pattern_mask,
)

from .dac_wrapper import DACConfig, DACModel
from transformers import AutoConfig, AutoModel

AutoConfig.register("dac", DACConfig)
AutoModel.register(DACConfig, DACModel)