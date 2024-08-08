__version__ = "0.2"


from transformers import AutoConfig, AutoModel

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .dac_wrapper import DACConfig, DACModel
from .modeling_parler_tts import (
    ParlerTTSForCausalLM,
    ParlerTTSForConditionalGeneration,
    apply_delay_pattern_mask,
    build_delay_pattern_mask,
)

from .streamer import ParlerTTSStreamer

AutoConfig.register("dac", DACConfig)
AutoModel.register(DACConfig, DACModel)
