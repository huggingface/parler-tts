__version__ = "0.2.2"


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

from importlib.metadata import version
from packaging.version import Version

if Version(version("transformers"))<= Version("4.44.2dev"):
    AutoConfig.register("dac", DACConfig)
else:
    AutoConfig.register("dac_on_the_hub", DACConfig)

AutoModel.register(DACConfig, DACModel)
