import dac
from transformers import AutoConfig, AutoModel, EncodecFeatureExtractor

from parler_tts import DACConfig, DACModel
from transformers import AutoConfig, AutoModel
from transformers import EncodecFeatureExtractor

from importlib.metadata import version
from packaging.version import Version

if Version(version("transformers"))<= Version("4.44.2dev"):
    AutoConfig.register("dac", DACConfig)
else:
    AutoConfig.register("dac_on_the_hub", DACConfig)

AutoModel.register(DACConfig, DACModel)

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

hf_dac = DACModel(DACConfig())
hf_dac.model.load_state_dict(model.state_dict())

hf_dac.push_to_hub("parler-tts/dac_44khZ_8kbps")
EncodecFeatureExtractor(sampling_rate=44100).push_to_hub("parler-tts/dac_44khZ_8kbps")
