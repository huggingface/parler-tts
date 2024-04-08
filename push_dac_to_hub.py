import dac

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

from parler_tts import DACConfig, DACModel

hf_dac = DACModel(DACConfig())
hf_dac.model.load_state_dict(model.state_dict())

from transformers import AutoConfig, AutoModel

AutoConfig.register("dac", DACConfig)
AutoModel.register(DACConfig, DACModel)

hf_dac.push_to_hub("ylacombe/dac_44khZ_8kbps")

# DACConfig.register_for_auto_class()
# DACModel.register_for_auto_class("AutoModel")

from transformers import EncodecFeatureExtractor

EncodecFeatureExtractor(sampling_rate=44100).push_to_hub("ylacombe/dac_44khZ_8kbps")
