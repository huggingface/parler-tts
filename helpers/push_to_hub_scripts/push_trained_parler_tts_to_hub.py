from transformers import AutoFeatureExtractor, AutoTokenizer

from parler_tts import ParlerTTSForConditionalGeneration


path = "TODO"
repo_id = "parler_tts_600M"


AutoFeatureExtractor.from_pretrained("ylacombe/dac_44khZ_8kbps").push_to_hub(repo_id)
AutoTokenizer.from_pretrained("google/t5-v1_1-base").push_to_hub(repo_id)

ParlerTTSForConditionalGeneration.from_pretrained(path).push_to_hub(repo_id)
