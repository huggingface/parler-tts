from datasets import load_dataset

dataset = load_dataset("parler-tts/libritts_r_tags_tagged_10k_generated", 'clean')

PROMPTS = dataset['test.clean']['text']
DESCRIPTIONS = dataset['test.clean']['text_description']

