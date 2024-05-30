import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
from tqdm import tqdm
from dataset import PROMPTS, DESCRIPTIONS
import time

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-expresso",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
).to("cuda:0")


for i in range(3):
    print(f"Wramming up decoder")
    z = torch.empty(1, 1024, 8).uniform_(-10,10).type(torch.FloatTensor).to(model.device).to(model.dtype)
    model.audio_encoder.model.decode(z)


tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")

def generate_speech(prompt, description):
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to("cuda:0")
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    generation_config = model.generation_config

    # Generate first second
    generation_config.max_length =  86 # default 2580. WTF

    _ = model.generate(input_ids=input_ids, 
        prompt_input_ids=prompt_input_ids,
        generation_config=generation_config,
        use_cache=True,
        past_key_values = None,
    )


if __name__ == "__main__":
    NUM_SAMPLE = 20

    latencies = []

    for i in tqdm(range(len(PROMPTS[:NUM_SAMPLE]))):
        prompt = PROMPTS[i]
        description = DESCRIPTIONS[i]

        start = time.perf_counter()

        _ = generate_speech(prompt, description)

        latencies.append(time.perf_counter() - start)


    print(f"AVG latency = {sum(latencies) / len(latencies)}")




