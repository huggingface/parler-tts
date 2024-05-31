import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
from tqdm import tqdm
from dataset import PROMPTS, DESCRIPTIONS
import time

model_eager = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-expresso",
    attn_implementation="eager",
    torch_dtype=torch.float16
).to("cuda:0")

model_flash = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-expresso",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
).to("cuda:0")


model_sdpa = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-expresso",
    attn_implementation="sdpa",
    torch_dtype=torch.float16
).to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")


for i in range(3):
    print(f"Wramming up decoder")
    z = torch.empty(1, 1024, 8).uniform_(-10,10).to(model_eager.device).to(model_eager.dtype)
    model_eager.audio_encoder.model.decode(z)
    model_flash.audio_encoder.model.decode(z)
    model_sdpa.audio_encoder.model.decode(z)





def generate_speech(model, prompt, description):
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


def benchmark(model):
    device = "cuda:0"
    # define Events that measure start and end of the generate pass
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # reset cuda memory stats and empty cache
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()


    NUM_SAMPLE = 50

    latencies = []

    for i in tqdm(range(len(PROMPTS[:NUM_SAMPLE]))):
        prompt = PROMPTS[i]
        description = DESCRIPTIONS[i]

        start = time.perf_counter()

        _ = generate_speech(model, prompt, description)

        latencies.append(time.perf_counter() - start)

     # get the end time
    end_event.record()
    torch.cuda.synchronize()

    # measure memory footprint and elapsed time
    max_memory = torch.cuda.max_memory_allocated(device)
    elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3

    print('Execution time:', elapsed_time/NUM_SAMPLE, 'seconds')
    print('Max memory footprint', max_memory*1e-9, ' GB')

if __name__ == "__main__":
    print("Benchmark model with Eager Attention")
    benchmark(model_eager)
    print("Benchmark model with Flash Attention 2")
    benchmark(model_flash)
    print("Benchmark model with SDPA Attention")
    benchmark(model_sdpa)
