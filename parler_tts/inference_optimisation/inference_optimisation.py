import time
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import numpy as np
import soundfile as sf

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
print(f"Using {device} for inference")

# Model setup
model_name = "ai4bharat/indic-parler-tts"
max_length = 50

print(f"Loading {model_name}...")
start = time.perf_counter()

# Load model and tokenizers
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
).to(device, dtype=torch_dtype)

tokenizer = AutoTokenizer.from_pretrained(model_name)
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Set pad tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if description_tokenizer.pad_token is None:
    description_tokenizer.pad_token = description_tokenizer.eos_token

# Configure model
model.config.pad_token_id = tokenizer.pad_token_id
if hasattr(model.generation_config.cache_config, 'max_batch_size'):
    model.generation_config.cache_config.max_batch_size = 1
model.generation_config.cache_implementation = "static"

# Compile model
compile_mode = "reduce-overhead"
model.forward = torch.compile(model.forward, mode=compile_mode)

# Warmup
warmup_inputs = tokenizer("Warmup text for compilation",
                         return_tensors="pt",
                         padding="max_length",
                         max_length=max_length).to(device)

model_kwargs = {
    "input_ids": warmup_inputs["input_ids"],
    "attention_mask": warmup_inputs["attention_mask"],
    "prompt_input_ids": warmup_inputs["input_ids"],
    "prompt_attention_mask": warmup_inputs["attention_mask"],
}

for _ in range(2):  # 2 steps for reduce-overhead mode
    _ = model.generate(**model_kwargs)

print(f"Loaded {model_name} in {time.perf_counter() - start:.2f} seconds")

def chunk_text(text, chunk_size=15):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_speech(text, voice_description, output_file="output.wav"):
    start = time.perf_counter()
    chunks = chunk_text(text)

    if len(chunks) <= 15:
        desc_inputs = description_tokenizer(voice_description,
                                          return_tensors="pt",
                                          padding="max_length",
                                          max_length=max_length).to(device)
        prompt_inputs = tokenizer(text,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=max_length).to(device)

        generation = model.generate(
            input_ids=desc_inputs["input_ids"],
            prompt_input_ids=prompt_inputs["input_ids"],
            attention_mask=desc_inputs["attention_mask"],
            prompt_attention_mask=prompt_inputs["attention_mask"]
        ).to(torch.float32)

        audio_arr = generation.cpu().float().numpy().squeeze()
    else:
        all_descriptions = [voice_description] * len(chunks)
        description_inputs = description_tokenizer(all_descriptions,
                                                 return_tensors="pt",
                                                 padding=True).to(device)
        prompts = tokenizer(chunks,
                          return_tensors="pt",
                          padding=True).to(device)

        set_seed(0)
        generation = model.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompts["input_ids"],
            prompt_attention_mask=prompts["attention_mask"],
            do_sample=True,
            return_dict_in_generate=True,
        )
        
        chunk_audios = [audio[:generation.audios_length[i]].cpu().float().numpy().squeeze()
                       for i, audio in enumerate(generation.sequences)]
        audio_arr = np.concatenate(chunk_audios)

    sf.write(output_file, audio_arr, model.config.sampling_rate)
    print(f"Generated audio saved to {output_file} in {time.perf_counter() - start:.2f} seconds")
    return audio_arr

# Example usage
if __name__ == "__main__":
    text = "Hello, this is a test of the text-to-speech system with a simple Python implementation."
    voice_description = "A clear, neutral male voice"
    generate_speech(text, voice_description)