import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, OrderedDict, List
import zipfile
import soundfile as sf
import torch
from fastapi import Body, FastAPI, HTTPException, Response
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import numpy as np
from config import SPEED, ResponseFormat, config
from logger import logger
import uvicorn
import argparse
from fastapi.responses import RedirectResponse, StreamingResponse
import io
import os
import logging

# Device setup
if torch.cuda.is_available():
    device = "cuda:0"
    logger.info("GPU will be used for inference")
else:
    device = "cpu"
    logger.info("CPU will be used for inference")
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

# Check CUDA availability and version
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else None

if torch.cuda.is_available():
    device_idx = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_idx)
    compute_capability_float = float(f"{capability[0]}.{capability[1]}")
    print(f"CUDA version: {cuda_version}")
    print(f"CUDA Compute Capability: {compute_capability_float}")
else:
    print("CUDA is not available on this system.")

class ModelManager:
    def __init__(self):
        self.model_tokenizer: OrderedDict[
            str, tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]
        ] = OrderedDict()
        self.max_length = 50

    def load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        logger.debug(f"Loading {model_name}...")
        start = time.perf_counter()
        
        model_name = "ai4bharat/indic-parler-tts"
        attn_implementation = "flash_attention_2"
        
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation=attn_implementation
        ).to(device, dtype=torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        # Set pad tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if description_tokenizer.pad_token is None:
            description_tokenizer.pad_token = description_tokenizer.eos_token

        # Update model configuration
        model.config.pad_token_id = tokenizer.pad_token_id
        # Update for deprecation: use max_batch_size instead of batch_size
        if hasattr(model.generation_config.cache_config, 'max_batch_size'):
            model.generation_config.cache_config.max_batch_size = 1
        model.generation_config.cache_implementation = "static"

        # Compile the model
        ##compile_mode = "default"
        compile_mode = "reduce-overhead"
        
        model.forward = torch.compile(model.forward, mode=compile_mode)

        # Warmup
        warmup_inputs = tokenizer("Warmup text for compilation", 
                                return_tensors="pt", 
                                padding="max_length", 
                                max_length=self.max_length).to(device)
        
        model_kwargs = {
            "input_ids": warmup_inputs["input_ids"],
            "attention_mask": warmup_inputs["attention_mask"],
            "prompt_input_ids": warmup_inputs["input_ids"],
            "prompt_attention_mask": warmup_inputs["attention_mask"],
        }
        
        n_steps = 1 if compile_mode == "default" else 2
        for _ in range(n_steps):
            _ = model.generate(**model_kwargs)

        logger.info(
            f"Loaded {model_name} with Flash Attention and compilation in {time.perf_counter() - start:.2f} seconds"
        )
        return model, tokenizer, description_tokenizer

    def get_or_load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        if model_name not in self.model_tokenizer:
            logger.info(f"Model {model_name} isn't already loaded")
            if len(self.model_tokenizer) == config.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.model_tokenizer[next(iter(self.model_tokenizer))]
            self.model_tokenizer[model_name] = self.load_model(model_name)
        return self.model_tokenizer[model_name]

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.lazy_load_model:
        model_manager.get_or_load_model(config.model)
    yield

app = FastAPI(lifespan=lifespan)

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

@app.post("/v1/audio/speech")
async def generate_audio(
    input: Annotated[str, Body()] = config.input,
    voice: Annotated[str, Body()] = config.voice,
    model: Annotated[str, Body()] = config.model,
    response_format: Annotated[ResponseFormat, Body(include_in_schema=False)] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = time.perf_counter()

    chunk_size = 15
    all_chunks = chunk_text(input, chunk_size)

    if len(all_chunks) <= chunk_size:
        desc_inputs = description_tokenizer(voice,
                                          return_tensors="pt",
                                          padding="max_length",
                                          max_length=model_manager.max_length).to(device)
        prompt_inputs = tokenizer(input,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=model_manager.max_length).to(device)
        
        # Use the tensor fields directly instead of BatchEncoding object
        input_ids = desc_inputs["input_ids"]
        attention_mask = desc_inputs["attention_mask"]
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_attention_mask = prompt_inputs["attention_mask"]

        generation = tts.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=prompt_attention_mask
        ).to(torch.float32)

        audio_arr = generation.cpu().float().numpy().squeeze()
    else:
        all_descriptions = [voice] * len(all_chunks)
        description_inputs = description_tokenizer(all_descriptions,
                                                 return_tensors="pt",
                                                 padding=True).to(device)
        prompts = tokenizer(all_chunks,
                          return_tensors="pt",
                          padding=True).to(device)

        set_seed(0)
        generation = tts.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompts["input_ids"],
            prompt_attention_mask=prompts["attention_mask"],
            do_sample=True,
            return_dict_in_generate=True,
        )
        
        chunk_audios = []
        for i, audio in enumerate(generation.sequences):
            audio_data = audio[:generation.audios_length[i]].cpu().float().numpy().squeeze()
            chunk_audios.append(audio_data)
        audio_arr = np.concatenate(chunk_audios)

    device_str = str(device)
    logger.info(
        f"Took {time.perf_counter() - start:.2f} seconds to generate audio for {len(input.split())} words using {device_str.upper()}"
    )

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_arr, tts.config.sampling_rate, format=response_format)
    audio_buffer.seek(0)

    return StreamingResponse(audio_buffer, media_type=f"audio/{response_format}")

def create_in_memory_zip(file_data):
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, 'w') as zipf:
        for file_name, data in file_data.items():
            zipf.writestr(file_name, data)
    in_memory_zip.seek(0)
    return in_memory_zip

@app.post("/v1/audio/speech_batch")
async def generate_audio_batch(
    input: Annotated[List[str], Body()] = config.input,
    voice: Annotated[List[str], Body()] = config.voice,
    model: Annotated[str, Body(include_in_schema=False)] = config.model,
    response_format: Annotated[ResponseFormat, Body()] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = time.perf_counter()

    chunk_size = 15
    all_chunks = []
    all_descriptions = []
    for i, text in enumerate(input):
        chunks = chunk_text(text, chunk_size)
        all_chunks.extend(chunks)
        all_descriptions.extend([voice[i]] * len(chunks))

    description_inputs = description_tokenizer(all_descriptions,
                                             return_tensors="pt",
                                             padding=True).to(device)
    prompts = tokenizer(all_chunks,
                       return_tensors="pt",
                       padding=True).to(device)

    set_seed(0)
    generation = tts.generate(
        input_ids=description_inputs["input_ids"],
        attention_mask=description_inputs["attention_mask"],
        prompt_input_ids=prompts["input_ids"],
        prompt_attention_mask=prompts["attention_mask"],
        do_sample=True,
        return_dict_in_generate=True,
    )

    audio_outputs = []
    current_index = 0
    for i, text in enumerate(input):
        chunks = chunk_text(text, chunk_size)
        chunk_audios = []
        for j in range(len(chunks)):
            audio_arr = generation.sequences[current_index][:generation.audios_length[current_index]].cpu().float().numpy().squeeze()
            chunk_audios.append(audio_arr)
            current_index += 1
        combined_audio = np.concatenate(chunk_audios)
        audio_outputs.append(combined_audio)

    file_data = {}
    for i, audio in enumerate(audio_outputs):
        file_name = f"out_{i}.{response_format}"
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, tts.config.sampling_rate, format=response_format)
        audio_bytes.seek(0)
        file_data[file_name] = audio_bytes.read()

    in_memory_zip = create_in_memory_zip(file_data)

    logger.info(
        f"Took {time.perf_counter() - start:.2f} seconds to generate audio"
    )

    return StreamingResponse(in_memory_zip, media_type="application/zip")

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for TTS.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)