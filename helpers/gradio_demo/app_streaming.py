import io
import math
from queue import Queue
from threading import Thread
from typing import Optional
import time

import numpy as np
import gradio as gr
import torch

from pydub import AudioSegment
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
from transformers.generation.streamers import BaseStreamer
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float32 if device != "cpu" else torch.float32

repo_id = "parler-tts/parler-tts-mini-v1"


model = ParlerTTSForConditionalGeneration.from_pretrained(
    repo_id,
    torch_dtype=torch_dtype,
    attn_implementation="sdpa",
).to(device)

compile_mode = "max-autotune-no-cudagraphs"  # chose "reduce-overhead" for 3 to 4x speed-up
model.generation_config.cache_implementation = "static"
# model.forward = torch.compile(model.forward, fullgraph=True, mode=compile_mode)


tokenizer = AutoTokenizer.from_pretrained(repo_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)


SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 42
TOKENIZER_MAX_LENGTH = 150


# warmup
inputs = tokenizer(
    "Remember - this is only the first iteration of the model! To improve the prosody and naturalness of the speech further, we're scaling up the amount of training data by a factor of five times.",
    return_tensors="pt",
    padding="max_length",
    max_length=TOKENIZER_MAX_LENGTH,
).to(device)

model_kwargs = {
    **inputs,
    "prompt_input_ids": inputs.input_ids,
    "prompt_attention_mask": inputs.attention_mask,
}

generation_config = model.generation_config
generation_config.max_new_tokens = 2564

n_steps = 1 if compile_mode == "default" else 2
for _ in range(n_steps):
    _ = model.generate(**model_kwargs, generation_config=generation_config)


default_text = "Please surprise me and speak in whatever voice you enjoy."
examples = [
    [
        "Remember - this is only the first iteration of the model! To improve the prosody and naturalness of the speech further, we're scaling up the amount of training data by a factor of five times.",
        "A male speaker with a low-pitched voice delivering his words at a fast pace in a small, confined space with a very clear audio and an animated tone.",
        3.0,
    ],
    [
        "'This is the best time of my life, Bartley,' she said happily.",
        "A female speaker with a slightly low-pitched, quite monotone voice delivers her words at a slightly faster-than-average pace in a confined space with very clear audio.",
        3.0,
    ],
    [
        "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.",
        "A male speaker with a slightly high-pitched voice delivering his words at a slightly slow pace in a small, confined space with a touch of background noise and a quite monotone tone.",
        3.0,
    ],
    [
        "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.",
        "A male speaker with a low-pitched voice delivers his words at a fast pace and an animated tone, in a very spacious environment, accompanied by noticeable background noise.",
        3.0,
    ],
]

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate


def generate_base(text, description, play_steps_in_s=2.0):
    play_steps = int(frame_rate * play_steps_in_s)

    streamer = ParlerTTSStreamer(model, device=device, play_steps=play_steps)

    inputs = tokenizer(description, return_tensors="pt", padding="max_length", max_length=TOKENIZER_MAX_LENGTH).to(
        device
    )
    prompt = tokenizer(text, return_tensors="pt", padding="max_length", max_length=TOKENIZER_MAX_LENGTH).to(device)

    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )

    set_seed(SEED)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    prev = time.time()
    for new_audio in streamer:
        print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds. Took: {time.time() - prev}")
        prev = time.time()

        yield (sampling_rate, new_audio)


css = """
        #share-btn-container {
            display: flex;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            background-color: #000000;
            justify-content: center;
            align-items: center;
            border-radius: 9999px !important; 
            width: 13rem;
            margin-top: 10px;
            margin-left: auto;
            flex: unset !important;
        }
        #share-btn {
            all: initial;
            color: #ffffff;
            font-weight: 600;
            cursor: pointer;
            font-family: 'IBM Plex Sans', sans-serif;
            margin-left: 0.5rem !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            right:0;
        }
        #share-btn * {
            all: unset !important;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
"""
with gr.Blocks(css=css) as block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                  Parler-TTS 🗣️
                </h1>
              </div>
            </div>
        """
    )
    gr.HTML(
        f"""
        <p><a href="https://github.com/huggingface/parler-tts"> Parler-TTS</a> is a training and inference library for
        high-fidelity text-to-speech (TTS) models. Two models are demonstrated here, <a href="https://huggingface.co/parler-tts/parler_tts_mini_v0.1"> Parler-TTS Mini v0.1</a>, 
        is the first iteration model trained using 10k hours of narrated audiobooks, and <a href="https://huggingface.co/ylacombe/parler-tts-mini-jenny-30H"> Parler-TTS Jenny</a>,
        a model fine-tuned on the <a href="https://huggingface.co/datasets/reach-vb/jenny_tts_dataset"> Jenny dataset</a>.
        Both models generates high-quality speech with features that can be controlled using a simple text prompt (e.g. gender, background noise, speaking rate, pitch and reverberation).</p>
        <p>Tips for ensuring good generation:
        <ul>
            <li>Include the term <b>"very clear audio"</b> to generate the highest quality audio, and "very noisy audio" for high levels of background noise</li>
            <li>When using the fine-tuned model, include the term <b>"Jenny"</b> to pick out her voice</li>
            <li>Punctuation can be used to control the prosody of the generations, e.g. use commas to add small breaks in speech</li>
            <li>The remaining speech features (gender, speaking rate, pitch and reverberation) can be controlled directly through the prompt</li>
        </ul>
        </p>
        """
    )
    with gr.Tab("Base"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input Text", lines=2, value=default_text, elem_id="input_text")
                description = gr.Textbox(label="Description", lines=2, value="", elem_id="input_description")
                play_seconds = gr.Slider(
                    0.1,
                    3,
                    value=0.2,
                    step=0.1,
                    label="Streaming interval in seconds",
                    info="Lower = shorter chunks, lower latency, more codec steps",
                )
                run_button = gr.Button("Generate Audio", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(
                    label="Parler-TTS generation", format="wav", elem_id="audio_out", streaming=True, autoplay=True
                )

        inputs = [input_text, description, play_seconds]
        outputs = [audio_out]
        gr.Examples(examples=examples, fn=generate_base, inputs=inputs, outputs=outputs, cache_examples=False)
        run_button.click(fn=generate_base, inputs=inputs, outputs=outputs, queue=True)

    gr.HTML(
        """
        <p>To improve the prosody and naturalness of the speech further, we're scaling up the amount of training data to 50k hours of speech.
        The v1 release of the model will be trained on this data, as well as inference optimisations, such as flash attention
        and torch compile, that will improve the latency by 2-4x. If you want to find out more about how this model was trained and even fine-tune it yourself, check-out the 
        <a href="https://github.com/huggingface/parler-tts"> Parler-TTS</a> repository on GitHub. The Parler-TTS codebase and its 
        associated checkpoints are licensed under <a href='https://github.com/huggingface/parler-tts?tab=Apache-2.0-1-ov-file#readme'> Apache 2.0</a>.</p>
        """
    )

block.queue()
block.launch(server_name="0.0.0.0")
