import gradio as gr
import torch
from transformers import AutoFeatureExtractor, AutoTokenizer, set_seed

from parler_tts import ParlerTTSForConditionalGeneration


device = "cuda:0" if torch.cuda.is_available() else "cpu"

repo_id = "parler-tts/parler_tts_mini_v0.1"

model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)


SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 41

default_text = "Please surprise me and speak in whatever voice you enjoy."

title = "# Parler-TTS </div>"

examples = [
    [
        "'This is the best time of my life, Bartley,' she said happily.",
        "A female speaker with a slightly low-pitched, quite monotone voice delivers her words at a slightly faster-than-average pace in a confined space with very clear audio.",
    ],
    [
        "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.	",
        "A male speaker with a slightly high-pitched voice delivering his words at a slightly slow pace in a small, confined space with a touch of background noise and a quite monotone tone.",
    ],
    [
        "montrose also after having experienced still more variety of good and bad fortune threw down his arms and retired out of the kingdom",
        "A male speaker with a low-pitched voice delivering his words at a fast pace in a small, confined space with a lot of background noise and an animated tone.",
    ],
]


def gen_tts(text, description):
    inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)

    set_seed(SEED)
    generation = model.generate(
        input_ids=inputs.input_ids, prompt_input_ids=prompt.input_ids, do_sample=True, temperature=1.0
    )
    audio_arr = generation.cpu().numpy().squeeze()

    return (SAMPLE_RATE, audio_arr)


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
    gr.Markdown(title)
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", lines=2, value=default_text, elem_id="input_text")
            description = gr.Textbox(label="Description", lines=2, value="", elem_id="input_description")
            run_button = gr.Button("Generate Audio", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Parler-TTS generation", type="numpy", elem_id="audio_out")

    inputs = [input_text, description]
    outputs = [audio_out]
    gr.Examples(examples=examples, fn=gen_tts, inputs=inputs, outputs=outputs, cache_examples=True)
    run_button.click(fn=gen_tts, inputs=inputs, outputs=outputs, queue=True)

block.queue()
block.launch(share=True)
