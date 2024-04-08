ATTENTION: don't forget to add group_by_length in configs.

# Parler-TTS

[[Paper we reproduce]](https://arxiv.org/abs/2402.01912)
[[Models]](https://huggingface.co/parler-tts)
[[Training Code]](training)
[[Interactive Demo]](TODO - linked to spaces)

> [!IMPORTANT]
> We're proud to release Parler-TTS v0.1, our first 300M-parameters Parler-TTS model, trained on 10.5K hours of audio data.

Parler-TTS is a reproduction of the text-to-speech (TTS) model from the paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com)
by Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively. 

Contrarily to standard TTS models, Parler-TTS allows you to directly describe the speaker characteristics with a simple text description where you can modulate gender, pitch, speaking style, accent, etc.

## Inference

> [!TIP]
> You can directly try it out in an interactive demo [here](TODO: add link to spaces)!

Using Parler-TTS is as simple as "bonjour". Simply use the following inference snippet.

```py
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor
import soundfile as sf

# TODO: change repo id

model = ParlerTTSForConditionalGeneration.from_pretrained("ylacombe/parler_tts_300M_v0.09")
tokenizer = AutoTokenizer.from_pretrained("ylacombe/parler_tts_300M_v0.09")

prompt = "Hey, how are you doing today?"
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."

input_ids = tokenizer(description, return_tensors="pt").input_ids
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```



## Installation steps

Parler-TTS has light-weight dependencies and can be installed in one line:
```sh
pip install parler-tts
```

## Gradio demo

You can host your own Parler-TTS demo. First, install [`gradio`](https://www.gradio.app/) with:

```sh
pip install gradio
```

Then, run:

```python
python helpers/gradio_demo/app.py
```


## Acknowledgements

This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:
- Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively, for publishing such a promising and clear research paper: [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912).
- and the many libraries used, namely [datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [accelerate](https://huggingface.co/docs/accelerate/en/index), [jiwer](https://github.com/jitsi/jiwer), [wandb](https://wandb.ai/), and [transformers](https://huggingface.co/docs/transformers/index).

## Citation
```
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ylacombe/dataspeech}}
}
```
