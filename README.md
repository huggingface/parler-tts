# Parler-TTS

Parler-TTS is a lightweight text-to-speech (TTS) model that can generate high-quality, natural sounding speech in the style of a given speaker (gender, pitch, speaking style, etc). It is a reproduction of work from the paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com) by Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively.

Contrarily to other TTS models, Parler-TTS is a **fully open-source** release. All of the datasets, pre-processing, training code and weights are released publicly under permissive license, enabling the community to build on our work and develop their own powerful TTS models.

This repository contains the inference and training code for Parler-TTS. It is designed to accompany the [Data-Speech](https://github.com/huggingface/dataspeech) repository for dataset annotation.

> [!IMPORTANT]
> We're proud to release [Parler-TTS Mini v0.1](https://huggingface.co/parler-tts/parler_tts_mini_v0.1), our first 600M parameter model, trained on 10.5K hours of audio data.
> In the coming weeks, we'll be working on scaling up to 50k hours of data, in preparation for the v1 model.

## 📖 Quick Index
* [Installation](#installation)
* [Usage](#usage)
* [Training](#training)
* [Demo](https://huggingface.co/spaces/parler-tts/parler_tts_mini)
* [Model weights and datasets](https://huggingface.co/parler-tts)

## Installation

Parler-TTS has light-weight dependencies and can be installed in one line:

```sh
pip install git+https://github.com/huggingface/parler-tts.git
```

Apple Silicon users will need to run a follow-up command to make use the nightly PyTorch (2.4) build for bfloat16 support:

```sh
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Usage

> [!TIP]
> You can directly try it out in an interactive demo [here](https://huggingface.co/spaces/parler-tts/parler_tts_mini)!

Using Parler-TTS is as simple as "bonjour". Simply use the following inference snippet.

```py
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
if torch.backends.mps.is_available():
    device = "mps"
if torch.xpu.is_available():
    device = "xpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device, dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

prompt = "Hey, how are you doing today?"
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```

https://github.com/huggingface/parler-tts/assets/52246514/251e2488-fe6e-42c1-81cd-814c5b7795b0

## Training
<a target="_blank" href="https://colab.research.google.com/github/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_on_a_single_speaker_dataset.ipynb"> 
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> 
</a>

The [training folder](/training/) contains all the information to train or fine-tune your own Parler-TTS model. It consists of:
- [1. An introduction to the Parler-TTS architecture](/training/README.md#1-architecture)
- [2. The first steps to get started](/training/README.md#2-getting-started)
- [3. A training guide](/training/README.md#3-training)

> [!IMPORTANT]
> **TL;DR:** After having followed the [installation steps](/training/README.md#requirements), you can reproduce the Parler-TTS Mini v0.1 training recipe with the following command line:

```sh
accelerate launch ./training/run_parler_tts_training.py ./helpers/training_configs/starting_point_0.01.json
```

## Acknowledgements

This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:
- Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively, for publishing such a promising and clear research paper: [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912).
- the many libraries used, namely [🤗 datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [🤗 accelerate](https://huggingface.co/docs/accelerate/en/index), [jiwer](https://github.com/jitsi/jiwer), [wandb](https://wandb.ai/), and [🤗 transformers](https://huggingface.co/docs/transformers/index).
- Descript for the [DAC codec model](https://github.com/descriptinc/descript-audio-codec)
- Hugging Face 🤗 for providing compute resources and time to explore!


## Citation

If you found this repository useful, please consider citing this work and also the original Stability AI paper:

```
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

```
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Contribution

Contributions are welcome, as the project offers many possibilities for improvement and exploration.

Namely, we're looking at ways to improve both quality and speed:
- Datasets:
    - Train on more data
    - Add more features such as accents
- Training:
    - Add PEFT compatibility to do Lora fine-tuning.
    - Add possibility to train without description column.
    - Add notebook training.
    - Explore multilingual training.
    - Explore mono-speaker finetuning.
    - Explore more architectures.
- Optimization:
    - Compilation and static cache
    - Support to FA2 and SDPA
- Evaluation:
    - Add more evaluation metrics

