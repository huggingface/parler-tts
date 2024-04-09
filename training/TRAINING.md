# Training Parler-TTS

This sub-folder contains all the information to train or finetune you own Parler-TTS model.

At the moment, Parler-TTS architecture is a carbon copy of [Musicgen architecture](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen#model-structure) and can be decomposed into three distinct stages:
>1. Text encoder: maps the text inputs to a sequence of hidden-state representations. The pre-trained MusicGen models use a frozen text encoder from either T5 or Flan-T5
>2. Parler-TTS decoder: a language model (LM) that auto-regressively generates audio tokens (or codes) conditional on the encoder hidden-state representations
>3. Audio encoder: used to recover the audio waveform from the audio tokens predicted by the decoder

Parler-TTS however introduces some small tweaks:
- The text **description** is passed through the text encoder and used in the cross-attention layers of the decoder.
- The text **prompt** is simply passed through an embedding layer and concatenated to the decoder input hidden states.
- The audio encoder used is [**DAC**](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5) instead of [Encodec](https://github.com/facebookresearch/encodec), as it exhibits better quality.

## Getting started

Before getting started, you need to follow a few steps:
1. Install the requirements.
2. Find or initialize the model you'll train on. 
3. Find and/or annotate the dataset you'll train your model on.

### 1. Requirements

The Parler-TTS code is written in [PyTorch](https://pytorch.org) and [Accelerate](https://huggingface.co/docs/accelerate/index). It uses some additional requirements, like [wandb](https://wandb.ai/), especially for logging and evaluation.

To install the package for training, you need to clone the repository from source...

```bash
git clone https://github.com/huggingface/parler-tts.git
cd parler-tts
```

... And then to install requirements.

```bash
pip install -e .[train]
```

Optionnally, you can create a wandb account and login to it by following [this guide](https://docs.wandb.ai/quickstart). [`wandb`](https://docs.wandb.ai/) allows for better tracking of the experiments metrics and losses.

You also have the option to configure Accelerate by running the following command. Note that you should set the number of GPUs you wish to use for distillation, and also the data type (dtype) to your preferred dtype for training/inference (e.g. `bfloat16` on A100 GPUs, `float16` on V100 GPUs, etc.):

```bash
accelerate config
```

Lastly, you can link you Hugging Face account so that you can push model repositories on the Hub. This will allow you to save your trained models on the Hub so that you can share them with the community. Run the command:

```bash
git config --global credential.helper store
huggingface-cli login
```
And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have one already. You should make sure that this token has "write" privileges.

### 2. Initalize a model from scratch or use a pre-trained one.

### 3. Create or find datasets

## Training


## Discussions and tips



ATTENTION: don't forget to add group_by_length in configs.


# Init model
python helpers/model_init_scripts/init_dummy_model.py /raid/yoach/artefacts/dummy_model/ "google-t5/t5-small" "ylacombe/dac_44khZ_8kbps"

text_model = "google-t5/t5-small"
encodec_version = "ylacombe/dac_44khZ_8kbps"
text_model = "google-t5/t5-small"
encodec_version = "facebook/encodec_24khz"
text_model = "google/flan-t5-base"
encodec_version = "ylacombe/dac_44khZ_8kbps"
