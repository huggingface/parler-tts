# Training Parler-TTS

<a target="_blank" href="https://github.com/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_v1_on_a_single_speaker_dataset.ipynb"> 
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> 
</a>

**TL;DR:** After having followed the [installation steps](#requirements), you can reproduce the [Parler-TTS Mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) training recipe with the following command line:

```sh
accelerate launch ./training/run_parler_tts_training.py ./helpers/training_configs/starting_point_v1.json
```

-------------

This sub-folder contains all the information to train or fine-tune your own Parler-TTS model. It consists of:
- [1. An introduction to the Parler-TTS architecture](#a-architecture)
- [2. First steps to get started](#b-getting-started)
- [3. Training guide](#c-training)

> [!IMPORTANT]
> You can also follow [this fine-tuning guide](https://github.com/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_v1_on_a_single_speaker_dataset.ipynb) on a mono-speaker dataset example.

## 1. Architecture

At the moment, Parler-TTS architecture is almost a carbon copy of the [MusicGen architecture](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/musicgen#model-structure) and can be decomposed into three distinct stages:
1. Text encoder: maps the text descriptions to a sequence of hidden-state representations. Parler-TTS uses a frozen text encoder initialised entirely from Flan-T5
2. Parler-TTS decoder: a language model (LM) that auto-regressively generates audio tokens (or codes) conditional on the encoder hidden-state representations
3. Audio codec: used to recover the audio waveform from the audio tokens predicted by the decoder. We use the [DAC model](https://github.com/descriptinc/descript-audio-codec) from Descript, although other codec models, such as [EnCodec](https://huggingface.co/facebook/encodec_48khz), can also be used.

Parler-TTS however introduces some small tweaks:
- The text **description** is passed through the text encoder and used in the cross-attention layers of the decoder.
- The text **prompt** is simply passed through an embedding layer and concatenated to the decoder input hidden states.
- The audio encoder used is [**DAC**](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5) instead of [Encodec](https://github.com/facebookresearch/encodec), as it exhibits better quality.


## 2. Getting started

To get started, you need to follow a few steps:
1. Install the requirements.
2. Find or initialize the model you'll train on. 
3. Find and/or annotate the dataset you'll train your model on.

### Requirements

The Parler-TTS code is written in [PyTorch](https://pytorch.org) and [Accelerate](https://huggingface.co/docs/accelerate/index). It uses some additional requirements, like [wandb](https://wandb.ai/), especially for logging and evaluation.

To install the package for training, you need to clone the repository from source...

```bash
git clone https://github.com/huggingface/parler-tts.git
cd parler-tts
```

... And then install the requirements:

```bash
pip install -e .[train]
```

Optionally, you can create a wandb account and login to it by following [this guide](https://docs.wandb.ai/quickstart). [`wandb`](https://docs.wandb.ai/) allows for better tracking of the experiments metrics and losses.

You also have the option to configure Accelerate by running the following command. Note that you should set the number of GPUs you wish to use for training, and also the data type (dtype) to your preferred dtype for training/inference (e.g. `bfloat16` on A100 GPUs, `float16` on V100 GPUs, etc.):

```bash
accelerate config
```

Lastly, you can link you Hugging Face account so that you can push model repositories on the Hub. This will allow you to save your trained models on the Hub so that you can share them with the community. Run the command:

```bash
git config --global credential.helper store
huggingface-cli login
```
And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have one already. You should make sure that this token has "write" privileges.

### Initialize a model from scratch or use a pre-trained one.

Depending on your compute resources and your dataset, you need to choose between fine-tuning a pre-trained model and training a new model from scratch.

In that sense, we released a 880M checkpoint trained on 45K hours of annotated data under the repository id: [`parler-tts/parler-tts-mini-v1`](https://huggingface.co/parler-tts/parler-tts-mini-v1), that you can fine-tune for your own use-case.

You can also train you own model from scratch. You can find [here](/helpers/model_init_scripts/) examples on how to initialize a model from scratch. For example, you can initialize a dummy model with:

```sh
python helpers/model_init_scripts/init_dummy_model.py ./parler-tts-untrained-dummy --text_model "google-t5/t5-small" --audio_model "parler-tts/dac_44khZ_8kbps"
```

In the rest of this guide, and to reproduce the Parler-TTS Mini v1 training recipe, we'll use a 880M parameters model that we'll initialize with:

```sh
python helpers/model_init_scripts/init_model_600M.py ./parler-tts-untrained-600M --text_model "google/flan-t5-large" --audio_model "parler-tts/dac_44khZ_8kbps"
```


### Create or find datasets

To train your own Parler-TTS, you need datasets with 3 main features:
- speech data
- text transcription of the speech data
- conditionning text description - that you can create using [Data-Speech](https://github.com/huggingface/dataspeech), a library that allows you to annotate the speaker and utterance characteristics with natural language description.

Note that we made the choice to use description of the main speech characteristics (speaker pitch, speaking rate, level of noise, etc.) but that you are free to use any handmade or generated text description that makes sense.

To train Parler-TTS Mini v1, we used:
* A [filtered version](https://huggingface.co/datasets/parler-tts/libritts_r_filtered) of [LibriTTS-R dataset](https://huggingface.co/datasets/blabble-io/libritts_r), a 1K hours high-quality speech dataset.
* The [English subset](https://huggingface.co/datasets/parler-tts/mls_eng) of [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech).

Both datasets have been annotated using the [Data-Speech](https://github.com/huggingface/dataspeech) recipe, respectively [here](https://huggingface.co/datasets/parler-tts/libritts-r-filtered-speaker-descriptions) and [here](https://huggingface.co/datasets/parler-tts/mls-eng-speaker-descriptions).


## 3. Training

The script [`run_parler_tts_training.py`](/training/run_parler_tts_training.py) is an end-to-end script that:
1. load dataset(s) and merge them to the annotation dataset(s) if necessary
2. pre-compute audio tokens
3. train Parler-TTS

To train Parler-TTS Mini v1, we roughly used:

```sh
accelerate launch ./training/run_parler_tts_training.py \
    --model_name_or_path "./parler-tts-untrained-600M/parler-tts-untrained-600M/" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "google/flan-t5-large" \
    --prompt_tokenizer_name "google/flan-t5-large" \
    --report_to "wandb" \
    --overwrite_output_dir true \
    --train_dataset_name "parler-tts/libritts_r_filtered+parler-tts/libritts_r_filtered+parler-tts/libritts_r_filtered+parler-tts/mls_eng" \
    --train_metadata_dataset_name "parler-tts/libritts-r-filtered-speaker-descriptions+parler-tts/libritts-r-filtered-speaker-descriptions+parler-tts/libritts-r-filtered-speaker-descriptions+parler-tts/mls-eng-speaker-descriptions" \
    --train_dataset_config_name "clean+clean+other+default" \
    --train_split_name "train.clean.360+train.clean.100+train.other.500+train" \
    --eval_dataset_name "parler-tts/libritts_r_filtered+parler-tts/mls_eng" \
    --eval_metadata_dataset_name "parler-tts/libritts-r-filtered-speaker-descriptions+parler-tts/mls-eng-speaker-descriptions" \
    --eval_dataset_config_name "other+default" \
    --eval_split_name "test.other+test" \
    --target_audio_column_name "audio" \
    --description_column_name "text_description" \
    --prompt_column_name "text" \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 600 \
    --add_audio_samples_to_wandb true \
    --id_column_name "id" \
    --preprocessing_num_workers 8 \
    --do_train true \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 6 \
    --gradient_checkpointing false \
    --per_device_train_batch_size 4 \
    --learning_rate 0.00095 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 20000 \
    --logging_steps 1000 \
    --freeze_text_encoder true \
    --do_eval true \
    --predict_with_generate true \
    --include_inputs_for_metrics true \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_steps 10000 \
    --per_device_eval_batch_size 4 \
    --audio_encoder_per_device_batch_size 24 \
    --dtype "bfloat16" \
    --seed 456 \
    --output_dir "./output_dir_training/" \
    --temporary_save_to_disk "./audio_code_tmp/" \
    --save_to_disk "./tmp_dataset_audio/" \
    --max_eval_samples 96 \
    --dataloader_num_workers 8 \
    --group_by_length true \
    --attn_implementation "sdpa"
```

In particular, note how multiple training datasets, metadataset, configurations and splits can be loaded by separating the dataset arguments by + symbols:
```sh
    "train_dataset_name": "parler-tts/libritts_r_filtered+parler-tts/libritts_r_filtered+parler-tts/libritts_r_filtered+parler-tts/mls_eng",
    "train_metadata_dataset_name": "parler-tts/libritts-r-filtered-speaker-descriptions+parler-tts/libritts-r-filtered-speaker-descriptions+parler-tts/libritts-r-filtered-speaker-descriptions+parler-tts/mls-eng-speaker-descriptions",
    "train_dataset_config_name": "clean+clean+other+default",
    "train_split_name": "train.clean.360+train.clean.100+train.other.500+train",
```


Additionally, you can also write a JSON config file. Here, [starting_point_v1.json](helpers/training_configs/starting_point_v1.json) contains the exact same hyper-parameters than above and can be launched like that:
```sh
accelerate launch ./training/run_parler_tts_training.py ./helpers/training_configs/starting_point_v1.json
```

Training logs will be reported to wandb, provided that you passed `--report_to "wandb"` to the arguments.

> [!TIP]
> Starting training a new model from scratch can easily be overwhelming, so here's what training looked like for v1: [logs](https://api.wandb.ai/links/ylacombe/j7g8isjn)

Scaling to multiple GPUs using [distributed data parallelism (DDP)](https://pytorch.org/tutorials/beginner/ddp_series_theory.html) is trivial: simply run `accelerate config` and select the multi-GPU option, specifying the IDs of the GPUs you wish to use. The above script can then be run using DDP with no code changes. In our case, we used 4 nodes of 8 H100 80GB to train Parler-TTS Mini for around 1.5 days.


There are a few other noteworthy arguments:
1. `train_metadata_dataset_name` and `eval_metadata_dataset_name` specify, if necessary, the names of the dataset(s) that contain(s) the conditionning text descriptions. For example, this [dataset resulting from the Data-Speech annotation process](https://huggingface.co/datasets/parler-tts/libritts-r-filtered-speaker-descriptions) is saved without the audio column, as it's costly to write and push audio data, so it needs to be concatenated back to the original LibriTTS-R dataset.
2. As noted above, the script pre-computes audio tokens as computing audio codes is costly and only needs to be done once, since we're freezing the audio encoder. `audio_encoder_per_device_batch_size` is used to precise the per devie batch size for this pre-processing step.
3. Additionnally, when scaling up the training data and iterating on the hyper-parameters or the model architecture, we might want to avoid recomputing the audio tokens at each training run. That's why we introduced two additional parameters, `save_to_disk` and `temporary_save_to_disk` that serves as temporary buffers to save intermediary datasets. Note that processed data is made of text and audio tokens which are much more memory efficient, so the additional required space is negligible.
4. `predict_with_generate` and `add_audio_samples_to_wandb` are required to store generated audios and to compute WER and CLAP similarity.
5. `freeze_text_encoder`: which allows to freeze the text encoder, to save compute resources.

And finally, two additional comments:
1. `lr_scheduler_stype`: defines the learning rate schedule, one of `constant_with_warmup` or `cosine`. When experimenting with a training set-up or training for very few epochs, using `constant_with_warmup` is typically beneficial, since the learning rate remains high over the short training run. When performing longer training runs, using a `cosine` schedule shoud give better results.
2. `dtype`: data type (dtype) in which the model computation should be performed. Note that this only controls the dtype of the computations (forward and backward pass), and not the dtype of the parameters or optimiser states.

> [!TIP]
> Fine-tuning is as easy as modifying `model_name_or_path` to a pre-trained model.
> For example: `--model_name_or_path parler-tts/parler-tts-mini-v1`.
