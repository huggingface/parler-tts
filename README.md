# Stable Speech

Work in-progress reproduction of the text-to-speech (TTS) model from the paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com)
by Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively.

Reproducing the TTS model requires the following 5 steps to be completed in order:
1. Train the Accent Classifier
2. Annotate the Training Set
3. Aggregate Statistics
4. Create Descriptions
5. Train the Model

## Step 1: Train the Accent Classifier

The script [`run_audio_classification.py`](run_audio_classification.py) can be used to train an audio encoder model from 
the [Transformers library](https://github.com/huggingface/transformers) (e.g. Wav2Vec2, MMS, or Whisper) for the accent
classification task.

Starting with a pre-trained audio encoder model, a simple linear classifier is appended to the last hidden-layer to map the 
audio embeddings to class label predictions. The audio encoder can either be frozen (`--freeze_base_model`) or trained. 
The linear classifier is randomly initialised, and is thus always trained.

The script can be used to train on a single accent dataset, or a combination of datasets, which should be specified by
separating dataset names, configs and splits by the `+` character in the launch command (see below for an example).

In the proceeding example, we follow Stability's approach by taking audio embeddings from a frozen [MMS-LID](https://huggingface.co/facebook/mms-lid-126) 
model, and training the linear classifier on a combination of three open-source datasets:
1. The English Accented (`en_accented`) subset of [Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)
2. The train split of [VCTK](https://huggingface.co/datasets/vctk) 
3. The dev split of [EdAcc](https://huggingface.co/datasets/sanchit-gandhi/edacc)

The model is subsequently evaluated on the test split of [EdAcc](https://huggingface.co/datasets/sanchit-gandhi/edacc)
to give the final classification accuracy.

```bash
#!/usr/bin/env bash

python run_audio_classification.py \
    --model_name_or_path "facebook/mms-lid-126" \
    --train_dataset_name "vctk+facebook/voxpopuli+sanchit-gandhi/edacc" \
    --train_dataset_config_name "main+en_accented+default" \
    --train_split_name "train+test+validation" \
    --train_label_column_name "accent+accent+accent" \
    --eval_dataset_name "sanchit-gandhi/edacc" \
    --eval_dataset_config_name "default" \
    --eval_split_name "test" \
    --eval_label_column_name "accent" \
    --output_dir "./" \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --fp16 \
    --learning_rate 1e-4 \
    --max_length_seconds 20 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "accuracy" \
    --save_total_limit 3 \
    --freeze_base_model \
    --push_to_hub \
    --trust_remote_code
```

Tips:
1. **Number of labels:** normalisation should be applied to the target class labels to group linguistically similar accents together (e.g. "Southern Irish" and "Irish" should both be "Irish"). This helps _balance_ the dataset by removing labels with very few examples. You can modify the function `preprocess_labels` to implement any custom normalisation strategy.

## Step 2: Annotate the Training Set

Annotate the training dataset with information on: SNR, C50, pitch and speaking rate. 

## Step 3: Aggregate Statistics

Aggregate statistics from Step 2. Convert continuous values to discrete labels.

## Step 4: Create Descriptions

Convert sequence of discrete labels to text description (using an LLM). 

## Step 5: Train the Model

Train MusicGen-style model on the TTS task.

