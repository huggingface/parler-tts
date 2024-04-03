#!/usr/bin/env bash

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "ylacombe/libritts_r_tags_tagged_10k" \
  --dataset_config_name "clean" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./libritts_r_tags_tagged_10k_generated" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "stable-speech/libritts_r_tags_tagged_10k_generated"

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "ylacombe/libritts_r_tags_tagged_10k" \
  --dataset_config_name "other" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./libritts_r_tags_tagged_10k_generated" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "stable-speech/libritts_r_tags_tagged_10k_generated"

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "ylacombe/mls-eng-10k-tags_tagged_10k" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./mls-eng-10k-tags_tagged_10k_generated" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "stable-speech/mls-eng-10k-tags_tagged_10k_generated"
