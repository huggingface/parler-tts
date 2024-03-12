#!/usr/bin/env bash

python run_dataset_concatenation.py \
    --dataset_name "stable-speech/common_voice_15_0_accented" \
    --dataset_config_name "en" \
    --dataset_split_name "train" \
    --label_column_name "accent" \
    --text_column_name "sentence" \
    --speaker_column_name "client_id" \
    --batch_size 250 \
    --preprocessing_num_workers 4 \
    --output_dir "./concatenated-dataset-cv"

python run_dataset_concatenation.py \
    --dataset_name "stable-speech/common_voice_15_0_accented" \
    --dataset_config_name "en" \
    --dataset_split_name "test" \
    --label_column_name "accent" \
    --text_column_name "sentence" \
    --speaker_column_name "client_id" \
    --batch_size 250 \
    --preprocessing_num_workers 4 \
    --output_dir "./concatenated-dataset-cv-test"
