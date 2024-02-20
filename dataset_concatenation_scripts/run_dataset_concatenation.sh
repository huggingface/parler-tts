#!/usr/bin/env bash

python run_dataset_concatenation.py \
    --dataset_name "sanchit-gandhi/vctk+facebook/voxpopuli+sanchit-gandhi/edacc" \
    --dataset_config_name "default+en_accented+default" \
    --dataset_split_name "train+test+validation" \
    --label_column_name "accent+accent+accent" \
    --text_column_name "text+normalized_text+text" \
    --speaker_column_name "speaker_id+speaker_id+speaker" \
    --batch_size 250 \
    --output_dir "./concatenated-dataset"
