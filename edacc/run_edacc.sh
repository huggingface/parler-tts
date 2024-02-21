#!/usr/bin/env bash

python prepare_edacc.py \
    --dataset_dir "/fsx/sanchit/edacc/edacc_v1.0" \
    --output_dir "/fsx/sanchit/edacc_processed" \
    --hub_dataset_id "sanchit-gandhi/edacc-normalized" \
    --push_to_hub
