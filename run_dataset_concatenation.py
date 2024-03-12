import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import Audio, concatenate_datasets, load_dataset
from huggingface_hub import get_full_repo_name
from transformers import HfArgumentParser, WhisperTokenizerFast


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    label_column_name: str = field(
        default="labels",
        metadata={"help": "The name of the dataset column containing the labels in the dataset. Defaults to 'label'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={
            "help": "The name of the dataset column containing the text transcriptions in the dataset. Defaults to 'text'"
        },
    )
    speaker_column_name: str = field(
        default="speaker_id",
        metadata={
            "help": "The name of the dataset column containing the speaker ids in the dataset. Defaults to 'speaker_id'"
        },
    )
    dataset_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    preprocessing_num_workers: int = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    batch_size: int = field(
        default=500,
        metadata={"help": "Number of examples per batch provided to the preprocessing function."},
    )
    download_only: bool = field(
        default=False,
        metadata={"help": "Whether to only do data download and skip pre-processing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    sampling_rate: int = field(
        default=16_000,
        metadata={
            "help": "Sampling rate at which to resample the audio data. Should be set to the same sampling rate as the target model."
        },
    )
    max_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples in the dataset to this value if set."
        },
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": "Where to save the processed dataset to disk. If unspecified, uses a 'pretty' version of the "
            "original dataset name. E.g. 'facebook/voxpopuli' will be saved under 'voxpopuli'."
        },
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    seed: int = field(
        default=0,
        metadata={"help": "RNG seed for reproducibility. Used during the final shuffling of the combined dataset."},
    )


def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    splits=None,
    label_column_names=None,
    text_column_names=None,
    speaker_column_names=None,
    dataset_samples=None,
    default_split="train",
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        label_column_names = label_column_names.split("+") if label_column_names is not None else None
        text_column_names = text_column_names.split("+") if text_column_names is not None else None
        speaker_column_names = speaker_column_names.split("+") if speaker_column_names is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if label_column_names is not None and len(label_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one label column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(label_column_names)} label column names."
        )
    if text_column_names is not None and len(text_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one text column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(text_column_names)} text column names."
        )
    if speaker_column_names is not None and len(speaker_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one text column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(speaker_column_names)} speaker column names."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    label_column_names = (
        label_column_names if label_column_names is not None else ["labels" for _ in range(len(dataset_names))]
    )
    text_column_names = (
        text_column_names if text_column_names is not None else ["text" for _ in range(len(dataset_names))]
    )
    speaker_column_names = (
        speaker_column_names if speaker_column_names is not None else ["speaker_id" for _ in range(len(dataset_names))]
    )
    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "label_column_name": label_column_names[i],
                "text_column_name": text_column_names[i],
                "speaker_column_name": speaker_column_names[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser(DataTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]

    dataset_names_dict = convert_dataset_str_to_list(
        data_args.dataset_name,
        data_args.dataset_config_name,
        splits=data_args.dataset_split_name,
        label_column_names=data_args.label_column_name,
        text_column_names=data_args.text_column_name,
        speaker_column_names=data_args.speaker_column_name,
    )

    # load whisper tokenizer for normalisation
    sampling_rate = data_args.sampling_rate
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny.en")
    max_input_length = int(data_args.max_duration_in_seconds * sampling_rate)
    batch_size = data_args.batch_size
    preprocessing_num_workers = data_args.preprocessing_num_workers
    all_vectorized_datasets = []

    for dataset_dict in dataset_names_dict:
        print(10 * "=", dataset_dict["name"], 10 * "=")
        raw_datasets = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            cache_dir=data_args.dataset_cache_dir,
            num_proc=data_args.preprocessing_num_workers,
        )

        if data_args.download_only:
            continue

        features = raw_datasets.column_names
        if dataset_dict["label_column_name"] not in features:
            raise ValueError(
                f"--label_column_name {dataset_dict['label_column_name']} not found in dataset '{dataset_dict['name']}'. "
                "Make sure to set `--label_column_name` to the correct text column - one of "
                f"{', '.join(features)}."
            )
        elif dataset_dict["label_column_name"] != "labels":
            raw_datasets = raw_datasets.rename_column(dataset_dict["label_column_name"], "labels")

        if dataset_dict["text_column_name"] not in features:
            raise ValueError(
                f"--text_column_name {dataset_dict['text_column_name']} not found in dataset '{dataset_dict['name']}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(features)}."
            )
        elif dataset_dict["text_column_name"] != "text":
            raw_datasets = raw_datasets.rename_column(dataset_dict["text_column_name"], "text")

        if dataset_dict["speaker_column_name"] not in features:
            raise ValueError(
                f"--speaker_column_name {dataset_dict['speaker_column_name']} not found in dataset '{dataset_dict['name']}'. "
                "Make sure to set `--speaker_column_name` to the correct speaker id column - one of "
                f"{', '.join(features)}."
            )
        elif dataset_dict["speaker_column_name"] != "speaker_id":
            raw_datasets = raw_datasets.rename_column(dataset_dict["speaker_column_name"], "speaker_id")

        raw_datasets = raw_datasets.remove_columns(
            set(raw_datasets.features.keys()) - {"audio", "labels", "text", "speaker_id"}
        )

        if data_args.max_samples is not None:
            raw_datasets = raw_datasets.select(range(data_args.max_samples))

        raw_datasets = raw_datasets.cast_column(data_args.audio_column_name, Audio(sampling_rate=sampling_rate))
        raw_datasets = raw_datasets.sort("speaker_id")

        def filter_transcriptions(text):
            normalized_text = tokenizer.normalize(text).strip()
            return bool(normalized_text) and text.lower() != "ignore_time_segment_in_scoring"

        raw_datasets = raw_datasets.filter(
            filter_transcriptions, input_columns=["text"], desc="Filtering non-speech transcriptions"
        )

        def prepare_dataset(batch):
            audio = [sample["array"] for sample in batch["audio"]]
            input_lengths = [len(sample) for sample in audio]

            concatenated_audio = []
            concatenated_text = []
            concatenated_speaker = []
            concatenated_labels = []
            audio_sample = audio[0]
            text_sample = batch["text"][0]
            label_sample = batch["labels"][0]

            for idx in range(1, len(audio)):
                prev_speaker = batch["speaker_id"][idx - 1]
                speaker = batch["speaker_id"][idx]

                if len(audio_sample) + input_lengths[idx] < max_input_length:
                    if speaker == prev_speaker:
                        # we have no information about whether the segments follow on sequentially
                        # so we just ensure the same speaker as we concatenate across files
                        audio_sample = np.append(audio_sample, audio[idx])
                        # extra spaces in the text transcription don't matter, since we only use it for the WER computation
                        text_sample += " " + batch["text"][idx]
                    else:
                        # segments do not follow sequentially, save the audio and start looping again
                        concatenated_audio.append(audio_sample)
                        concatenated_text.append(text_sample)
                        concatenated_labels.append(label_sample)
                        concatenated_speaker.append(speaker)
                        audio_sample = audio[idx]
                        text_sample = batch["text"][idx]
                        label_sample = batch["labels"][idx]

                else:
                    # concatenated audio exceeds max length, save the audio and start looping again
                    concatenated_audio.append(audio_sample)
                    concatenated_text.append(text_sample)
                    concatenated_labels.append(label_sample)
                    concatenated_speaker.append(speaker)
                    audio_sample = audio[idx]
                    text_sample = batch["text"][idx]
                    label_sample = batch["labels"][idx]

            batch["audio"] = [{"array": array, "sampling_rate": sampling_rate} for array in concatenated_audio]
            batch["text"] = concatenated_text
            batch["labels"] = concatenated_labels
            batch["speaker_id"] = concatenated_speaker
            return batch

        raw_datasets = raw_datasets.map(
            prepare_dataset,
            batched=True,
            batch_size=batch_size,
            num_proc=preprocessing_num_workers,
            desc="Concatenating dataset...",
        )

        pretty_name = dataset_dict["name"].split("/")[-1]

        def postprocess_ids(speaker_id, idx):
            formatted_idx = f"{pretty_name}-{speaker_id}-{idx}"
            return {"id": formatted_idx}

        raw_datasets = raw_datasets.map(
            postprocess_ids,
            input_columns=["speaker_id"],
            with_indices=True,
            desc="Setting sample idxs...",
            num_proc=preprocessing_num_workers,
        )
        print(f"Final length {pretty_name}: ", len(raw_datasets))
        # Re-format transcriptions and condition on prev as numpy arrays
        raw_datasets = raw_datasets.with_format("np")
        all_vectorized_datasets.append(raw_datasets)

    all_vectorized_datasets = concatenate_datasets(all_vectorized_datasets)
    dataset_features = all_vectorized_datasets.features.copy()
    dataset_features["audio"] = Audio(sampling_rate=sampling_rate)
    all_vectorized_datasets = all_vectorized_datasets.cast(
        dataset_features, batch_size=batch_size, writer_batch_size=batch_size, num_proc=preprocessing_num_workers
    )
    all_vectorized_datasets = all_vectorized_datasets.shuffle(seed=data_args.seed)

    all_vectorized_datasets.save_to_disk(data_args.output_dir)
    repo_name = get_full_repo_name(Path(data_args.output_dir).absolute().name)
    if data_args.push_to_hub:
        all_vectorized_datasets.push_to_hub(repo_name, config_name="train", max_shard_size="1GB")


if __name__ == "__main__":
    main()
