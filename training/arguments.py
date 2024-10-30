from dataclasses import dataclass, field
from typing import Optional, List

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained feature extractor name or path if not the same as model_name"}
    )
    description_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained description tokenizer name or path if not the same as model_name"}
    )
    prompt_tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained prompt tokenizer name or path if not the same as description_tokenizer_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    pad_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model pad token id."},
    )
    decoder_start_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model decoder start token id."},
    )
    freeze_text_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the text encoder."},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to do sampling or greedy decoding."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature if sampling."},
    )
    max_length: int = field(
        default=2580,
        metadata={"help": "Generation max length."},
    )
    bandwidth: float = field(
        default=6,
        metadata={"help": "Audio encoder bandwidth."},
    )
    asr_model_name_or_path: str = field(
        default="distil-whisper/distil-large-v2",
        metadata={
            "help": "Used to compute WER during evaluation. Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    clap_model_name_or_path: str = field(
        default="laion/larger_clap_music_and_speech",
        metadata={
            "help": "Used to compute audio similarity during evaluation. Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention implementation used. One of `eager`, `sdpa`, `flash_attention_2`"},
    )
    cross_attention_implementation_strategy: str = field(
        default=None,
        metadata={
            "help": "If not specified, the cross-attention implementation will be the same as `_attn_implementation`. If `always_eager`, it will always be the eager implementation. If `always_sdpa`, it will always be the sdpa implementation."
        },
    )
    prompt_padding_side: Optional[str] = field(
        default="left",
        metadata={
            "help": "Prompt tokenizer padding side. Defaults to `left`. If the prompt is pre-pended to the codebooks hidden states, it should be padded on the left."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load and combine "
            " librispeech and common voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol."
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": ("The name of the training data set split to use (via the datasets library). Defaults to 'train'")
        },
    )
    train_dataset_samples: str = field(
        default=None,
        metadata={
            "help": "Number of samples in the training data. Load and combine "
            "multiple datasets by separating dataset samples by a '+' symbol."
        },
    )
    train_metadata_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the metadata training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load and combine "
            " librispeech and common voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    eval_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training dataset name if unspecified."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the training dataset config name if unspecified"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    eval_metadata_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the metadata training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load and combine "
            " librispeech and common voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    target_audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the target audio data. Defaults to 'audio'"},
    )
    description_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the description text data. Defaults to 'None'."},
    )
    prompt_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the prompt text data. Defaults to 'None'."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    max_duration_in_seconds: float = field(
        default=35.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`."
                "Also, used to set maximum audio length if `pad_to_max_length=True`."
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    max_text_length: int = field(
        default=500, metadata={"help": "If set, max description lengths in number of characters."}
    )
    max_prompt_token_length: int = field(
        default=None,
        metadata={
            "help": (
                "If set, filter samples with prompts that are longer than `max_prompt_token_length` tokens."
                "Also, used to set maximum prompt token length if `pad_to_max_length=True`."
            )
        },
    )
    max_description_token_length: int = field(
        default=None,
        metadata={
            "help": (
                "If set, filter samples with descriptions that are longer than `max_description_token_length` tokens."
                "Also, used to set maximum description token length if `pad_to_max_length=True`."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True`, pad audio, prompt and description to a maximum length set with respectively "
                "`max_duration_in_seconds`, `max_prompt_token_length`, `max_description_token_length`."
            )
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training."
                " In this training script, `save_to_disk` must be set to the path in which the dataset should be saved. "
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    add_audio_samples_to_wandb: bool = field(
        default=False,
        metadata={"help": "If set and if `wandb` in args.report_to, will add generated audio samples to wandb logs."},
    )
    id_column_name: str = field(default=None, metadata={"help": "id column name."})
    wandb_project: str = field(
        default="parler-speech",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_run_name: str = field(
        default=None,
        metadata={
            "help": "If specified, the name of the run. If not specified, wandb will give a random name to this run."
        },
    )
    save_to_disk: str = field(
        default=None,
        metadata={
            "help": "If set, will save the dataset to this path if this is an empyt folder. If not empty, will load the datasets from it."
        },
    )
    temporary_save_to_disk: str = field(default=None, metadata={"help": "Temporarily save audio labels here."})
    save_codec_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Temporarily save the audio labels every `save_steps`."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=2,
        metadata={"help": ("Pad to multiple of for tokenizers.")},
    )


@dataclass
class ParlerTTSTrainingArguments(Seq2SeqTrainingArguments):
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    audio_encoder_per_device_batch_size: int = field(
        default=8,
        metadata={"help": ("Specify the batch size of the audio encoding pre-processing steps.")},
    )
    eval_dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for evaluation data loading (PyTorch only). 0 means that the data will be loaded in the main process."
            )
        },
    )
    compute_clap_similarity_metric: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to compute the clap similarity metric between the description and the generation during evalution."
            )
        },
    )
    compute_noise_level_metric: bool = field(
        default=True,
        metadata={"help": ("Whether or not to compute the squim si-sdr measure of the generations.")},
    )
    noise_level_to_compute_clean_wer: float = field(
        default=25,
        metadata={
            "help": (
                "if `compute_noise_level_metric=True`, will compute a 'clean' WER on samples with generated noise higher than `noise_level_to_compute_clean_wer`."
                "This is a proxy measure to compute WER on clean audios, provided that the model learn to generate clean audios."
            )
        },
    )
    eval_generation_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of update steps between two generation evaluation.  Will default to the same"
                "value as `eval_steps` if not set. Should be an integer and a multiple of `eval_steps`."
            )
        },
    )       
    codebook_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Weights applied to each codebook."},
    )