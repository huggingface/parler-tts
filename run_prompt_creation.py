import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_name_or_path: str = field(
        metadata={"help": "The name of the model to use (via the transformers library) for the prompt annotation."},
    )
    per_device_eval_batch_size: int = field(
        metadata={"help": "The per-device batch size to use for inference."},
    )
    model_variant: str = field(
        default=None,
        metadata={"help": "If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. "},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and the computations run. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={"help": "Which attn type to use: ['eager', 'sdpa', 'flash_attention_2']"},
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 8-bit precision for inference."}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 4-bit precision for inference."}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "Use fast tokenizer for encoding/decoding input ids"}
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
    do_sample: Optional[bool] = field(default=True, metadata={"help": "Whether to use sampling mode for generation"})
    temperature: Optional[float] = field(default=0.6, metadata={"help": "Temperature for sampling-based generation"})
    max_new_tokens: Optional[int] = field(
        default=256, metadata={"help": "Maximum number of new tokens during generation"}
    )
    compile_generate: Optional[bool] = field(
        default=False, metadata={"help": "Whether to compile the forward pass (not sampling) in generate."}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        metadata={
            "help": "Where to save the processed dataset to disk. If unspecified, uses a 'pretty' version of the "
            "original dataset name. E.g. 'facebook/voxpopuli' will be saved under 'voxpopuli'."
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples for generation - use for debugging purposes."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of processes to use for the dataloader."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    hub_dataset_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository namespace if pushing to the Hugging Face Hub."},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory each time the script is run."},
    )

    def __post_init__(self):
        if self.push_to_hub and self.hub_dataset_id is None:
            raise ValueError("You must specify the `hub_dataset_id` when setting `--push_to_hub=True`")


def get_quantization_config(model_args: ModelArguments) -> Union[BitsAndBytesConfig, None]:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Union[Dict[str, int], None]:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch.
    """

    tokenizer: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_ids = {"input_ids": [feature["input_ids"] for feature in features]}
        batch = self.tokenizer.pad(input_ids, return_tensors="pt", padding="longest", return_attention_mask=True)
        return batch


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    accelerator = Accelerator()

    if data_args.overwrite_output_dir and os.path.exists(data_args.output_dir) and os.path.isdir(data_args.output_dir):
        logger.info("Cleaning output dir from previous run...")
        shutil.rmtree(data_args.output_dir)

    # 3. Load annotated dataset
    logger.info("*** Load annotated dataset ***")
    if data_args.dataset_split_name is not None:
        raw_datasets = DatasetDict()
        data_splits = data_args.dataset_split_name.split("+")
        # load on a split-wise basis
        for split in data_splits:
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )
    else:
        # load all splits for annotation
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
        )

    raw_datasets_features = set(raw_datasets[next(iter(raw_datasets))].features.keys())

    if data_args.max_eval_samples is not None:
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    # TODO(SG): add accent
    EXPECTED_COLUMNS = {"gender", "pitch", "noise", "reverberation", "speech_monotony", "speaking_rate"}
    if not EXPECTED_COLUMNS.issubset(raw_datasets_features):
        missing_columns = EXPECTED_COLUMNS - raw_datasets_features
        raise ValueError(
            f"Missing columns {missing_columns} from the dataset features. Got dataset features {raw_datasets_features}"
        )

    # 4. Load pre-trained model
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        variant=model_args.model_variant,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        token=model_args.token,
    ).eval()

    if model_args.compile_generate:
        if not callable(getattr(model, "_setup_cache", None)):
            raise ValueError(
                f"Static k/v cache is not compatible with the model {model.__class__.__name__}. Set `--compile_generate=False"
                "for dynamic k/v cache"
            )
        model.generation_config.cache_implementation = "static"
        model._forward = model.forward
        compiled_forward = torch.compile(model.forward)

        def compiled(func, input_ids, **kwargs):
            return func(input_ids, **kwargs)

        def call(input_ids, **kwargs):
            if input_ids.shape[-1] == 1:
                return compiled(compiled_forward, input_ids, **kwargs)

            return model._forward(input_ids, **kwargs)

        model.forward = call

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = model.generation_config.eos_token_id


    PROMPT = """ We have seven keywords that describe different attributes of an audio sample spoken by a given speaker: the speaker's gender, the speaker's accent, the amount of reverberation in the sample (high or low reverberation), the amount of noise in the sample (how clear or noisy), how monotone or animated the sample is, the speaker's pitch (high or low voice), the speaker's speed (how fast or slow the speaker is speaking).
    Given these keywords, form a coherent sentence that summarises the seven attributes in a meaningful way. You can change the order of the keywords in the sentence and use common synonyms for these words, provided that the sentence summarises the attributes clearly. Keep the sentence simple - don't introduce additional information other than the keywords provided. Only return the generated sentence, not any other assistant remarks.
    For example, given the following descriptors: 'female', 'Hungarian', 'slightly roomy sounding', 'fairly noisy', 'quite monotone', 'fairly low pitch', 'very slowly', a valid sentence would be: 'a woman with a deep voice speaking slowly and somewhat monotonously with a Hungarian accent in an echoey room with background noise'. Note how the seven attributes have been combined together in a simple sentence, with the ordering changed but no additional information added.
    For the descriptors: {gender}, {accent}, {reverberation}, {noise}, {speech_monotony}, {pitch}, {speaking_rate}, the corresponding sentence is:"""

    SUBSET_PROMPT = """ We have six keywords that describe different attributes of an audio sample spoken by a given speaker: the speaker's gender, the amount of reverberation in the sample (high or low reverberation), the amount of noise in the sample (how clear or noisy), how monotone or animated the sample is, the speaker's pitch (high or low voice), the speaker's speed (how fast or slow the speaker is speaking).
    Given these keywords, form a coherent sentence that summarises the six attributes in a meaningful way. You can change the order of the keywords in the sentence and use common synonyms for these words, provided that the sentence summarises the attributes clearly. Keep the sentence simple - don't introduce additional information other than the keywords provided. Only return the generated sentence, not any other assistant remarks.
    For example, given the following descriptors: 'female', 'slightly roomy sounding', 'fairly noisy', 'quite monotone', 'fairly low pitch', 'very slowly', a valid sentence would be: 'a woman with a deep voice speaking slowly and somewhat monotonously in an echoey room with background noise'. Note how the six attributes have been combined together in a simple sentence, with the ordering changed but no additional information added.
    For the descriptors: {gender}, {reverberation}, {noise}, {speech_monotony}, {pitch}, {speaking_rate}, the corresponding sentence is:"""

    def prepare_dataset(sample):
        sample_prompt = SUBSET_PROMPT
        for key in EXPECTED_COLUMNS:
            sample_prompt = sample_prompt.replace(f"[{key}]", sample[key])
        sample_prompt = [{"role": "user", "content": sample_prompt}]
        token_ids = tokenizer.apply_chat_template(sample_prompt)
        sample["input_ids"] = token_ids
        return sample

    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset, num_proc=data_args.preprocessing_num_workers, desc="Preparing prompts"
        )

    # Prepare everything with our `accelerator`
    model = accelerator.prepare(model)
    data_collator = DataCollatorWithPadding(tokenizer)

    def generate_step(batch):
        output_ids = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=model_args.do_sample,
            temperature=model_args.temperature,
            max_new_tokens=model_args.max_new_tokens,
        )
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    for split in vectorized_datasets:
        data_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=model_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=data_args.dataloader_num_workers,
            pin_memory=True,
        )
        data_loader = accelerator.prepare(data_loader)

        all_generated_ids = []
        for batch in tqdm(data_loader, disable=not accelerator.is_local_main_process):
            generated_ids = generate_step(batch)
            all_generated_ids.extend(generated_ids.cpu())

        def postprocess_dataset(sample, idx):
            prompt_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            generated_text = tokenizer.decode(all_generated_ids[idx], skip_special_tokens=True)
            sample["text_description"] = generated_text[len(prompt_text) :]
            return sample

        if accelerator.is_main_process:
            vectorized_datasets[split] = vectorized_datasets[split].map(
                postprocess_dataset,
                num_proc=data_args.preprocessing_num_workers,
                desc="Postprocessing dataset",
                remove_columns=["input_ids"],
                with_indices=True,
            )

    accelerator.end_training()

    if accelerator.is_main_process:
        vectorized_datasets.save_to_disk(data_args.output_dir)
        if data_args.push_to_hub:
            vectorized_datasets.push_to_hub(data_args.hub_dataset_id)


if __name__ == "__main__":
    main()
