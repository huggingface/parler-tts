#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from random import randint
from typing import List, Optional, Union, Dict

import torch

import datasets
import evaluate
import numpy as np
import transformers
from datasets import Dataset, DatasetDict, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.trainer_pt_utils import LengthGroupedSampler

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Model
from transformers import Wav2Vec2BertForSequenceClassification, Wav2Vec2BertModel
from transformers.models.wav2vec2.modeling_wav2vec2 import _HIDDEN_STATES_START_POSITION
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0.dev0")


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000) -> np.ndarray:
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


def deterministic_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000) -> np.ndarray:
    """Take first `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    return wav[0:sample_length]

class SequenceClassificationModel(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # To bypass w2v2
        self.compute_w2v2 = True

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_features=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        hidden_states=None, # added
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True

        if self.compute_w2v2:
            outputs = self.wav2vec2(
                input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[_HIDDEN_STATES_START_POSITION][-1] # take last embedding layer

            if attention_mask is None:
                pooled_output = hidden_states.mean(dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
                hidden_states[~padding_mask] = 0.0
                pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        else:
            pooled_output = hidden_states

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

# This list first defines the accent prefixes, which we use to strip the accent from CV
# e.g. England, southern accent, slight west-country expression -> England
# TODO(YL): update this with any CV test prefixes not present in the train set
STARTS_WITH = [
    "Afrikaans",
    "American",
    "Australian",
    "Bangladeshi",
    "Canadian",
    "Chinese",
    "Dutch",
    "Eastern European",
    "European",
    "England",
    "English",
    "German",
    "Filipino",
    "India",
    "Irish",
    "Israeli",
    "Italian",
    "Japanese",
    "Kenyan",
    "Northern Irish",
    "New Zealand",
    "Nigerian",
    "Malaysian",
    "Russian",
    "Scottish",
    "Singaporean",
    "Slavic",
    "South African",
    "Southern African",
    "Swedish",
    "Swiss",
    "United States English",
    "West Indies",
    "french",
    "polish",
    "serbian",
]


# This dictionary is used to map the un-normalised accent names to normalised ones
# TODO(YL): update this with any CV test mappings not present in the train set
ACCENT_MAPPING = {
    "British": "English",
    # "Canadian": "American",  TODO(SG): decide whether to normalize these to closely related accents
    # "New zealand": "Australian",
    "Northern irish": "Irish",
    "Pakistani": "Indian",
    "Mainstream u s english": "American",
    "Southern british english": "English",
    "Indian english": "Indian",
    "Scottish english": "Scottish",
    "Don't know": "Unknown",
    "Nigerian english": "Nigerian",
    "Kenyan english": "Kenyan",
    "Ghanain english": "Ghanain",
    "Jamaican english": "Jamaican",
    "Indonesian english": "Indonesian",
    "South african english": "South african",
    "Irish english": "Irish",
    "Latin": "Latin american",
    "European": "Unknown",  # Too general
    "Eastern european": "Eastern european",  # TODO(SG): keep for now, but maybe remove later as too general
    "Bangladeshi": "Indian",
    "England": "English",
    "India": "Indian",
    "Afrikaans": "South african",
    "California": "American",
    "Nepali": "Indian",
    "New york city": "American",
    "New jerseyan": "American",
    "Northumbrian british english": "English",
    "Nottinghamshire,east midlands": "English",
    "Southern african": "South african",
    "United states english": "American",
    "West indies": "Jamaican",
    "2nd language": "Unknown",  # Too vague
    "A savage texas gentleman": "American",
    "A variety of texan english with some german influence that has undergone the cot-caught merger": "American",
    "A'lo": "Unknown",  # Unclear
    "Academic southern english,england english": "English",
    "Argentinian english": "Latin american",
    "Austrian": "German",
    "Bangladesh,india and south asia (india, pakistan, sri lanka)": "Indian",
    "Brazillian accent": "Brazilian",
    "British accent": "English",
    "Caribbean canadian": "Unknown",  # Specific combination not listed
    "Colombian accent": "Latin american",
    "Czech accent": "Czech",
    "East african khoja": "Unknown",  # Specific community
    "East indian": "Indian",
    "East london": "English",
    "England,london,academic": "English",
    "Filipino": "Unknown",  # Unique blend
    "Fluent,e sl,european": "Unknown",  # Too vague
    "Generic european": "Unknown",  # Too vague
    "Georgian english": "Unknown",  # No direct match
    "Ghanaian english accent,african regular reader": "Unknown",  # Specific category not listed
    "Haitian creole": "Unknown",  # Unique blend
    "Hispanic": "Latin american",
    "Hispanic/latino": "Latin american",
    "Hong kong english": "Chinese",
    "Hong kong english,scottish english": "Chinese",
    "Hunglish": "Hungarian",
    "I think mine accent is influenced by indian accent ,yes please. ,india and south asia (india, pakistan, sri lanka)": "Indian",
    "I was born in england and have lived in australia, canada and france.": "English",
    "International english,united states english,australian english": "American",
    "Israeli": "Unknown",  # No direct match
    "Israeli english": "Unknown",  # No direct match
    "Javanese,indonesian english,malaysian english": "Indonesian",
    "Kazakhstan english": "Unknown",  # No direct match
    "Kiwi": "New zealand",  # Could be generalised to Australian
    "Latin america,united states english": "Latin american",
    "Latin american accent": "Latin american",
    "Latin english": "Unknown",  # Too vague
    "Latino": "Latin american",
    "Latvian": "Latvian",  # Note: added new
    "Little latino,united states english,second language": "Latin american",
    "Liverpool english,lancashire english,england english": "English",
    "Liverpudlian english": "English",
    "Malaysian english": "Malaysian",  # Note: added new
    "Mexican accent": "Latin american",
    "Mid-atlantic united states english,philadelphia, pennsylvania, united states english,united states english,philadelphia style united states english": "American",
    "Mid-atlantic,england english,united states english": "American",
    "Midatlantic,england english": "American",
    "Midwestern states (michigan),united states english": "American",
    "Mild northern england english": "English",
    "Minor french accent": "French",
    "Mix of american and british ,native polish": "Polish",
    "Mix of american and british accent": "Unknown",  # Combination not clearly mapped
    "Mostly american with some british and australian inflections": "Unknown",  # Combination not clearly mapped
    "My accent is influenced by the phones of all letters within a sentence.,southern african (south africa, zimbabwe, namibia)": "South african",
    "New zealand english": "New Zealand English",
    "Nigeria english": "Nigerian",  # Note: added new
    "Non native speaker from france": "French",
    "Non-native": "Unknown",  # Too vague
    "Non-native,german accent": "German",
    "North european english": "Unknown",  # Too broad
    "Norwegian": "Norwegian",  # Note: added new
    "Ontario,canadian english": "Canadian",  # Note: added new
    "Polish english": "Polish",
    "Rhode island new england accent": "American",
    "Singaporean english": "Singaporean",  # Note: added new
    "Slavic": "Eastern european",
    "Slighty southern affected by decades in the midwest, 4 years in spain and germany, speak some german, spanish, polish. have lived in nine states.": "Unknown",  # Complex blend
    "South african": "South african",
    "South atlantic (falkland islands, saint helena)": "Unknown",  # Specific regions not listed
    "South australia": "Australian",
    "South indian": "Indian",
    "Southern drawl": "American",
    "Southern texas accent,united states english": "American",
    "Southern united states,united states english": "American",
    "Spanish bilingual": "Spanish",
    "Spanish,foreign,non-native": "Spanish",
    "Strong latvian accent": "Latvian",
    "Swedish accent": "Swedish",  # Note: added new
    "Transnational englishes blend": "Unknown",  # Too vague
    "U.k. english": "English",
    "Very slight russian accent,standard american english,boston influence": "American",
    "Welsh english": "Welsh",
    "West african": "Unknown",  # No specific West African category
    "West indian": "Unknown",  # Caribbean, but no specific match
    "Western europe": "Unknown",  # Too broad
    "With heavy cantonese accent": "Chinese",
}


def preprocess_labels(label: str) -> str:
    """Apply pre-processing formatting to the accent labels"""
    if "_" in label:
        # voxpopuli stylises the accent as a language code (e.g. en_pl for "polish") - convert to full accent
        language_code = label.split("_")[-1]
        label = LANGUAGES[language_code]
    # VCTK labels for two words are concatenated into one (NewZeleand-> New Zealand)
    label = re.sub(r"(\w)([A-Z])", r"\1 \2", label).strip()
    for prefix in STARTS_WITH:
        if label.startswith(prefix):
            label = prefix
    # convert Whisper language code (polish) to capitalised (Polish)
    label = label.capitalize()
    if label in ACCENT_MAPPING:
        label = ACCENT_MAPPING[label]
    return label

    
@dataclass
class DataCollatorFeatureExtractorWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch.
    """

    feature_extractor: AutoFeatureExtractor
    max_length_seconds: int
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        audios = [torch.tensor(deterministic_subsample(feature["audio"]["array"], max_length=self.max_length_seconds, sample_rate=self.feature_extractor.sampling_rate)).squeeze().numpy() for feature in features]
        batch = self.feature_extractor(audios, return_tensors="pt", padding="longest", return_attention_mask=True)
        
        batch["labels"] = torch.tensor([feature["labels_id"] for feature in features])
        return batch

@dataclass
class DataCollatorHiddenStatesPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch.
    """

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:        
        hidden_states = torch.stack([torch.tensor(feature["hidden_states"]) for feature in features])

        batch = {"hidden_states": hidden_states}
        
        batch["labels"] = torch.tensor([feature["labels_id"] for feature in features])
        return batch



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
        default="validation",
        metadata={
            "help": (
                "The name of the evaluation data set split to use (via the datasets"
                " library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    train_label_column_name: str = field(
        default="labels",
        metadata={
            "help": "The name of the dataset column containing the labels in the train set. Defaults to 'label'"
        },
    )
    eval_label_column_name: str = field(
        default="labels",
        metadata={"help": "The name of the dataset column containing the labels in the eval set. Defaults to 'label'"},
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
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: Optional[float] = field(
        default=20,
        metadata={"help": "Audio samples will be randomly cut to this length during training if the value is set."},
    )
    min_length_seconds: Optional[float] = field(
        default=5,
        metadata={"help": "Audio samples less than this value will be filtered during training if the value is set."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    filter_threshold: Optional[float] = field(
        default=1.0,
        metadata={"help": "Filter labels that occur less than `filter_threshold` percent in the training/eval data."},
    )
    max_samples_per_label: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, randomly limits the number of samples per label."
            )
        },   
    )
    save_to_disk: str = field(
        default=None,
        metadata={
            "help": "If set, will save the dataset to this path if this is an empyt folder. If not empty, will load the datasets from it."
        }
    )
    temporary_save_to_disk: str = field(
        default=None,
        metadata={
            "help": "Temporarily save audio labels here."
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. Only works with Wav2Vec2 models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze the feature encoder layers of the model. Only relevant for Wav2Vec2-style models."
        },
    )
    freeze_base_model: bool = field(
        default=True, metadata={"help": "Whether to freeze the base encoder of the model."}
    )
    attention_mask: bool = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
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
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    use_weighted_layer_sum: bool = field(default=False, metadata={"help": "Whether to use a weighted average of layer outputs with learned weights."})
    use_last_embedding_layer: bool = field(default=False, metadata={"help": "Whether to use the last layer hidden state. Only work with W2V model."})


def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    splits=None,
    label_column_names=None,
    dataset_samples=None,
    default_split="train",
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        label_column_names = label_column_names.split("+") if label_column_names is not None else None
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
    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "label_column_name": label_column_names[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict


def load_multiple_datasets(
    dataset_names: Union[List, str],
    dataset_config_names: Union[List, str],
    splits: Optional[Union[List, str]] = None,
    label_column_names: Optional[List] = None,
    sampling_rate: Optional[int] = 16000,
    stopping_strategy: Optional[str] = "first_exhausted",
    dataset_samples: Optional[Union[List, np.array]] = None,
    streaming: Optional[bool] = False,
    seed: Optional[int] = None,
    audio_column_name: Optional[str] = "audio",
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    dataset_names_dict = convert_dataset_str_to_list(
        dataset_names, dataset_config_names, splits, label_column_names, dataset_samples
    )

    if dataset_samples is not None:
        dataset_samples = [ds_dict["samples"] for ds_dict in dataset_names_dict]
        probabilities = np.array(dataset_samples) / np.sum(dataset_samples)
    else:
        probabilities = None

    all_datasets = []
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(dataset_names_dict, desc="Combining datasets..."):
        dataset = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            streaming=streaming,
            **kwargs,
        )
        dataset_features = dataset.features.keys()

        if audio_column_name not in dataset_features:
            raise ValueError(
                f"Audio column name '{audio_column_name}' not found in dataset"
                f" '{dataset_dict['name']}'. Make sure to set `--audio_column_name` to"
                f" the correct audio column - one of {', '.join(dataset_features)}."
            )
        # resample to specified sampling rate
        dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate))

        if dataset_dict["label_column_name"] not in dataset_features:
            raise ValueError(
                f"Label column name {dataset_dict['label_column_name']} not found in dataset"
                f" '{dataset_dict['name']}'. Make sure to set `--label_column_name` to the"
                f" correct text column - one of {', '.join(dataset_features)}."
            )

        # blanket renaming of all label columns to label
        if dataset_dict["label_column_name"] != "labels":
            dataset = dataset.rename_column(dataset_dict["label_column_name"], "labels")

        dataset_features = dataset.features.keys()
        columns_to_keep = {"audio", "labels"}
        dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        # we have a single dataset so just return it as is
        return all_datasets[0]

    if streaming:
        interleaved_dataset = interleave_datasets(
            all_datasets,
            stopping_strategy=stopping_strategy,
            probabilities=probabilities,
            seed=seed,
        )
    else:
        interleaved_dataset = concatenate_datasets(all_datasets)

    return interleaved_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if data_args.save_to_disk is not None:
        os.makedirs(data_args.save_to_disk, exist_ok=True)
    
    # assume that the dataset has been saved to `save_to_disk` if the latter is not empty
    dataset_was_precomputed = len(os.listdir(data_args.save_to_disk)) > 0
    if dataset_was_precomputed:
        raw_datasets = datasets.load_from_disk(data_args.save_to_disk)
    else:
        # Initialize our dataset and prepare it for the audio classification task.
        raw_datasets = DatasetDict()

        if training_args.do_train:
            raw_datasets["train"] = load_multiple_datasets(
                data_args.train_dataset_name,
                data_args.train_dataset_config_name,
                splits=data_args.train_split_name,
                label_column_names=data_args.train_label_column_name,
                dataset_samples=data_args.train_dataset_samples,
                seed=training_args.seed,
                cache_dir=model_args.cache_dir,
                token=True if model_args.token else None,
                trust_remote_code=model_args.trust_remote_code,
                num_proc=data_args.preprocessing_num_workers,
                # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
            )

        if training_args.do_eval:
            dataset_names_dict = convert_dataset_str_to_list(
                data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
                (
                    data_args.eval_dataset_config_name
                    if data_args.eval_dataset_config_name
                    else data_args.train_dataset_config_name
                ),
                splits=data_args.eval_split_name,
                label_column_names=data_args.eval_label_column_name,
            )
            all_eval_splits = []
            # load multiple eval sets
            for dataset_dict in dataset_names_dict:
                pretty_name = (
                    f"{dataset_dict['name'].split('/')[-1]}/{dataset_dict['split'].replace('.', '-')}"
                    if len(dataset_names_dict) > 1
                    else "eval"
                )
                all_eval_splits.append(pretty_name)
                raw_datasets[pretty_name] = load_dataset(
                    dataset_dict["name"],
                    dataset_dict["config"],
                    split=dataset_dict["split"],
                    cache_dir=model_args.cache_dir,
                    token=True if model_args.token else None,
                    trust_remote_code=model_args.trust_remote_code,
                    num_proc=data_args.preprocessing_num_workers,
                    # streaming=data_args.streaming,
                )
                features = raw_datasets[pretty_name].features.keys()
                if dataset_dict["label_column_name"] not in features:
                    raise ValueError(
                        f"--label_column_name {data_args.eval_label_column_name} not found in dataset '{data_args.dataset_name}'. "
                        "Make sure to set `--label_column_name` to the correct text column - one of "
                        f"{', '.join(raw_datasets['train'].column_names)}."
                    )
                elif dataset_dict["label_column_name"] != "labels":
                    raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                        dataset_dict["label_column_name"], "labels"
                    )
                raw_datasets[pretty_name] = raw_datasets[pretty_name].remove_columns(
                    set(raw_datasets[pretty_name].features.keys()) - {"audio", "labels"}
                )

        if not training_args.do_train and not training_args.do_eval:
            raise ValueError(
                "Cannot not train and not do evaluation. At least one of training or evaluation has to be performed."
            )

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    if not dataset_was_precomputed:
        # `datasets` takes care of automatically loading and resampling the audio,
        # so we just need to set the correct target sampling rate.
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

        with training_args.main_process_first():
            if training_args.do_train:
                if data_args.max_train_samples is not None:
                    raw_datasets["train"] = (
                        raw_datasets["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
                    )

            if training_args.do_eval:
                if data_args.max_eval_samples is not None:
                    raw_datasets["eval"] = (
                        raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
                    )

    sampling_rate = feature_extractor.sampling_rate
    model_input_name = feature_extractor.model_input_names[0]

    if not dataset_was_precomputed:
        def prepare_dataset(audio, labels):
            batch = {}
            batch["length"] = len(audio["array"])
            batch["labels"] = preprocess_labels(labels)
            return batch

        with training_args.main_process_first():
            tmp_datasets = raw_datasets.map(
                prepare_dataset,
                num_proc=data_args.preprocessing_num_workers,
                input_columns=["audio", "labels"],
                remove_columns=[col for col  in next(iter(raw_datasets.values())).column_names if col != "labels"], # this is a trick to avoid to rewrite the entire audio column which takes ages
                desc="Computing audio length",
            )
            
        for split in raw_datasets:
            raw_datasets[split] = concatenate_datasets([raw_datasets[split].remove_columns(["labels"]), tmp_datasets[split]], axis=1)

    # filter training data with inputs < min_input_length
    min_input_length = data_args.min_length_seconds * sampling_rate

    if not dataset_was_precomputed:
        def is_audio_valid(input_length):
            return input_length > min_input_length

        with training_args.main_process_first():
            raw_datasets = raw_datasets.filter(
                is_audio_valid,
                input_columns=["length"],
                num_proc=data_args.preprocessing_num_workers,
                desc="Filtering by audio length",
            )

        # filter training data with non-valid labels
        def is_label_valid(label):
            return label != "Unknown"

        with training_args.main_process_first():
            raw_datasets = raw_datasets.filter(
                is_label_valid,
                input_columns=["labels"],
                num_proc=data_args.preprocessing_num_workers,
                desc="Filtering by labels",
            )
        
        if training_args.do_train and data_args.max_samples_per_label:
            label_names = set(raw_datasets["train"]["labels"])
            labels = np.array(raw_datasets["train"]["labels"])
            indices = np.arange(len(labels))
            indices_to_keep = []
            set_seed(training_args.seed)
            for label in label_names:
                label_indices = indices[labels==label]
                label_indices = np.random.choice(label_indices, size=min(data_args.max_samples_per_label, len(label_indices)), replace=False)
                indices_to_keep.extend(np.random.choice(label_indices, size=min(data_args.max_samples_per_label, len(label_indices)), replace=False))
                    
            with training_args.main_process_first():
                raw_datasets["train"] = raw_datasets["train"].select(indices_to_keep).shuffle(seed=training_args.seed)

        # Print a summary of the labels to the stddout (helps identify low-label classes that could be filtered)
        # sort by freq
        count_labels_dict = Counter(raw_datasets["train"]["labels"])
        count_labels_dict = sorted(count_labels_dict.items(), key=lambda item: (-item[1], item[0]))
        labels, frequencies = zip(*count_labels_dict)
        total_labels = sum(frequencies)
        labels_to_remove = []

        logger.info(f"{'Accent':<15} {'Perc.':<5}")
        logger.info("-" * 20)
        for lab, freq in zip(labels, frequencies):
            freq = 100 * freq / total_labels
            logger.info(f"{lab:<15} {freq:<5}")
            if freq < data_args.filter_threshold:
                labels_to_remove.append(lab)

        if len(labels_to_remove):
            # filter training data with label freq below threshold
            def is_label_valid(label):
                return label not in labels_to_remove

            logger.info(f"Labels removed: {labels_to_remove}")
            
            with training_args.main_process_first():
                raw_datasets = raw_datasets.filter(
                    is_label_valid,
                    input_columns=["labels"],
                    num_proc=data_args.preprocessing_num_workers,
                    desc="Filtering low freq labels",
                )

        # We'll include these in the model's config to get human readable labels in the Inference API.
        set_labels = set(raw_datasets["train"]["labels"])
        if training_args.do_eval:
            logger.info(f"The following accents are present in the eval set but not the test set: {set(raw_datasets['eval']['labels']) - set_labels}")
            set_labels = set_labels.union(set(raw_datasets["eval"]["labels"]))
        label2id, id2label = {}, {}
        for i, label in enumerate(sorted(list(set_labels))):
            label2id[label] = str(i)
            id2label[str(i)] = label
            
        with training_args.main_process_first():
            raw_datasets = raw_datasets.map(
                    lambda label: {"labels_id": int(label2id[label])},
                    input_columns=["labels"],
                    num_proc=data_args.preprocessing_num_workers,
                    desc="Apply labels id",
                )
    else:
        label2id, id2label = {}, {}

        # TODO: slow - probably not best
        if training_args.do_train:
            for sample in raw_datasets["train"]:
                candidate = label2id.get(sample["labels"])
                if candidate is not None and candidate != sample["labels_id"]:
                    print(f"issue {candidate} should be {sample['labels_id']}")
                label2id[sample["labels"]] = sample["labels_id"]
        if training_args.do_eval:
            for sample in raw_datasets["eval"]:
                candidate = label2id.get(sample["labels"])
                if candidate is not None and candidate != sample["labels_id"]:
                    print(f"issue {candidate} should be {sample['labels_id']}")
                label2id[sample["labels"]] = sample["labels_id"]
                
        # if training_args.do_train:
        #     label2id = dict(zip(raw_datasets["train"]["labels"], raw_datasets["train"]["labels_id"]))
        # if training_args.do_eval:
        #     label2id.update(dict(zip(raw_datasets["eval"]["labels"], raw_datasets["eval"]["labels_id"])))
        id2label = {str(val): key for (key, val) in label2id.items()}
        
    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="audio-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # adapt config with regularization
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "layerdrop": model_args.layerdrop,
            "activation_dropout": model_args.activation_dropout,
            "use_weighted_layer_sum": model_args.use_weighted_layer_sum,
        }
    )

    if model_args.use_last_embedding_layer:
        model = SequenceClassificationModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        
        if model_args.freeze_base_model: 
            model.compute_w2v2 = False
    else:
        model = AutoModelForAudioClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    # freeze the convolutional waveform encoder for wav2vec2-style models
    if model_args.freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        else:
            raise ValueError("Method for freezing the feature encoder is not defined for Whisper-style models.")

    if model_args.freeze_base_model:
        if hasattr(model, "freeze_base_model"):
            # wav2vec2-style models
            model.freeze_base_model()
            if hasattr(model, "freeze_feature_encoder"):
                model.freeze_feature_encoder()
        elif hasattr(model, "freeze_encoder"):
            # whisper-style models
            model.freeze_encoder()
        else:
            raise ValueError("Method for freezing the base module of the audio encoder is not defined")

    if model_args.freeze_base_model and model_args.use_last_embedding_layer:
        if not dataset_was_precomputed:
            # precomputing hidden states
            from torch.utils.data import DataLoader
            from accelerate import Accelerator
            
            _HIDDEN_STATES_START_POSITION = 2
            
            if training_args.fp16:
                mixed_precision = "fp16"
            elif training_args.bf16:
                mixed_precision = "bf16"
            else:
                mixed_precision = "no"

            accelerator = Accelerator(
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                mixed_precision=mixed_precision,
                project_dir=training_args.output_dir,
            )
            
            audio_data_collator = DataCollatorFeatureExtractorWithPadding(feature_extractor, max_length_seconds=data_args.max_length_seconds, feature_extractor_input_name=feature_extractor_input_name)

            for split in raw_datasets:
                data_loader = DataLoader(
                    raw_datasets[split],
                    batch_size=training_args.per_device_train_batch_size, # TODO: chose another one
                    collate_fn=audio_data_collator,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=True,
                )
                data_loader = accelerator.prepare(data_loader)        
                
                all_encoder_outputs = []
                for batch in tqdm(data_loader, disable=not accelerator.is_local_main_process):
                    model.wav2vec2.to(batch[feature_extractor_input_name].device)
                    
                    with torch.no_grad():
                        encoder_outputs = model.wav2vec2(batch[feature_extractor_input_name], attention_mask=batch.get("attention_mask", None), output_hidden_states=True)
                    hidden_states = encoder_outputs[_HIDDEN_STATES_START_POSITION][-1]
                    if batch.get("attention_mask", None) is None:
                        hidden_states = hidden_states.mean(dim=1)
                    else:
                        padding_mask = model._get_feature_vector_attention_mask(hidden_states.shape[1], batch.get("attention_mask", None))
                        hidden_states[~padding_mask] = 0.0
                        hidden_states = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

                    encoder_outputs = accelerator.gather_for_metrics(hidden_states)
                    
                    # TODO: check it works multi device
                    if accelerator.is_main_process:
                        all_encoder_outputs.extend(encoder_outputs.to("cpu").numpy())
                        
                if accelerator.is_main_process:
                    tmp_hidden_states = Dataset.from_dict({"hidden_states": all_encoder_outputs})
                    tmp_hidden_states.save_to_disk(os.path.join(data_args.temporary_save_to_disk, split))
                accelerator.wait_for_everyone()
                del all_encoder_outputs

                tmp_hidden_states = datasets.load_from_disk(os.path.join(data_args.temporary_save_to_disk, split))
                with accelerator.main_process_first():
                    raw_datasets[split] = concatenate_datasets([raw_datasets[split].remove_columns("audio"), tmp_hidden_states], axis=1)
                        
            accelerator.free_memory()
            del data_loader, batch, accelerator
            
        data_collator = DataCollatorHiddenStatesPadding()
    else:
        data_collator = DataCollatorFeatureExtractorWithPadding(feature_extractor, max_length_seconds=data_args.max_length_seconds)
    
    if data_args.save_to_disk is not None and not dataset_was_precomputed:
        raw_datasets.save_to_disk(data_args.save_to_disk)
        logger.info(f"Dataset saved at {data_args.save_to_disk}. Be careful of changing data parameters, which won't change the current saved dataset if reloaded.")

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    ignore_keys = ["hidden_states","attentions"] if model_args.use_weighted_layer_sum else None

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=ignore_keys)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(ignore_keys=ignore_keys)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "audio-classification",
        "dataset": data_args.train_dataset_name.split("+")[0],
        "tags": ["audio-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
