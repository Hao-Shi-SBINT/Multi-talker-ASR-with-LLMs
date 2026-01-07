# Created by Hao at 2025-06-30
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import argparse

class CSVAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        flat = []
        for v in values:
            flat.extend([s for s in v.split(",") if s])
        setattr(namespace, self.dest, flat)

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
    adapter_only_decoder: bool = field(
        default=True,
        metadata={"help": "Whether the adapters is only inserted into Decoder."},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    pretrain_separator_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of pre-trained separator and serilized CTCs"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    talker_ctc: bool = field(
        default=False,
        metadata={"help": "Whether to use ctc after speech encoder."},
    )
    talker_ctc_refine: bool = field(
        default=False,
        metadata={"help": "Whether to use ctc refiner after speech encoder."},
    )
    talker_numbers: int = field(
        default=2,
        metadata={"help": "The number of talker."},
    )
    separator_hidden: int = field(
        default=896,
        metadata={"help": "The number of hidden nodes of separator."},
    )
    ctc_bridge: bool = field(
        default=False,
        metadata={"help": "Whether to use ctc bridge module."},
    )
    ctc_bridge_type: str = field(
        default="raw",
        metadata={"help": "Type of CTC bridge module."}
    )
    decoder_cross_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use cross attention module."},
    )
    decoder_cross_attention_type: str = field(
        default="tiny",
        metadata={"help": "Type of cross attention module."}
    )
    decoder_cross_attention_feature: str = field(
        default="raw",
        metadata={"help": "Feature of cross attention module."}
    )
    decoder_cross_attention_dynamic: str = field(
        default="false",
        metadata={"help": "Layer gates for cross attention"}
    )
    decoder_cross_attention_dynamic_threshold: float = field(
        default=0.0,
        metadata={"help": "Threshold for layer gating."},
    )
    decoder_cross_attention_dynamic_loss : bool = field(
        default=False,
        metadata={"help": "Whether to use dynamic loss for training cross-attention gate."},
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
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    freeze_decoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    train_mode: str = field(
        default="attention",
        metadata={"help": "The mode for training: only ctc / only attention / attention-ctc hybrid: please set is as one of ctc/attention/hybrid."},
    )
    ctc_alpha: float = field(
        default=0.7,
        metadata={"help": "CTC loss weight (0–1)."},
    )
    ctc_decoding: bool = field(
        default=False, metadata={"help": "Whether using CTC for decoding."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={"help": "Deprecated. Please use the `language` and `task` arguments instead."},
    )
    suppress_tokens: List[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated. The use of `suppress_tokens` should not be required for the majority of fine-tuning examples."
                "Should you need to use `suppress_tokens`, please manually update them in the fine-tuning script directly."
            )
        },
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )
    partial_encoder_unfreeze: List[str] = field(
        default_factory=lambda: ["masked_spec_embed"],
        metadata={
            "help": (
                "Comma-separated substrings of *encoder* parameter names to keep "
                "trainable, e.g. 'adapter,masked_spec_embed'. "
                "Empty string → no partial unfreeze."
            ),
            "action": CSVAction,
        },
    )
    partial_decoder_unfreeze: List[str] = field(
        default_factory=lambda: ["lm_head", "embed_tokens", "embed_positions", "layernorm_embedding"],
        metadata={
            "help": (
                "Comma-separated substrings of *decoder* parameter names to keep "
                "trainable, e.g. 'lm_head,embed_tokens'."
            ),
            "action": CSVAction,
        },
    )
    partial_others_unfreeze: List[str] = field(
        default_factory=lambda: ["enc_to_dec_proj", "ctc"],
        metadata={
            "help": (
                "Comma-separated substrings of any parameter names (encoder/decoder) "
                "to keep trainable, e.g. 'enc_to_dec_proj,ctc'."
            ),
            "action": CSVAction,
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
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
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
