# Created by Hao: 2025-07-01
# Modified from the original file at:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py

from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import os
import sys
parent = os.path.abspath(os.path.join(__file__, "..", ".."))
if parent not in sys.path:
    sys.path.insert(0, parent)
from utils.generation_utils import GenerationMixin_Instruct
from utils.generation_ctc_utils import GenerationMixin_CTC
from utils.split_labels_by_sc import split_k_speakers_and_lengths

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.speech_encoder_decoder.configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig

from modeling_llama import LlamaForCausalLM
from modeling_wavlm import WavLMModel
from separator import Separator
from tiny_crossatt_module import TinyCrossAttnAdapter
from gate_tiny_crossatt_module import GatedTinyCrossAttnAdapter
from ctcaware_crossatt_module import CTCAwareTinyCrossAttnAdapter
from ctc import CTC
from down_sampling import WavLMPostDownsample
from losses import HybridLoss
from mt_ctctoken_builder import MultiSpkCTCTokenBuilder

from refiners_weightsconcat import (
    CTCPerSpeakerExtractorConcatSoftmax,
    CTCPerSpeakerExtractorConcatNNG,
)
from serilized_feature_refine import CTCAwareFrameRefiner

from ctc_prompt import (
    build_multi_ctc_prefix_from_heads,
)

from torch.nn.utils.rnn import pad_sequence

import logging
logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "SpeechEncoderDecoderConfig"

# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def align_mask_len(mask_bt, T_target):
    T = mask_bt.size(1)
    if T == T_target:
        return mask_bt
    if T > T_target:                      # 多1帧 → 直接截掉
        return mask_bt[:, :T_target]
    # 少1帧 → 在末尾补最后一个值（一般是 False/0）
    pad_val = mask_bt[:, -1:].clone()
    return torch.cat([mask_bt, pad_val], dim=1)


class SpeechEncoderDecoderModelLlama(PreTrainedModel, GenerationMixin_Instruct):
    config_class = SpeechEncoderDecoderConfig
    base_model_prefix = "speech_encoder_decoder"
    main_input_name = "inputs"
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = WavLMModel._from_config(config.encoder)

        if decoder is None:
            decoder = LlamaForCausalLM._from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.config.encoder._attn_implementation = self.encoder.config._attn_implementation
        self.config.decoder._attn_implementation = self.decoder.config._attn_implementation
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # For special tokens
        self.ignore_token_id   = getattr(self.config, "ignore_token_id", None)
        self.pad_token_id      = getattr(self.config, "pad_token_id",   None)
        self.sc_token_id       = getattr(self.config, "sc_token_id",    None)
        self.talker_ctc        = getattr(self.config, "talker_ctc",     False)
        self.talker_ctc_refine = getattr(self.config, "talker_ctc_refine",     False)
        self.talker_numbers    = getattr(self.config, "talker_numbers", 2)
        self.instruct          = getattr(self.config, "instruct",       False)
        self.eos_token_id      = getattr(self.config, "eos_token_id", None)
        self.ctc_bridge        = getattr(self.config, "ctc_bridge", False)
        self.ctc_bridge_type   = getattr(self.config, "ctc_bridge_type", "raw")
        self.decoder_cross_attention = getattr(self.config, "decoder_cross_attention", False)
        self.decoder_cross_attention_type = getattr(self.config, "decoder_cross_attention_type", "tiny")
        self.decoder_cross_attention_feature = getattr(self.config, "decoder_cross_attention_feature", "raw")
        self.decoder_cross_attention_dynamic = getattr(self.config, "decoder_cross_attention_dynamic", "false")
        self.decoder_cross_attention_dynamic_threshold = getattr(self.config, "decoder_cross_attention_dynamic_threshold", 0.0)
        self.decoder_cross_attention_dynamic_loss = getattr(self.config, "decoder_cross_attention_dynamic_loss", False)

        self.ctc_token_builder = MultiSpkCTCTokenBuilder()

        if self.instruct:
            self.bosp_token_id = self.config.bosp_token_id
            self.bosr_token_id = self.config.bosr_token_id
            self.boss_token_id = self.config.boss_token_id
            self.eosp_token_id = self.config.eosp_token_id
            self.eosr_token_id = self.config.eosr_token_id
            self.eoss_token_id = self.config.eoss_token_id

        if self.talker_ctc:
            self.separator = Separator(
                in_dim=self.config.encoder.output_hidden_size,
                hidden_size=config.separator_hidden,
                talker_numbers=self.talker_numbers
            )
            self.ctc_blank_id = config.decoder.vocab_size+1
            def make_ctc():
                return CTC(
                    odim=self.ctc_blank_id,
                    encoder_output_size=config.encoder.output_hidden_size
                )
            self.serialized_ctc = nn.ModuleList([make_ctc() for _ in range(self.talker_numbers)])
        else:
            self.ctc_blank_id = config.decoder.vocab_size+1
            self.serialized_ctc = []

        if self.talker_ctc_refine:
            self.serilized_refine = CTCAwareFrameRefiner(
                self.config.encoder.output_hidden_size,
            )

        if self.decoder_cross_attention:
            if(self.decoder_cross_attention_type == "tiny"):
                self.cross_att_adap = nn.ModuleList([
                    TinyCrossAttnAdapter(
                        hidden_size=self.decoder.config.hidden_size,
                        mem_dim=self.config.encoder.output_hidden_size,
                        attn_dim=512,
                        dropout=self.config.decoder.attention_dropout,
                    )
                    for _ in range(self.decoder.config.num_hidden_layers)
                ])
            elif(self.decoder_cross_attention_type == "gatetiny"):
                self.cross_att_adap = nn.ModuleList([
                    GatedTinyCrossAttnAdapter(
                        hidden_size=self.decoder.config.hidden_size,
                        mem_dim=self.config.encoder.output_hidden_size,
                        attn_dim=512,
                        dropout=self.config.decoder.attention_dropout,
                    )
                    for _ in range(self.decoder.config.num_hidden_layers)
                ])
            elif(self.decoder_cross_attention_type == "ctcaware"):
                self.cross_att_adap = nn.ModuleList([
                    CTCAwareTinyCrossAttnAdapter(
                        hidden_size=self.decoder.config.hidden_size,
                        mem_dim=self.config.encoder.output_hidden_size,
                        attn_dim=512,
                        dropout=self.config.decoder.attention_dropout, 
                    )
                    for _ in range(self.decoder.config.num_hidden_layers)
                ])

        else:
            self.cross_att_adap = None

        if self.decoder_cross_attention_dynamic:
            self.layer_gate_logits = nn.Parameter(
                torch.full((self.decoder.config.num_hidden_layers,), float(-2.0))
            )
        else:
            self.layer_gate_logits = None

        # get encoder output hidden size
        self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.ctc_bridge:
            if(self.ctc_bridge_type == "softmax"):
                # Currently help for 2-talker, but no significant improment for 3-talker
                self.ctc_extractor_concat = CTCPerSpeakerExtractorConcatSoftmax(
                    d_in=self.config.encoder.output_hidden_size,
                    d_model=self.config.encoder.output_hidden_size,
                    K_spk=self.talker_numbers,
                    use_repair=True,
                    n_heads=8,
                    band_repair=24,
                    resample_mode="nearest",
                )

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # we define the losses computing class here
        self.losses = HybridLoss(
                alpha=config.ctc_alpha, 
                mode=config.train_mode,
                blank_id=self.ctc_blank_id-1, 
                enable_blank_check=True,
                log_every_steps=100, 
                ) # using hybrid mode to do the initilization, then we will rewrite the mode

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def freeze_feature_encoder(self):
        self.encoder.freeze_feature_encoder()

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the SpeechEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for SpeechEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.
        Example:

        ```python
        >>> from transformers import SpeechEncoderDecoderModel

        >>> # initialize a wav2vec2bert from a pretrained Wav2Vec2 and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
        >>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "facebook/wav2vec2-base-960h", "google-bert/bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./wav2vec2bert")
        >>> # load fine-tuned model
        >>> model = SpeechEncoderDecoderModel.from_pretrained("./wav2vec2bert")
        ```"""
        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def generate(self, *args, **kwargs):
        """
        Forward every call to the mix-in’s version explicitly.
        """
        return GenerationMixin_Instruct.generate(self, *args, **kwargs)

    def generate_ctc(self, *args, **kwargs):
        """
        Forward every call to the mix-in’s version explicitly.
        """
        return GenerationMixin_CTC.generate(self, *args, **kwargs)

    @staticmethod
    def _expand_inputs_for_generation(*args, **kwargs):
        # delegate to the mix-in’s staticmethod
        return GenerationMixin_Instruct._expand_inputs_for_generation(*args, **kwargs)

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer=None,
        **model_kwargs,
    ):
        """
        Forward every call to the mix-in’s custom `_sample`.
        """
        return GenerationMixin_Instruct._sample(
            self,
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    def _sample_ctc(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer=None,
        **model_kwargs,
    ):
        """
        Forward every call to the mix-in’s custom `_sample`.
        """
        return GenerationMixin_CTC._sample(
            self,
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )


    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        wavlm_hidden_stages   = encoder_outputs[1]      # un-downsampled feature
        wavlm_down_hidden_stages = encoder_outputs[2]
        mixed_encoding_feature = wavlm_hidden_stages

        # Here we add serialized CTC
        if self.talker_ctc:
            sep_hidden_states = self.separator(mixed_encoding_feature)
        else:
            sep_hidden_states = None

        # optionally project encoder_hidden_states
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # compute correct encoder attention mask
        if attention_mask is not None:
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
            if self.talker_ctc:
                encoder_attention_mask_ctc = self.encoder._get_feature_vector_attention_mask_x0(
                    mixed_encoding_feature.shape[1], attention_mask
                )
                # encoder_attention_mask_ctc = self.encoder._get_feature_vector_attention_mask_x4(
                #         mixed_encoding_feature.shape[1], attention_mask
                # )

            else:
                encoder_attention_mask_ctc = None
        else:
            encoder_attention_mask = None
            if self.talker_ctc:
                encoder_attention_mask_ctc = None

        # Here we add the serized refined CTC
        if self.talker_ctc_refine:
            sep_hidden_states = self.serilized_refine(
                sep_hidden_list=sep_hidden_states,
                mixed_hidden=mixed_encoding_feature,
                enc_mask=encoder_attention_mask_ctc,
                ctc_modules=self.serialized_ctc,
            )

        if self.decoder_cross_attention:
            acoustic_conf = None
            if(self.decoder_cross_attention_feature == "mix"):
                acoustic_mem = mixed_encoding_feature
                acoustic_mask = encoder_attention_mask_ctc
            elif(self.decoder_cross_attention_feature == "sep"):
                acoustic_mem = torch.cat(sep_hidden_states, dim=1)
                acoustic_mask = encoder_attention_mask_ctc.repeat(1, self.talker_numbers)
                T_target = acoustic_mem.size(1)
                acoustic_mask = align_mask_len(acoustic_mask, T_target)

            """
            if(self.decoder_cross_attention_type == "ctcaware"):
                acoustic_mem, acoustic_mask, acoustic_conf = self.ctc_token_builder(
                    sep_hidden_list=sep_hidden_states,
                    encoder_attention_mask_ctc=encoder_attention_mask_ctc,
                    ctc_modules=self.serialized_ctc,
                )
                acoustic_mask = ~acoustic_mask
            """

        # Here we check whether use CTC bridge module
        if self.ctc_bridge:
            # directly use the separated encoding
            if(self.ctc_bridge_type == "raw"):
                X_ref = torch.cat(sep_hidden_states, dim=1)
                X_ref, _ = self.encoder.adapter(X_ref)
                X_ref = self.enc_to_dec_proj(X_ref)
                encoder_hidden_states = X_ref

                encoder_attention_mask = encoder_attention_mask.repeat(1, self.talker_numbers)
                T_target = encoder_hidden_states.size(1)
                encoder_attention_mask = align_mask_len(encoder_attention_mask, T_target)

            # ASRU 2025 proposed method: concat ctc transcription embedding + mixed encoding
            elif(self.ctc_bridge_type == "ctcprompt"):
                # Here we add serialized CTC
                ctc_transcription_list = []
                for i, ctc_head in enumerate(self.serialized_ctc):
                    _argmax = ctc_head.argmax(sep_hidden_states[i])
                    _transcription, _transcription_shape = \
                        self.ctc_remove_duplicates_and_blank(_argmax, blank_id=self.ctc_blank_id-1, pad_id=self.pad_token_id)
                    ctc_transcription_list.append(_transcription)

                ctc_prefix_embeds, ctc_prefix_mask, ctc_prefix_ids = build_multi_ctc_prefix_from_heads(
                    ctc_transcription_list=ctc_transcription_list,
                    decoder=self.decoder,
                    pad_id=self.pad_token_id,
                    max_prefix_len_per_head=None,
                )

                encoder_hidden_states = torch.cat(
                    [ctc_prefix_embeds, encoder_hidden_states],
                    dim=1,
                )

                encoder_attention_mask = torch.cat(
                    [ctc_prefix_mask, encoder_attention_mask.bool()],
                    dim=1,
                )   # [B, Lp_total + Tm]

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            # labels-> decoder_input_ids : add  
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            if self.instruct:
                # TODO: here we only use same prompt, so it should be modified when the prompts are different
                skip_eosr_ids = decoder_input_ids.masked_fill(
                    decoder_input_ids == self.eosr_token_id,
                    self.config.pad_token_id
                )
                _bosr_pos = (skip_eosr_ids[0] == self.bosr_token_id).nonzero(as_tuple=True)[0]
                splited_decoder_input_ids = skip_eosr_ids[:, _bosr_pos+1:]
            else:
                splited_decoder_input_ids = decoder_input_ids[:, 1:]

            label_spks, label_spks_lengths = split_k_speakers_and_lengths(
                    labels=splited_decoder_input_ids,
                    k_speakers=self.talker_numbers,
                    sep_id=self.sc_token_id,
                    pad_token_id=self.config.pad_token_id,
                    end_token_id=self.config.pad_token_id,
                    ignore_id=-100,
                    allow_empty_segment=False,
            )

            # Here we make the speech-padded labels: with self.ignore_token_id (-100)
            batch, speech_len, _ = encoder_hidden_states.shape

            # Insert <eos>
            # For input_ids, the <eos> should not be inserted, the <pad> is inserted
            # <eos> is only inserted into labels
            decoder_ids_pad = torch.full((batch, 1), self.pad_token_id, device=labels.device)
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_ids_pad), dim=1)
            pad_eos = torch.full((batch, 1), self.ignore_token_id, device=labels.device)
            labels = torch.cat((labels, pad_eos), dim=1)

            mask = (labels == self.ignore_token_id) # self.ignore_token_id = -100
            first_pad_id = mask.float().argmax(dim=1)
            # Here we add the <eos> in labels
            labels[torch.arange(batch), first_pad_id] = eos_token_id = self.config.eos_token_id[0] if isinstance(self.config.eos_token_id, (list, tuple)) else self.config.eos_token_id

            # Compute the length of prompt
            # !!!! Should be fixed a little bit later
            # Currently, since all the prompts are same, we directly use one sample to compute the length
            # TODO: But should be modifed for variable-prompt condition
            if self.instruct:
                seq = labels[0]
                len_prompt = (seq.eq(self.eosp_token_id).nonzero()[0] - seq.eq(self.bosp_token_id).nonzero()[0] - 1).item()

                # The ignore part during computing loss:
                # (<bos_prompt>, prompt, <eos_prompt>, <bos_speech>, speech_emb, <eos_speech>, <bos_response>)
                # The corresponding length is:
                # (1, prompt_length, 1, 1, speech_length, 1, 1) => prompt_length + speech_length + 5
                # Currently, the labels is:
                # (<bos_prompt>, prompt, <eos_prompt>, <bos_speech>, speech_emb, <eos_speech>, <bos_response>, transcription)
                # For simple processing, we directly generate a mask for the above part (without transcription)
                # and use the text contents (after <bos_response>)
                ignore_contents_mask = torch.full((batch, speech_len + len_prompt + 5), self.config.ignore_token_id, device=labels.device)
                bos_response_idx = labels[0].eq(self.bosr_token_id).nonzero()[0]
                contents = labels[:, bos_response_idx+1:]
                labels = torch.cat((ignore_contents_mask, contents), dim=1)

            else:
                # The ignore part during computing loss: 
                # (speech_emb)
                # The corresponding length is:
                # (speech_length) => speech_length
                # Currently, the labels is:
                # (prompt, text)
                # For simple processing, we directly generate a mask for the above part
                ignore_contents_mask = torch.full((batch, speech_len), self.config.ignore_token_id, device=labels.device)
                labels = torch.cat((ignore_contents_mask, labels), dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            acoustic_mem=acoustic_mem, 
            acoustic_sep=sep_hidden_states,
            acoustic_mask=~acoustic_mask,
            acoustic_conf=acoustic_conf,
            acoustic_ctc_mask=encoder_attention_mask_ctc,
            adaptation_modules=self.cross_att_adap,
            adaptation_layer_gate_modules=self.layer_gate_logits,
            adaptation_layer_gate_modules_threshold=self.decoder_cross_attention_dynamic_threshold,
            ctc_modules=self.serialized_ctc,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        ctc_per_head = None
        if labels is not None:
            shared_params = list(self.encoder.parameters()) + list(self.separator.parameters())
            loss = self.losses(
                decoder_outputs=decoder_outputs,
                labels=labels,
                decoder_vocab_size=self.decoder.config.vocab_size,
                talker_ctc=self.serialized_ctc,
                sep_hidden_states=sep_hidden_states,
                encoder_attention_mask_ctc=encoder_attention_mask_ctc,
                label_spks=label_spks,
                label_spks_lengths=label_spks_lengths,
                cross_att_layer_gate=self.layer_gate_logits,
                cross_att_layer_gate_loss=self.decoder_cross_attention_dynamic_loss,
                talker_numbers=self.talker_numbers,
                shared_params=shared_params,
                return_dict=return_dict,
            )
            
        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        out = Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        ctc_per_head = getattr(self.losses, "last_ctc_per_head", None)
        if ctc_per_head is not None:
            out.ctc_per_head = ctc_per_head   # or out["ctc_per_head"]=...

        return out

    def forward_ctc(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        wavlm_hidden_stages   = encoder_outputs[1]

        # Here we add serialized CTC
        ctc_transcription_list = []
        if self.talker_ctc:
            # wavlm_hidden_stages, new_length = self.down_sampling(wavlm_hidden_stages)
            sep_hidden_states = self.separator(wavlm_hidden_stages)
            for i, ctc_head in enumerate(self.serialized_ctc):
                _argmax = ctc_head.argmax(sep_hidden_states[i])
                _transcription, _transcription_shape = \
                    self.ctc_remove_duplicates_and_blank(_argmax, blank_id=self.ctc_blank_id-1, pad_id=self.pad_token_id)
                ctc_transcription_list.append(_transcription)

        ctc_transcription = torch.cat(ctc_transcription_list, dim=1)

        return ctc_transcription

    def ctc_remove_duplicates_and_blank(
        self,
        argmax_tensor: torch.Tensor,
        blank_id: int = 128258,
        pad_id: int = 128257,
        collapse_across_blanks: bool = True,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Remove CTC blanks and collapse duplicates from argmax outputs, then right-pad.

        Behavior:
          - Always removes tokens equal to `blank_id`.
          - If `collapse_across_blanks` is True:
              Collapses duplicates even when they are separated by blanks.
              Example: A, blank, A  ->  A
            If False:
              Only collapses strictly adjacent duplicates (classic CTC collapse).
              Example: A, A, blank, A  ->  A, blank, A
          - Ignores any `pad_id` encountered in the argmax (if present).
          - Pads all sequences on the right with `pad_id` to a common length.

        Args:
            argmax_tensor: Long tensor of shape (B, T). Each row is argmax over time.
            blank_id: CTC blank token id.
            pad_id: Padding token id.
            collapse_across_blanks: Whether to collapse duplicates across blanks.

        Returns:
            padded_batch: Long tensor of shape (B, Lmax) padded with `pad_id`.
            lengths: List of true (unpadded) lengths per sequence.
        """
        batch_sequences: List[torch.Tensor] = []
        lengths: List[int] = []

        # Iterate over each sequence in the batch (shape per seq: (T,))
        for seq in argmax_tensor:
            processed: List[int] = []
            # Tracks the last token that was actually kept (non-blank)
            last_kept = None

            # Convert to Python list for simplicity and device safety
            for token in seq.detach().cpu().tolist():
                # Skip any padding that leaked into argmax sequence
                if token == pad_id:
                    continue

                # Skip blanks and DO NOT update last_kept
                if token == blank_id:
                    continue

                if collapse_across_blanks:
                    # If the token equals the last kept non-blank token, skip it
                    if last_kept is not None and token == last_kept:
                        continue
                else:
                    # Classic CTC: only collapse strictly adjacent duplicates
                    if processed and token == processed[-1]:
                        continue

                # Keep the token and update last_kept
                processed.append(token)
                last_kept = token

            # Move back to the original device for padding
            out = torch.tensor(processed, dtype=torch.long, device=argmax_tensor.device)
            batch_sequences.append(out)
            lengths.append(len(processed))

        # Right-pad sequences to the same length using `pad_id`
        padded_batch = pad_sequence(batch_sequences, batch_first=True, padding_value=pad_id)
        return padded_batch, lengths

__all__ = ["SpeechEncoderDecoderModelLlama"]
