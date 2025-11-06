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
from ctc import CTC
from down_sampling import WavLMPostDownsample
from losses import HybridLoss

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
        self.ignore_token_id = getattr(self.config, "ignore_token_id", None)
        self.pad_token_id    = getattr(self.config, "pad_token_id",   None)
        self.sc_token_id     = getattr(self.config, "sc_token_id",    None)
        self.talker_ctc      = getattr(self.config, "talker_ctc",     False)
        self.talker_numbers  = getattr(self.config, "talker_numbers", 2)
        self.instruct        = getattr(self.config, "instruct",       False)
        self.eos_token_id    = getattr(self.config, "eos_token_id", None)

        if self.instruct:
            self.bosp_token_id = self.config.bosp_token_id
            self.bosr_token_id = self.config.bosr_token_id
            self.boss_token_id = self.config.boss_token_id
            self.eosp_token_id = self.config.eosp_token_id
            self.eosr_token_id = self.config.eosr_token_id
            self.eoss_token_id = self.config.eoss_token_id

        if self.talker_ctc:
            """
            self.down_sampling = WavLMPostDownsample(
                    d_in=self.config.encoder.output_hidden_size,
                    d_mid=2 * self.config.encoder.output_hidden_size,
                    d_out=self.config.encoder.output_hidden_size,
                )
            """
            self.separator = Separator(hidden_size=self.config.encoder.output_hidden_size, talker_numbers=self.talker_numbers)
            self.ctc_blank_id = config.decoder.vocab_size+1
            def make_ctc():
                return CTC(
                    odim=self.ctc_blank_id,
                    encoder_output_size=config.encoder.output_hidden_size
                )
            # Here I found that
            # Using modulelist the final ctc head can not be trained well
            # This may be caused by loss plateau
            # But currecntly I do not want to modify the training loop, so I directly define the ctc head 1, 2, 3
            self.serialized_ctc = nn.ModuleList([make_ctc() for _ in range(self.talker_numbers)])
            # self.ctc_spk1 = make_ctc()
            # self.ctc_spk2 = make_ctc()
            # if self.talker_numbers == 3:
            #     self.ctc_spk3 = make_ctc()

        # get encoder output hidden size
        self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # we define the losses computing class here
        self.losses = HybridLoss(
                alpha=0.7, 
                mode="hybrid",
                blank_id=self.ctc_blank_id-1, 
                enable_blank_check=True,
                log_every_steps=100, 
                ) # using hybrid mode to do the initilization, then we will rewrite the mode

    def reset_loss_mode(self, alpha=0.7, mode='hybrid'):
        self.losses.mode = mode
        self.losses.alpha = alpha

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
        wavlm_hidden_stages   = encoder_outputs[1]
        # wavlm_down_hidden_stages = encoder_outputs[2]

        # Here we add serialized CTC
        if self.talker_ctc:
            # wavlm_hidden_stages, new_length = self.down_sampling(wavlm_hidden_stages)
            sep_hidden_stages = self.separator(wavlm_hidden_stages)
            ctc_transcriptions = []

            # _argmax_spk1 = self.ctc_spk1.argmax(sep_hidden_stages[0])
            # _argmax_spk2 = self.ctc_spk2.argmax(sep_hidden_stages[1])

            # _transcription_spk1, _transcription_spk1_shape = self.ctc_remove_duplicates_and_blank(_argmax_spk1, blank_id=self.ctc_blank_id, pad_id=self.pad_token_id)
            # _transcription_spk2, _transcription_spk2_shape = self.ctc_remove_duplicates_and_blank(_argmax_spk2, blank_id=self.ctc_blank_id, pad_id=self.pad_token_id)
            # ctc_transcriptions.append(_transcription_spk1)
            # ctc_transcriptions.append(_transcription_spk2)

            # if(self.talker_numbers == 3):
            #     _argmax_spk3 = self.ctc_spk3.argmax(sep_hidden_stages[2])
            #     _transcription_spk3, _transcription_spk3_shape = self.ctc_remove_duplicates_and_blank(_argmax_spk3, blank_id=self.ctc_blank_id, pad_id=self.pad_token_id)
            #     ctc_transcriptions.append(_transcription_spk3)
            # for i, ctc_head in enumerate(self.serialized_ctc):
            #     _argmax = ctc_head.argmax(sep_hidden_stages[i])
            #     _transcription, _transcription_shape = self.ctc_remove_duplicates_and_blank(_argmax, blank_id=self.ctc_blank_id, pad_id=self.pad_token_id)
            #     ctc_transcriptions.append(_transcription)

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
                 # encoder_attention_mask_ctc, len_encoder_attention_mask_ctc = self.encoder._get_downsampled_feature_mask(
                encoder_attention_mask_ctc = self.encoder._get_feature_vector_attention_mask_without_adapter(
                    wavlm_hidden_stages.shape[1], attention_mask, add_adapter=None,
                )
        else:
            encoder_attention_mask = None
            if self.talker_ctc:
                encoder_attention_mask_ctc = None

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
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:

            # talker_ctc = [self.ctc_spk1, self.ctc_spk2]
            # if self.talker_numbers == 3:
            #     talker_ctc.append(self.ctc_spk3)

            loss = self.losses(
                decoder_outputs=decoder_outputs,
                labels=labels,
                decoder_vocab_size=self.decoder.config.vocab_size,
                talker_ctc=self.serialized_ctc,
                sep_hidden_states=sep_hidden_stages,
                encoder_attention_mask_ctc=encoder_attention_mask_ctc,
                label_spks=label_spks,
                label_spks_lengths=label_spks_lengths,
                talker_numbers=self.talker_numbers,
                return_dict=return_dict,
            )
            
            # Now we create a new class for losses computing, if the training goes well
            # the following contents should be deleted
            """
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0] 
            loss_fct = CrossEntropyLoss() 
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1)) 
            # Here we add serialized CTC loss 
            if self.talker_ctc: 
                hlens = encoder_attention_mask_ctc.sum(dim=1) 
                loss_ctc = 0 
                for i, ctc_head in enumerate(self.serialized_ctc[:self.talker_numbers]): 
                    loss_ctc += ctc_head(sep_hidden_states[i], hlens, label_spks[i], label_spks_lengths[i]) 

                loss_ctc /= self.talker_numbers 
                loss = loss * 0.7 + loss_ctc * 0.3
            """

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

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
            wavlm_hidden_stages, new_length = self.down_sampling(wavlm_hidden_stages)
            sep_hidden_states = self.separator(wavlm_hidden_stages)
            for i, ctc_head in enumerate(self.serialized_ctc):
                _argmax = ctc_head.argmax(sep_hidden_states[i])
                _transcription, _transcription_shape = \
                    self.ctc_remove_duplicates_and_blank(_argmax, blank_id=self.ctc_blank_id, pad_id=self.pad_token_id)
                ctc_transcription_list.append(_transcription)

        ctc_transcription = torch.cat(ctc_transcription_list, dim=1)

        return ctc_transcription

    def ctc_remove_duplicates_and_blank(
        self,
        argmax_tensor: torch.Tensor,
        blank_id: int = 128258,
        pad_id: int = 128257,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Process CTC argmax outputs by removing blanks and collapsing consecutive duplicates,
        then pad sequences to a uniform length.

        Args:
            argmax_tensor (torch.Tensor): Shape (B, Tmax). Argmax over CTC logits per timestep.
            blank_id (int): Token ID used for the CTC blank.
            pad_id (int): Token ID used for right-side padding.

        Returns:
            padded_batch (torch.Tensor): Shape (B, max_seq_len) with sequences padded by `pad_id`.
            lengths (List[int]): The true (unpadded) length of each processed sequence.
        """
        batch_sequences: List[torch.Tensor] = []
        lengths: List[int] = []

        for seq in argmax_tensor:
            processed_seq: List[int] = []
            prev_token = None

            # Convert to Python list for easy iteration (detach/CPU safe for general use)
            for token in seq.detach().cpu().tolist():
                # Keep token if it's not blank and not a duplicate of the previous token
                if token != blank_id and token != prev_token:
                    processed_seq.append(token)
                prev_token = token

            batch_sequences.append(torch.tensor(processed_seq, dtype=torch.long))
            lengths.append(len(processed_seq))

        # Right-pad all sequences to the same length using `pad_id`
        padded_batch = pad_sequence(batch_sequences, batch_first=True, padding_value=pad_id)

        return padded_batch, lengths


__all__ = ["SpeechEncoderDecoderModelLlama"]
