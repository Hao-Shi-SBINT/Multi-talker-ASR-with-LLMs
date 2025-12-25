# Created by Hao at 2025-06-30
import os
import sys
import logging
import datasets

from dataclasses import dataclass, field
import transformers

from src.arguments import ModelArguments, DataTrainingArguments
from src.dataset_loader import load_dataset_or_fail
from src.feature_extractor_loader import load_feature_extractor
from src.config_loader import load_config
from src.tokenizer_loader import load_tokenizer
from src.model_loader import load_aed_model
from src.insert_adapter_decoder import insert_adapters
from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.trainer_seq2seq import Seq2SeqTrainer

from utils.checkpoint_checking_utils import resume_or_raise
from utils.param_utils import checking_trainable_params
from utils.unfreeze_utils import unfreeze_selected_params
from utils.freeze_utils import freeze_model
from utils.resample_dataset_utils import maybe_resample_dataset
from utils.vectorized_dataset_utils import preprocess_and_filter
from utils.metric_utils import compute_metrics
from utils.processor_utils import save_and_create_processor
from utils.training_stats_utils import build_training_kwargs
from utils.load_sep_ctc_from_partial import load_sep_ctc_from_partial

from transformers import (
    HfArgumentParser, 
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.utils import check_min_version, send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from safetensors.torch import save_file, load_file
from safetensors.torch import load_model, save_model


require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")


def main():
    # 1. Set configurations
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_speech_recognition_seq2seq", model_args, data_args, training_args)

    if model_args.pretrain_separator_path in ("None", "none", ""):
        pretrain_separator_path = None
    else:
        pretrain_separator_path = model_args.pretrain_separator_path

    # 2. Set logs
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    if not is_main_process(training_args.local_rank):
        logger.setLevel(logging.WARN)

    # Log on each process the small summary:
    logger.info("Training settings %s", training_args)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_ckpt = resume_or_raise(training_args, logger)
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = load_dataset_or_fail(data_args, logger)

    # 5. Load pretrained model, tokenizer, and feature extractor
    config = load_config(model_args)
    config.talker_ctc = model_args.talker_ctc
    config.talker_ctc_refine = model_args.talker_ctc_refine
    config.talker_numbers = model_args.talker_numbers
    config.separator_hidden = model_args.separator_hidden
    config.train_mode = model_args.train_mode
    config.ctc_alpha = model_args.ctc_alpha
    config.ctc_bridge = model_args.ctc_bridge
    config.ctc_bridge_type = model_args.ctc_bridge_type
    config.decoder_cross_attention = model_args.decoder_cross_attention
    config.decoder_cross_attention_type = model_args.decoder_cross_attention_type
    config.decoder_cross_attention_feature = model_args.decoder_cross_attention_feature
    logger.info("Model configuration %s", config)

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = load_feature_extractor(model_args, logger)
    logger.info("Feature extractor configuration %s", feature_extractor)

    tokenizer = load_tokenizer(model_args, logger)
    logger.info("Tokenizer %s", tokenizer)

    model = load_aed_model(model_args, config, logger)
    if(pretrain_separator_path):
        model = load_sep_ctc_from_partial(model, pretrain_separator_path)

    # 6. Insert adapters into deocder and set the trainable parameters
    # If the training mode is 'ctc': do not insert adapter and we should freeze all parameters
    if(model_args.train_mode == 'ctc'):
        freeze_model(model)
        unfreeze_selected_params(model, model_args.train_mode, model_args)
    else:
        freeze_model(model)
        if(model_args.adapter_only_decoder):
            insert_adapters(model, model_args, config)
        unfreeze_selected_params(model, model_args.train_mode, model_args)

    # 7. Some other settings for configuration
    # Here we write a new get_input_embeddings for fix the undefined of pre-defined function of SpeechEncoderDecoderModel
    def get_input_embeddings(self):
        return self.decoder.model.embed_tokens
    model.get_input_embeddings = get_input_embeddings.__get__(model)

    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # 8. Dataset
    raw_datasets = maybe_resample_dataset(raw_datasets, data_args, feature_extractor)
    vectorized_datasets = preprocess_and_filter(
            raw_datasets,
            data_args,
            feature_extractor,
            tokenizer,
            config,
            training_args,
            )

    # 9. Create a single speech processor
    processor = save_and_create_processor(
        training_args, feature_extractor, tokenizer, config
    )

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        decoder_end_token_id=model.config.eos_token_id,
        config=config,
    )

    # Some Special settings for increase dataloding speed
    # training_args.remove_unused_columns = False
    # training_args.dataloader_num_workers = 4
    # training_args.dataloader_pin_memory = True
    # training_args.group_by_length = False

    # 11. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics(tokenizer, cache_dir=model_args.cache_dir, ignore_id=model.config.ignore_token_id) if training_args.predict_with_generate else None,
    )

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_ckpt is not None:
            checkpoint = last_ckpt

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        best_model = trainer.model
        if(model_args.train_mode == 'ctc'):
            save_model(best_model, os.path.join(training_args.output_dir, "model.safetensors"))
        else:
            save_model(best_model, os.path.join(training_args.output_dir, "model_unmerge.safetensors"))

        tokenizer.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Write Training Stats
    kwargs = build_training_kwargs(model_args, data_args)

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
