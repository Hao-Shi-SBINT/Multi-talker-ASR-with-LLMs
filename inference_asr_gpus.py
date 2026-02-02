# Created by Hao at 2025-06-30
import os
import sys
import logging
import datasets
import torch
import re

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
from utils.resample_dataset_utils import maybe_resample_dataset
from utils.vectorized_dataset_utils import preprocess_and_filter
from utils.metric_utils import compute_metrics
from utils.processor_utils import save_and_create_processor
from utils.training_stats_utils import build_training_kwargs

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

from datasets import DatasetDict, load_dataset, load_from_disk


require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

import torch.distributed as dist

def setup_distributed():
    # torchrun 会设置这些环境变量
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, world_size, local_rank, device
    else:
        # 单卡/非分布式
        return False, 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_distributed(is_dist):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()



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
    raw_datasets = DatasetDict()
    raw_datasets = load_from_disk(data_args.dataset_name)

    # 5. Load pretrained model, tokenizer, and feature extractor
    config = load_config(model_args)
    config.talker_ctc = model_args.talker_ctc
    config.talker_numbers = model_args.talker_numbers
    config.separator_hidden = model_args.separator_hidden
    config.train_mode = model_args.train_mode
    config.ctc_alpha = model_args.ctc_alpha
    config.ctc_bridge = model_args.ctc_bridge
    config.ctc_bridge_type = model_args.ctc_bridge_type
    config.decoder_cross_attention = model_args.decoder_cross_attention
    config.decoder_cross_attention_type = model_args.decoder_cross_attention_type
    config.decoder_cross_attention_feature = model_args.decoder_cross_attention_feature
    config.decoder_cross_attention_dynamic = model_args.decoder_cross_attention_dynamic
    config.decoder_cross_attention_dynamic_threshold = model_args.decoder_cross_attention_dynamic_threshold
    config.decoder_cross_attention_dynamic_loss = model_args.decoder_cross_attention_dynamic_loss
    config.decoder_cross_attention_dynamic_ratio = model_args.decoder_cross_attention_dynamic_ratio
    logger.info("Model configuration %s", config)

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = load_feature_extractor(model_args, logger)
    logger.info("Feature extractor configuration %s", feature_extractor)

    tokenizer = load_tokenizer(model_args, logger)
    logger.info("Tokenizer %s", tokenizer)

    model = load_aed_model(model_args, config, logger)
    model.eval()

    # Setting CUDA
    model = model.to("cuda")
    device = model.device

    # 7. Some other settings for configuration
    # Here we write a new get_input_embeddings for fix the undefined of pre-defined function of SpeechEncoderDecoderModel
    def get_input_embeddings(self):
        return self.decoder.model.embed_tokens
    model.get_input_embeddings = get_input_embeddings.__get__(model)

    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # 8. Dataset
    # raw_datasets = maybe_resample_dataset(raw_datasets, data_args, feature_extractor)
    vectorized_datasets = preprocess_and_filter(
            raw_datasets,
            data_args,
            feature_extractor,
            tokenizer,
            config,
            training_args,
            inference_mode=True,
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

    # 11. Define skip special tokens during inference
    def skip_special_tokens(est_text):
        allowed_special_tokens = ["<sc>", "<bos_prompt>", "<eos_prompt>", "<bos_speech>", "<eos_speech>", "<bos_response>", "<eos_response>"]
        tokens = re.findall(r"<[^>]+>|[^<>\s]+", est_text)
        processed_text = " ".join(
            token for token in tokens
            if token in allowed_special_tokens or not (token.startswith("<") and token.endswith(">"))
            )
        return processed_text

    is_dist, rank, world_size, local_rank, device = setup_distributed()

    # 12. Inference
    _set = os.path.basename(data_args.dataset_name)

    # 每个rank写各自的文件，避免多进程写同一个文件导致冲突
    label_path_rank = os.path.join(training_args.output_dir, f"{_set}_label.rank{rank}.text")
    decod_path_rank = os.path.join(training_args.output_dir, f"{_set}_decod.rank{rank}.text")

    model.eval()

    # datasets.Dataset 支持 shard，会更干净
    eval_ds = vectorized_datasets["eval"]
    if world_size > 1:
        eval_ds = eval_ds.shard(num_shards=world_size, index=rank, contiguous=True)

    with torch.inference_mode(), \
         open(label_path_rank, "w") as l_fid, \
         open(decod_path_rank, "w") as d_fid:

        if rank == 0:
            logger.info("Decoding begins for %s", data_args.dataset_name)

        for i in range(len(eval_ds)):
            # 只让rank0打log，避免刷屏
            if rank == 0 and (i % 100 == 0):
                logger.info("decoding samples %d (rank0 view)", i)

            idx = eval_ds[i]["idx"]

            input_feature = torch.tensor(eval_ds[i]["input_values"]).reshape(1, -1).to(device)
            if config.instruct:
                prompts = torch.tensor(eval_ds[i]["prompt_ids"]).reshape(1, -1).to(device)
            else:
                prompts = None

            if model_args.ctc_decoding:
                est = model.generate_ctc(
                    inputs=input_feature,
                    prompt_ids=prompts,
                    max_length=150,
                    num_beams=1,
                    synced_gpus=False,
                    use_cache=True,
                )
            else:
                est = model.generate(
                    inputs=input_feature,
                    prompt_ids=prompts,
                    max_length=150,
                    num_beams=1,
                    synced_gpus=False,
                    use_cache=True,
                )

            label_text = tokenizer.decode(torch.tensor(eval_ds[i]["labels"]))
            label_text = skip_special_tokens(label_text)

            if model_args.ctc_decoding:
                label_text = label_text.replace(tokenizer.decode(eval_ds[i]["prompt_ids"][1:-4]), "")

            est_text = tokenizer.decode(est.reshape(-1), skip_special_tokens=False)
            est_text = skip_special_tokens(est_text)

            if rank == 0 and (i % 100 == 0):
                logger.info("label: %s", label_text)
                logger.info("estim: %s", est_text)

            l_fid.write(idx + " " + label_text + "\n")
            d_fid.write(idx + " " + est_text + "\n")

    # 等所有rank写完
    if world_size > 1:
        dist.barrier()

    # rank0 合并各rank文件到最终文件
    final_label_path = os.path.join(training_args.output_dir, f"{_set}_label.text")
    final_decod_path = os.path.join(training_args.output_dir, f"{_set}_decod.text")

    if rank == 0:
        with open(final_label_path, "w") as out_l, open(final_decod_path, "w") as out_d:
            for r in range(world_size):
                lp = os.path.join(training_args.output_dir, f"{_set}_label.rank{r}.text")
                dp = os.path.join(training_args.output_dir, f"{_set}_decod.rank{r}.text")
                with open(lp, "r") as f:
                    out_l.write(f.read())
                with open(dp, "r") as f:
                    out_d.write(f.read())

        logger.info("Merged outputs written to: %s and %s", final_label_path, final_decod_path)

    cleanup_distributed(is_dist)


if __name__ == "__main__":
    main()
