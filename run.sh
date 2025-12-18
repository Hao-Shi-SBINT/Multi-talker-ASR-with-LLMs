#!/bin/bash

# source /lustre/users/shi/toolkits/m_speaker_llm/Multi-Speaker-ASR-with-LLM/venv/bin/activate
set -euo pipefail

export TORCH_DISTRIBUTED_DEBUG=DETAIL

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
root_dir="$SCRIPT_DIR"

cd "$root_dir"
echo "[run] Current path: ${root_dir}"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# General configuration
for arg in "$@"; do
  case $arg in
    stage=*)                stage="${arg#*=}" ;;
    stop_stage=*)           stop_stage="${arg#*=}" ;;
    epoch=*)                epoch="${arg#*=}" ;;
    corpus=*)               corpus="${arg#*=}" ;;
    encoder=*)              encoder="${arg#*=}" ;;
    decoder=*)              decoder="${arg#*=}" ;;
    encoder_freeze=*)       encoder_freeze="${arg#*=}" ;;
    decoder_freeze=*)       decoder_freeze="${arg#*=}" ;;
    train_mode=*)           train_mode="${arg#*=}" ;;
    adapter_only_decoder=*) adapter_only_decoder="${arg#*=}" ;;
    instruct=*)             instruct="${arg#*=}" ;;
    talker_ctc=*)           talker_ctc="${arg#*=}" ;;
    talker_ctc_refine=*)    talker_ctc_refine="${arg#*=}" ;;
    talker_numbers=*)       talker_numbers="${arg#*=}" ;;
    separator_hidden=*)     separator_hidden="${arg#*=}" ;;
    output_dir=*)           output_dir="${arg#*=}" ;;
    decoder_cross_attention=*)        decoder_cross_attention="${arg#*=}" ;;
    decoder_cross_attention_type=*)   decoder_cross_attention_type="${arg#*=}" ;;
    per_device_train_batch_size=*)   per_device_train_batch_size="${arg#*=}" ;;
    per_device_eval_batch_size=*)    per_device_eval_batch_size="${arg#*=}" ;;
    partial_encoder_unfreeze=*)      partial_encoder_unfreeze="${arg#*=}" ;;
    partial_decoder_unfreeze=*)      partial_decoder_unfreeze="${arg#*=}" ;;
    partial_others_unfreeze=*)       partial_others_unfreeze="${arg#*=}" ;;
    seed=*)                          seed="${arg#*=}" ;;
    pretrain_model_path=*)       pretrain_model_path="${arg#*=}" ;;
    pretrain_separator_path=*)        pretrain_separator_path="${arg#*=}" ;;
    precision=*)            precision="${arg#*=}" ;;
    eval_steps=*)           eval_steps="${arg#*=}" ;;
    virtual_env=*)          virtual_env="${arg#*=}" ;;
    cache_dir=*)            cache_dir="${arg#*=}" ;;
    ctc_bridge=*)           ctc_bridge="${arg#*=}" ;;
    ctc_bridge_type=*)      ctc_bridge_type="${arg#*=}" ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "[run] stage=$stage"
echo "[run] stop_stage=$stop_stage"
echo "[run] train_mode=$train_mode"
echo "[run] epoch=$epoch"
echo "[run] corpus=$corpus"
echo "[run] encoder=$encoder"
echo "[run] decoder=$decoder"
echo "[run] encoder_freeze=$encoder_freeze"
echo "[run] decoder_freeze=$decoder_freeze"
echo "[run] adapter_only_decoder=$adapter_only_decoder"
echo "[run] instruct=$instruct"
echo "[run] per_device_train_batch_size=$per_device_train_batch_size"
echo "[run] per_device_eval_batch_size=$per_device_eval_batch_size"
echo "[run] talker_ctc=$talker_ctc"
echo "[run] talker_ctc_refine=$talker_ctc_refine"
echo "[run] talker_numbers=$talker_numbers"
echo "[run] separator_hidden=$separator_hidden"
echo "[run] partial_encoder_unfreeze=$partial_encoder_unfreeze"
echo "[run] partial_decoder_unfreeze=$partial_decoder_unfreeze"
echo "[run] partial_others_unfreeze=$partial_others_unfreeze"
echo "[run] eval_steps=$eval_steps"
echo "[run] virtual_env=$virtual_env"
echo "[run] cache_dir=$cache_dir"
seed="${seed-42}"
echo "[run] seed=$seed"
echo "[run] pretrain_separator_path=$pretrain_separator_path"
echo "[run] ctc_bridge=$ctc_bridge"
echo "[run] ctc_bridge_type=$ctc_bridge_type"
echo "[run] decoder_cross_attention=$decoder_cross_attention"
echo "[run] decoder_cross_attention_type=$decoder_cross_attention_type"

# output_dir=${output_dir}/${encoder}-${decoder}
output_dir=${output_dir}/mode_${train_mode}-${encoder}-${decoder}

if [ "${encoder_freeze}" = "true" ]; then
    output_dir="${output_dir}-encoder_freeze"
else
    output_dir="${output_dir}-encoder_unfreeze"
fi
if [ "${decoder_freeze}" = "true" ]; then
    output_dir="${output_dir}-decoder_freeze"
else
    output_dir="${output_dir}-decoder_unfreeze"
fi
if [ "${adapter_only_decoder}" = "true" ]; then
    output_dir="${output_dir}-adater_decoder"
# else
#     output_dir="${output_dir}-adater_encoder_decoder"
fi
if [ "${talker_ctc}" = "true" ]; then
    output_dir="${output_dir}-ctc"
fi
if [ "${talker_ctc_refine}" = "true" ]; then
   output_dir="${output_dir}-refine"
fi
if [ "${decoder_cross_attention}" = "true" ]; then
   output_dir="${output_dir}-cross_attention_${decoder_cross_attention_type}"
fi
if [ "${ctc_bridge}" = "true" ]; then
    output_dir="${output_dir}-${ctc_bridge_type}"
fi
output_dir=${output_dir}-${corpus}

echo "[run] output_dir=$output_dir"

if [[ "$instruct" == "true" ]]; then
  extra_args+=" --instruct"
else
  extra_args+="  "
fi

PRECISION_FLAGS=()
case "$precision" in
  fp16) PRECISION_FLAGS+=(--fp16) ;;
  bf16) PRECISION_FLAGS+=(--bf16) ;;
  fp32) : ;; 
  *) echo "Unknown precision: $precision (use fp32|fp16|bf16)"; exit 1 ;;
esac
echo "[run] precision=$precision"

PY_BIN="$virtual_env/bin/python"
master_port=$(( 29501 + RANDOM % 4900 ))

# 1. Data preparing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    "$PY_BIN" utils/generate_dataset.py \
        --base_data_path /lustre/users/shi/toolkits/espnet/egs2/librimix/sot_asr1/data \
        --number ${talker_numbers} \
	--suffix '' \
	--wav_scp_name wav.scp \
	--output_dir datasets/libri${talker_numbers}mix_noisy
fi

# 2. Create the pre-trained AED from pre-trained speech encoder and LLMs
model_ids=${encoder}-${decoder}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   "$PY_BIN" utils/create_from_pretrained.py \
        --encoder_id microsoft/wavlm-large \
	--decoder_base /lustre/share/downloaded/models/meta-llama \
	--llm_id ${decoder} \
	--save_dir dump/${model_ids} \
	--talker_ctc \
	--talker_numbers ${talker_numbers} \
	--check_generate \
	$extra_args
fi

# 3. Training
: "${pretrain_model_path:=dump/${model_ids}}"
echo "[run] pretrain_model_path=$pretrain_model_path"
NUM_GPUS=$("$PY_BIN" -c 'import torch; print(torch.cuda.device_count())')
echo "Detected $NUM_GPUS GPUs"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    "$PY_BIN" -m torch.distributed.launch \
  	--nproc_per_node=$NUM_GPUS --master_port="${master_port}" finetune_asr.py \
	--dataset_name="datasets/${corpus}" \
	--model_name_or_path=${pretrain_model_path} \
	--pretrain_separator_path=${pretrain_separator_path} \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--adapter_only_decoder=${adapter_only_decoder} \
	--output_dir=${output_dir} \
	--metric_for_best_model="eval_loss" \
	--train_mode=${train_mode} \
	--greater_is_better=false \
	--preprocessing_num_workers="16" \
	--audio_column_name="audio" \
	--text_column_name="text" \
	--overwrite_output_dir false\
	--num_train_epochs=${epoch} \
	--per_device_train_batch_size=${per_device_train_batch_size} \
	--per_device_eval_batch_size=${per_device_eval_batch_size} \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-5" \
	--max_grad_norm="1.0" \
	--warmup_steps="400" \
	--evaluation_strategy="steps" \
	--save_steps="1600" \
	--eval_steps=${eval_steps} \
	--logging_steps="10" \
	--save_total_limit="5" \
	--seed="${seed}" \
	--ctc_bridge="${ctc_bridge}" \
	--ctc_bridge_type="${ctc_bridge_type}" \
	--decoder_cross_attention="${decoder_cross_attention}" \
	--decoder_cross_attention_type="${decoder_cross_attention_type}" \
	--freeze_feature_encoder true \
	--freeze_encoder ${encoder_freeze} \
	--freeze_decoder ${decoder_freeze} \
        --partial_encoder_unfreeze="${partial_encoder_unfreeze}" \
        --partial_decoder_unfreeze="${partial_decoder_unfreeze}" \
        --partial_others_unfreeze="${partial_others_unfreeze}" \
	--gradient_checkpointing \
	"${PRECISION_FLAGS[@]}" \
	--talker_ctc=${talker_ctc} \
	--talker_ctc_refine=${talker_ctc_refine} \
	--talker_numbers=${talker_numbers} \
	--separator_hidden=${separator_hidden} \
	--group_by_length \
	--predict_with_generate \
	--do_train true \
	--do_eval true \
	--cache_dir $cache_dir \
	--do_lower_case

    if [ "${train_mode}" != "ctc" ]; then
	"$PY_BIN" utils/merge_adapter.py ${output_dir}
    fi
    "$PY_BIN" utils/fix_safetensors_metadata.py --output_dir ${output_dir}
    cp ${pretrain_model_path}/generation_config.json ${output_dir}

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    _set="validation test"

    for subset in $_set; do
        dataset_name=data/${corpus}/${subset}
        "$PY_BIN" -m torch.distributed.launch \
            --nproc_per_node 1 --master_port="${master_port}" inference_asr.py \
            --dataset_name="datasets/${corpus}/${subset}" \
            --model_name_or_path="${output_dir}" \
            --train_split_name="train" \
            --eval_split_name="validation" \
            --adapter_only_decoder=${adapter_only_decoder} \
            --output_dir=${output_dir} \
            --metric_for_best_model="eval_loss" \
            --greater_is_better=false \
            --preprocessing_num_workers="16" \
            --audio_column_name="audio" \
            --text_column_name="text" \
            --overwrite_output_dir false\
            --num_train_epochs=${epoch} \
            --per_device_train_batch_size="16" \
            --per_device_eval_batch_size="16" \
            --gradient_accumulation_steps="1" \
            --learning_rate="3e-5" \
            --warmup_steps="400" \
            --evaluation_strategy="steps" \
            --save_steps="1600" \
            --eval_steps=${eval_steps} \
            --logging_steps="10" \
            --save_total_limit="5" \
            --freeze_feature_encoder true \
            --ctc_bridge="${ctc_bridge}" \
            --ctc_bridge_type="${ctc_bridge_type}" \
            --decoder_cross_attention="${decoder_cross_attention}" \
            --decoder_cross_attention_type="${decoder_cross_attention_type}" \
            --freeze_encoder ${encoder_freeze} \
            --freeze_decoder ${decoder_freeze} \
            --partial_encoder_unfreeze="${partial_encoder_unfreeze}" \
            --partial_decoder_unfreeze="${partial_decoder_unfreeze}" \
            --partial_others_unfreeze="${partial_others_unfreeze}" \
            --gradient_checkpointing \
            --fp16 \
            --group_by_length \
            --predict_with_generate \
            --talker_ctc=${talker_ctc} \
            --talker_numbers=${talker_numbers} \
            --separator_hidden=${separator_hidden} \
            --do_train false \
            --do_eval true \
            --do_lower_case

        "$PY_BIN" utils/compute-wer.py \
            --char=1 \
            --v=1 ${output_dir}/${subset}_label.text ${output_dir}/${subset}_decod.text \
            > ${output_dir}/${subset}_decod.wer
    done

    for subset in $_set; do
	echo "${output_dir}_${subset}"
	tail -n 5 ${output_dir}/${subset}_decod.wer
    done

fi

# CTC decoding
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    _set="validation test"

    for subset in $_set; do
        dataset_name=data/${corpus}/${subset}
        "$PY_BIN" -m torch.distributed.launch \
            --nproc_per_node 1 --master_port="${master_port}" inference_asr.py \
            --dataset_name="datasets/${corpus}/${subset}" \
            --model_name_or_path="${output_dir}" \
            --train_split_name="train" \
            --eval_split_name="validation" \
            --adapter_only_decoder=${adapter_only_decoder} \
            --output_dir=${output_dir} \
            --metric_for_best_model="eval_loss" \
            --greater_is_better=false \
            --preprocessing_num_workers="16" \
            --audio_column_name="audio" \
            --text_column_name="text" \
            --overwrite_output_dir false\
            --num_train_epochs=${epoch} \
            --per_device_train_batch_size="16" \
            --per_device_eval_batch_size="16" \
            --gradient_accumulation_steps="1" \
            --learning_rate="3e-5" \
            --warmup_steps="400" \
            --evaluation_strategy="steps" \
            --save_steps="1600" \
            --eval_steps=${eval_steps} \
            --logging_steps="10" \
            --save_total_limit="5" \
            --freeze_feature_encoder true \
            --freeze_encoder ${encoder_freeze} \
            --freeze_decoder ${decoder_freeze} \
            --partial_encoder_unfreeze="${partial_encoder_unfreeze}" \
            --partial_decoder_unfreeze="${partial_decoder_unfreeze}" \
            --partial_others_unfreeze="${partial_others_unfreeze}" \
            --gradient_checkpointing \
            --ctc_decoding=true \
            --fp16 \
            --group_by_length \
            --predict_with_generate \
            --talker_ctc=${talker_ctc} \
            --talker_numbers=${talker_numbers} \
            --separator_hidden=${separator_hidden} \
            --ctc_bridge="${ctc_bridge}" \
            --ctc_bridge_type="${ctc_bridge_type}" \
            --do_train false \
            --do_eval true \
            --do_lower_case

        "$PY_BIN" utils/compute-wer.py \
            --char=1 \
            --v=1 ${output_dir}/${subset}_label.text ${output_dir}/${subset}_decod.text \
            > ${output_dir}/${subset}_decod.wer
    done

    for subset in $_set; do
        echo "${output_dir}_${subset}"
        tail -n 5 ${output_dir}/${subset}_decod.wer
    done

fi

