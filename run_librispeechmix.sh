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
    decoder_cross_attention_feature=*) decoder_cross_attention_feature="${arg#*=}" ;;
    decoder_cross_attention_dynamic=*) decoder_cross_attention_dynamic="${arg#*=}" ;;
    decoder_cross_attention_dynamic_threshold=*) decoder_cross_attention_dynamic_threshold="${arg#*=}" ;;
    decoder_cross_attention_dynamic_ratio=*) decoder_cross_attention_dynamic_ratio="${arg#*=}" ;;
    decoder_cross_attention_dynamic_loss=*) decoder_cross_attention_dynamic_loss="${arg#*=}" ;;
    per_device_train_batch_size=*)   per_device_train_batch_size="${arg#*=}" ;;
    per_device_eval_batch_size=*)    per_device_eval_batch_size="${arg#*=}" ;;
    partial_encoder_unfreeze=*)      partial_encoder_unfreeze="${arg#*=}" ;;
    partial_decoder_unfreeze=*)      partial_decoder_unfreeze="${arg#*=}" ;;
    partial_others_unfreeze=*)       partial_others_unfreeze="${arg#*=}" ;;
    seed=*)                          seed="${arg#*=}" ;;
    pretrain_model_path=*)       pretrain_model_path="${arg#*=}" ;;
    pretrain_separator_path=*)        pretrain_separator_path="${arg#*=}" ;;
    precision=*)            precision="${arg#*=}" ;;
    r_max=*)                          r_max="${arg#*=}" ;;
    lora_alpha=*)                     lora_alpha="${arg#*=}" ;;
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
echo "[run] decoder_cross_attention_feature=$decoder_cross_attention_feature"
echo "[run] decoder_cross_attention_dynamic=$decoder_cross_attention_dynamic"
echo "[run] decoder_cross_attention_dynamic_threshold=$decoder_cross_attention_dynamic_threshold"
echo "[run] decoder_cross_attention_dynamic_ratio=$decoder_cross_attention_dynamic_ratio"
echo "[run] decoder_cross_attention_dynamic_loss=$decoder_cross_attention_dynamic_loss"
echo "[run] r_max=$r_max"
echo "[run] lora_alpha=$lora_alpha"

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
   # output_dir="${output_dir}-cross_attention_${decoder_cross_attention_feature}"
   output_dir="${output_dir}-cross_attention_${decoder_cross_attention_type}_${decoder_cross_attention_feature}"
fi

if [ "${decoder_cross_attention_type}" = "adapgatetiny" ]; then
   output_dir="${output_dir}-rmax_$r_max-lora_alpha_$lora_alpha"
fi
if [ "${decoder_cross_attention_dynamic}" = "true" ]; then
   output_dir="${output_dir}-dynamic"
fi
if [ "${decoder_cross_attention_dynamic_loss}" = "true" ]; then
   output_dir="${output_dir}-dy_loss-${decoder_cross_attention_dynamic_ratio}"
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

output_dir=/lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs/exp_sot_finished/wavlm-Llama-3.2-1B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_clean

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    _set="librispeech_validation librispeech_test"

    for subset in $_set; do
        dataset_name=data/${corpus}/${subset}
        "$PY_BIN" -m torch.distributed.launch \
            --nproc_per_node 1 --master_port="${master_port}" inference_asr.py \
            --dataset_name="/lustre/users/shi/corpus/LibriSpeechMix/hf_dataset/${corpus}/${subset}" \
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
	    --decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
	    --decoder_cross_attention_dynamic="${decoder_cross_attention_dynamic}" \
            --decoder_cross_attention_dynamic_threshold="${decoder_cross_attention_dynamic_threshold}" \
            --decoder_cross_attention_dynamic_ratio="${decoder_cross_attention_dynamic_ratio}" \
            --decoder_cross_attention_dynamic_loss="${decoder_cross_attention_dynamic_loss}" \
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

output_dir=/lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs/ckpt_cheking
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


