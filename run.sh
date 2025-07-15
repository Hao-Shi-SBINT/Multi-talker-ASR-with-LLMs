#!/bin/bash

# !!!!!!!!!!!!!!!!!!! Should be care !!!!!!!!!!!!!!!!!!!!!
# We need to enter the corresponding directory after slurm
cd /lustre/users/shi/toolkits/m_speaker_llm/Multi-Speaker-ASR-with-LLM/
source venv/bin/activate
root_dir=/lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs
cd ${root_dir}
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
    adapter_only_decoder=*) adapter_only_decoder="${arg#*=}" ;;
    instruct=*)             instruct="${arg#*=}" ;;
    talker_ctc=*)           talker_ctc="${arg#*=}" ;;
    talker_numbers=*)       talker_numbers="${arg#*=}" ;;
    output_dir=*)       output_dir="${arg#*=}" ;;
    partial_encoder_unfreeze=*)      partial_encoder_unfreeze="${arg#*=}" ;;
    partial_decoder_unfreeze=*)       partial_decoder_unfreeze="${arg#*=}" ;;
    partial_others_unfreeze=*)       partial_others_unfreeze="${arg#*=}" ;;
    eval_steps=*)       eval_steps="${arg#*=}" ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "stage=$stage"
echo "stop_stage=$stop_stage"
echo "epoch=$epoch"
echo "corpus=$corpus"
echo "encoder=$encoder"
echo "decoder=$decoder"
echo "encoder_freeze=$encoder_freeze"
echo "decoder_freeze=$decoder_freeze"
echo "adapter_only_decoder=$adapter_only_decoder"
echo "instruct=$instruct"
echo "talker_ctc=$talker_ctc"
echo "talker_numbers=$talker_numbers"
echo "partial_encoder_unfreeze=$partial_encoder_unfreeze"
echo "partial_decoder_unfreeze=$partial_decoder_unfreeze"
echo "partial_others_unfreeze=$partial_others_unfreeze"
echo "eval_steps=$eval_steps"

output_dir=${output_dir}/${encoder}-${decoder}
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
else
    output_dir="${output_dir}-adater_encoder_decoder"
fi
output_dir=${output_dir}-${corpus}
echo "output_dir=$output_dir"

if [[ "$instruct" == "true" ]]; then
  extra_args+=" --instruct"
fi

# 1. Data preparing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python utils/generate_dataset.py \
        --base_data_path /lustre/users/shi/toolkits/espnet/egs2/librimix/sot_asr1/data \
        --number ${talker_numbers} \
	--suffix _clean \
	--wav_scp_name wav_clean.scp \
	--output_dir datasets/libri2mix_clean
fi

# 2. Create the pre-trained AED from pre-trained speech encoder and LLMs
model_ids=${encoder}-${decoder}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python utils/create_from_pretrained.py \
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
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python -m torch.distributed.launch \
  	--nproc_per_node=$NUM_GPUS finetune_asr.py \
	--dataset_name="datasets/${corpus}" \
	--model_name_or_path="dump/${model_ids}" \
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
	--fp16 \
	--group_by_length \
	--predict_with_generate \
	--do_train true \
	--do_eval true \
	--do_lower_case

    python utils/merge_adapter.py ${output_dir}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    _set="validation test"
    for subset in $_set; do
        dataset_name=data/${corpus}/${subset}
	python -m torch.distributed.launch \
	    --nproc_per_node 1 inference_asr.py \
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
            --fp16 \
            --group_by_length \
            --predict_with_generate \
            --do_train false \
            --do_eval true \
            --do_lower_case

	python utils/compute-wer.py \
	    --char=1 \
	    --v=1 ${output_dir}/${subset}_label.text ${output_dir}/${subset}_decod.text \
	    > ${output_dir}/${subset}_decod.wer
    done

    for subset in $_set; do
	echo "${output_dir}_${subset}"
	tail -n 5 ${output_dir}/${subset}_decod.wer
    done

fi


