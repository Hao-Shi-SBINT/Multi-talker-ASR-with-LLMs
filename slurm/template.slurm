#!/bin/bash
#SBATCH --partition=002-partition-all
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --container-image=/lustre/users/shi/audio_llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre

set -euo pipefail
cd /lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs/slurm

########################################
# 1) Parse script args: key=value after template.slurm
########################################
for arg in "$@"; do
  case "$arg" in
    stage=*)                          stage="${arg#*=}" ;;
    stop_stage=*)                     stop_stage="${arg#*=}" ;;
    epoch=*)                          epoch="${arg#*=}" ;;
    encoder=*)                        encoder="${arg#*=}" ;;
    decoder=*)                        decoder="${arg#*=}" ;;
    corpus=*)                         corpus="${arg#*=}" ;;
    instruct=*)                       instruct="${arg#*=}" ;;
    talker_ctc=*)                     talker_ctc="${arg#*=}" ;;
    talker_numbers=*)                 talker_numbers="${arg#*=}" ;;
    train_mode=*)                     train_mode="${arg#*=}" ;;
    encoder_freeze=*)                 encoder_freeze="${arg#*=}" ;;
    decoder_freeze=*)                 decoder_freeze="${arg#*=}" ;;
    adapter_only_decoder=*)           adapter_only_decoder="${arg#*=}" ;;
    per_device_train_batch_size=*)    per_device_train_batch_size="${arg#*=}" ;;
    per_device_eval_batch_size=*)     per_device_eval_batch_size="${arg#*=}" ;;
    partial_encoder_unfreeze=*)       partial_encoder_unfreeze="${arg#*=}" ;;
    partial_decoder_unfreeze=*)       partial_decoder_unfreeze="${arg#*=}" ;;
    partial_others_unfreeze=*)        partial_others_unfreeze="${arg#*=}" ;;
    pretrain_model_path=*)            pretrain_model_path="${arg#*=}" ;;
    output_dir=*)                     output_dir="${arg#*=}" ;;
    eval_steps=*)                     eval_steps="${arg#*=}" ;;
    virtual_env=*)                    virtual_env="${arg#*=}" ;;
    cache_dir=*)                      cache_dir="${arg#*=}" ;;
    # ignore unknown keys instead of failing:
    # *) echo "[WARN] Unknown arg: $arg" >&2 ;;
  esac
done

########################################
# 2) Defaults — use single '-' (respect empty strings)
########################################
# Fixed/common parameters
stage="${stage-3}"
stop_stage="${stop_stage-3}"
epoch="${epoch-50}"
encoder="${encoder-wavlm}"
eval_steps="${eval_steps-1600}"
virtual_env="${virtual_env-/lustre/users/shi/toolkits/m_speaker_llm/venv}"
cache_dir="${cache_dir-/lustre/users/shi/.hf_cache}"

# Sweep parameters
decoder="${decoder-Llama-3.2-1B}"
corpus="${corpus-libri3mix_clean}"
instruct="${instruct-false}"
talker_ctc="${talker_ctc-true}"
talker_numbers="${talker_numbers-3}"

# Other toggles
train_mode="${train_mode-hybrid}"
encoder_freeze="${encoder_freeze-false}"
decoder_freeze="${decoder_freeze-true}"
adapter_only_decoder="${adapter_only_decoder-true}"

# Batch sizes
per_device_train_batch_size="${per_device_train_batch_size-16}"
per_device_eval_batch_size="${per_device_eval_batch_size-16}"

# IMPORTANT: allow empty string to pass through
partial_encoder_unfreeze="${partial_encoder_unfreeze-}"                   # keep "" if provided
partial_decoder_unfreeze="${partial_decoder_unfreeze-}"                   # keep "" if provided
partial_others_unfreeze="${partial_others_unfreeze-enc_to_dec_proj}"      # only if UNSET, not empty
pretrain_model_path="${pretrain_model_path-}"                             # keep "" if provided

# (Optional) quick debug
echo "[INFO] decoder=${decoder}"
echo "[INFO] corpus=${corpus}"
echo "[INFO] train_mode=${train_mode}"
echo "[INFO] instruct=${instruct}  talker_ctc=${talker_ctc}  talker_numbers=${talker_numbers}"
echo "[INFO] partial_encoder_unfreeze='${partial_encoder_unfreeze}'"
echo "[INFO] partial_decoder_unfreeze='${partial_decoder_unfreeze}'"
echo "[INFO] partial_others_unfreeze='${partial_others_unfreeze}'"
echo "[INFO] pretrain_model_path='${pretrain_model_path}'"

########################################
# 3) Output directory tag/naming (optional)
########################################
tag="dec=${decoder}_corpus=${corpus}_ins=${instruct}_ctc=${talker_ctc}_tn=${talker_numbers}"
output_dir="${output_dir-exp}"   # root; run.sh can append tag/time if needed

echo "[INFO] combo: ${tag}"
echo "[INFO] output_dir: ${output_dir}"

bash ../run.sh \
  stage=$stage \
  stop_stage=$stop_stage \
  epoch=$epoch \
  corpus=$corpus \
  encoder=$encoder \
  decoder=$decoder \
  encoder_freeze=$encoder_freeze \
  decoder_freeze=$decoder_freeze \
  per_device_train_batch_size=$per_device_train_batch_size \
  per_device_eval_batch_size=$per_device_eval_batch_size \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  pretrain_model_path="${pretrain_model_path}" \
  adapter_only_decoder=$adapter_only_decoder \
  train_mode=$train_mode \
  instruct=$instruct \
  talker_ctc=$talker_ctc \
  cache_dir=$cache_dir \
  talker_numbers=$talker_numbers \
  output_dir="$output_dir" \
  eval_steps=$eval_steps \
  virtual_env=$virtual_env

