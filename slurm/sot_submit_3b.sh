EXCLUDE_FILE="/lustre/teams/mmai/mmai-job-setting/exclude_nodes.txt"
EXCLUDE_NODES=""

if [[ -r "$EXCLUDE_FILE" ]]; then
  EXCLUDE_NODES="$(
    grep -Ev '^[[:space:]]*(#|$)' "$EXCLUDE_FILE" |   # 去掉注释行与空行
    tr -d ' ' |                                       # 去掉可能的空格
    paste -sd, -                                      # 用逗号拼接成一行
  )"
else
  echo "[WARN] exclude file not readable: $EXCLUDE_FILE (skip --exclude)"
fi



ctc=false
train_mode=attention # mode can be choose from ctc/hybrid/attention
ef=false
adapter_only_decoder=true

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"


per_device_train_batch_size=12
per_device_eval_batch_size=12

stage=4
stop_stage=4

# -----------------------> two talker condition
tn=2

dec=Llama-3.2-3B
corp=libri2mix_noisy
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B
corp=libri2mix_clean
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

# -----------------------> three talker condition
tn=3

dec=Llama-3.2-3B
corp=libri3mix_noisy
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B
corp=libri3mix_clean
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"
