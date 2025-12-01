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
adapter_only_decoder=true
train_mode=attention
ef=false

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="down_sampling,separator,serialized_ctc,enc_to_dec_proj"

per_device_train_batch_size=12
per_device_eval_batch_size=12

epoch=100

stage=4
stop_stage=4

# -----------------------> three talker condition
tn=3
dec=Llama-3.2-1B
corp=libri3mix_noisy
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" epoch="$epoch" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" epoch="$epoch" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B
corp=libri3mix_clean
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" epoch="$epoch" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" epoch="$epoch" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"



dec=Llama-3.2-3B
corp=libri3mix_noisy
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" epoch="$epoch" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" epoch="$epoch" \
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
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" epoch="$epoch" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-3B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" stage="${stage:-}" stop_stage="${stop_stage:-}" epoch="$epoch" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"



per_device_train_batch_size=8
per_device_eval_batch_size=8


dec=Meta-Llama-3.1-8B
corp=libri3mix_noisy
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",epoch="$epoch" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",epoch="$epoch" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B
corp=libri3mix_clean
ins=false
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",epoch="$epoch" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",epoch="$epoch" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"



