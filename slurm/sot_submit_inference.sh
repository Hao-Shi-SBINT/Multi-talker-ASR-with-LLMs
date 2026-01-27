EXCLUDE_FILE="/lustre/teams/mmai/mmai-job-setting/exclude_nodes.txt"
EXCLUDE_NODES=""

if [[ -r "$EXCLUDE_FILE" ]]; then
  # 解析出逗号分隔的 node list（可能为空）
  EXCLUDE_NODES="$(
    awk '
      /^[[:space:]]*#/ {next}   # skip comment
      NF==0 {next}              # skip blank
      {
        gsub(/[[:space:]]+/, "", $0)
        if ($0 == "") next
        if (c++) printf ","
        printf "%s", $0
      }
    ' "$EXCLUDE_FILE"
  )"

  # 如果为空 -> 仍然保持空字符串；不为空 -> 变成完整参数字符串
  [[ -n "$EXCLUDE_NODES" ]] && EXCLUDE_NODES="--exclude=$EXCLUDE_NODES"
else
  echo "[WARN] exclude file not readable: $EXCLUDE_FILE (skip --exclude)" >&2
fi
EXCLUDE_NODES="${EXCLUDE_NODES#--exclude=}"


ctc=false
train_mode=attention # mode can be choose from ctc/hybrid/attention
ef=false
adapter_only_decoder=true

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

per_device_train_batch_size=12
per_device_eval_batch_size=12

stage=6
stop_stage=6

# -----------------------> two talker condition
tn=2

corp=libri2mix_noisy
dec=Llama-3.2-1B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

!<<COMMENT
corp=libri2mix_clean
dec=Llama-3.2-1B-Instruct
ins=true
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"
COMMENT


