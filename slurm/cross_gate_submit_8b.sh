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

ctc=true

adapter_only_decoder=false
train_mode=attention
ef=true

talker_ctc_refine=false

partial_encoder_unfreeze=""
partial_decoder_unfreeze=""
partial_others_unfreeze="cross_att_adap,serilized_refine"

decoder_cross_attention=true
decoder_cross_attention_type=gatetiny
decoder_cross_attention_feature=sep

per_device_train_batch_size=8
per_device_eval_batch_size=8

stage=3
stop_stage=4


# -----------------------> two talker condition
tn=2

dec=Meta-Llama-3.1-8B
corp=libri2mix_noisy
ins=false
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri2mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri2mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B
corp=libri2mix_clean
ins=false
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri2mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri2mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

# -----------------------> three talker condition
tn=3

dec=Meta-Llama-3.1-8B
corp=libri3mix_noisy
ins=false
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri3mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri3mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B
corp=libri3mix_clean
ins=false
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri3mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_crossatt_finished/mode_attention-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri3mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",decoder_cross_attention="${decoder_cross_attention}",decoder_cross_attention_type="${decoder_cross_attention_type}",decoder_cross_attention_feature="${decoder_cross_attention_feature}",talker_ctc_refine="${talker_ctc_refine}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"


