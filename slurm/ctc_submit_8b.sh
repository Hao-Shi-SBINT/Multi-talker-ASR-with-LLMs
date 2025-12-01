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

ctc=true

adapter_only_decoder=false
train_mode=ctc
ef=true

partial_encoder_unfreeze=""
partial_decoder_unfreeze=""
partial_others_unfreeze="separator,serialized_ctc"


per_device_train_batch_size=8
per_device_eval_batch_size=8

stage=5
stop_stage=5


# -----------------------> two talker condition
tn=2

dec=Meta-Llama-3.1-8B
corp=libri2mix_noisy
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
pt_separator=exp_separator/libri2mix_noisy_llama-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
pt_separator=exp_separator/libri2mix_noisy_llama-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}",seed="1220" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B
corp=libri2mix_clean
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_clean
pt_separator=exp_separator/libri2mix_clean_llama-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_clean
pt_separator=exp_separator/libri2mix_clean_llama-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}",seed="1220" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

# -----------------------> three talker condition
tn=3

dec=Meta-Llama-3.1-8B
corp=libri3mix_noisy
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_noisy
pt_separator=exp_separator/libri3mix_noisy_llama-ins-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_noisy
pt_separator=exp_separator/libri3mix_noisy_llama-ins-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B
corp=libri3mix_clean
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_clean
pt_separator=exp_separator/libri3mix_clean_llama-ins-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}",seed="1220" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_clean
pt_separator=exp_separator/libri3mix_clean_llama-ins-1b.pt
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="$dec-$corp-$ins" \
  --export=ALL,decoder="$dec",corpus="$corp",instruct="$ins",talker_ctc="$ctc",talker_numbers="$tn",pretrain_model_path="${pmp:-}",per_device_train_batch_size="$per_device_train_batch_size",per_device_eval_batch_size="$per_device_eval_batch_size",encoder_freeze="${ef:-}",train_mode="${train_mode:-}",adapter_only_decoder="${adapter_only_decoder:-}",stage="${stage:-}",stop_stage="${stop_stage:-}",pretrain_separator_path="${pt_separator}" \
  template.slurm \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"


