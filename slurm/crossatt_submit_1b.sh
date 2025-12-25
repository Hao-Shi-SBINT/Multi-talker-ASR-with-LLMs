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
train_mode=attention           # mode can be choose from ctc/hybrid/attention
ef=true
adapter_only_decoder=false

# partial_encoder_unfreeze="adapter"
partial_encoder_unfreeze=""
partial_decoder_unfreeze=""
# partial_others_unfreeze="ctc_extractor_concat,enc_to_dec_proj"
partial_others_unfreeze="cross_att_adap"

decoder_cross_attention=true
decoder_cross_attention_type=tiny
decoder_cross_attention_feature=sep

per_device_train_batch_size=12
per_device_eval_batch_size=12

stage=3
stop_stage=4

# -----------------------> two talker condition
tn=2
corp=libri2mix_noisy

dec=Llama-3.2-1B
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

corp=libri2mix_clean
dec=Llama-3.2-1B
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"
# -----------------------> three talker condition
tn=3

dec=Llama-3.2-1B
corp=libri3mix_noisy
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_noisy
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B
corp=libri3mix_clean
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

dec=Llama-3.2-1B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Llama-3.2-1B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_clean
sbatch \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES// /}"} \
  --job-name="${dec}-${corp}-${ins}" \
  template.slurm \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" pretrain_model_path="${pmp:-}" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" decoder_cross_attention="${decoder_cross_attention}" decoder_cross_attention_type="${decoder_cross_attention_type}" decoder_cross_attention_feature="${decoder_cross_attention_feature}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze"

