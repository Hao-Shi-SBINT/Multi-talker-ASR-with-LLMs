#!/bin/bash
#SBATCH --job-name=1b-2spk_n
#SBATCH --partition=002-partition-all
#SBATCH --gpus=8
#SBATCH --container-image=/lustre/users/shi/audio_llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

# source /lustre/users/shi/toolkits/m_speaker_llm/Multi-Speaker-ASR-with-LLM/venv/bin/activate
cd /lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs/slurm

# export PATH="/lustre/users/shi/toolkits/m_speaker_llm/Multi-Speaker-ASR-with-LLM/venv/bin:$PATH"

# wavlm-Llama-3.2-1B-Instruct-encoder_freeze-decoder_freeze-adater_decoder-ctc-libri2mix_noisy
stage=3
stop_stage=3
epoch=30
corpus=libri2mix_mini
# corpus=libri3mix_noisy
talker_numbers=2

decoder_cross_attention=true
decoder_cross_attention_type=adapgatetiny
decoder_cross_attention_feature=sep
decoder_cross_attention_dynamic=false
decoder_cross_attention_dynamic_threshold=0.01
decoder_cross_attention_dynamic_ratio=0.6
decoder_cross_attention_dynamic_loss=false

encoder=wavlm
decoder=Llama-3.2-1B
# decoder=Llama-3.2-1B-Instruct
# decoder=Llama-3.2-3B
# decoder=Llama-3.2-3B-Instruct
# decoder=Meta-Llama-3.1-8B
# decoder=Meta-Llama-3.1-8B-Instruct

encoder_freeze=true
decoder_freeze=true
adapter_only_decoder=false
train_mode=attention # mode can be choose from ctc/hybrid/attention

instruct=false
# instruct=true


talker_ctc=true
talker_ctc_refine=false
eval_steps=160
virtual_env=/lustre/users/shi/toolkits/m_speaker_llm/venv
ctc_bridge=false
ctc_bridge_type=gate

r_max=4
lora_alpha=4


cache_dir=/lustre/users/shi/.hf_cache

output_dir=exp

# partial_encoder_unfreeze="adapter"
partial_encoder_unfreeze=""
partial_decoder_unfreeze=""
# partial_others_unfreeze="ctc_extractor_concat,enc_to_dec_proj"
partial_others_unfreeze="q_lora_A,q_lora_B,q_rank_logits,out_lora_A,out_lora_B,out_rank_logits,k_lora_A,k_lora_B,k_rank_logits,v_lora_A,v_lora_B,v_rank_logits"



#pretrain_model_path=exp_finished/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
pretrain_model_path=""
per_device_train_batch_size=16
per_device_eval_batch_size=16

separator_hidden=796

pt_separator="${pretrain_separator_path:-none}"
pmp=exp_crossatt_finished/mode_attention-wavlm-Llama-3.2-1B-encoder_freeze-decoder_freeze-ctc-cross_attention_sep-libri2mix_noisy



precision=fp32

bash ../run.sh \
	stage=${stage} \
	stop_stage=${stop_stage} \
	epoch=$epoch \
       	corpus=$corpus \
	encoder=$encoder \
	decoder=$decoder \
	encoder_freeze=$encoder_freeze \
	decoder_freeze=$decoder_freeze \
	partial_encoder_unfreeze=$partial_encoder_unfreeze \
	partial_decoder_unfreeze=$partial_decoder_unfreeze \
	train_mode=${train_mode} \
	partial_others_unfreeze=$partial_others_unfreeze \
	adapter_only_decoder=$adapter_only_decoder \
	per_device_train_batch_size=$per_device_train_batch_size \
	per_device_eval_batch_size=$per_device_eval_batch_size \
        pretrain_model_path="${pmp}" \
	train_mode=$train_mode \
	ctc_bridge=$ctc_bridge \
	instruct=$instruct \
	talker_ctc=${talker_ctc} \
	talker_ctc_refine=${talker_ctc_refine} \
	talker_numbers=$talker_numbers \
	ctc_bridge_type=$ctc_bridge_type \
        separator_hidden=$separator_hidden \
	decoder_cross_attention=$decoder_cross_attention \
	decoder_cross_attention_type=$decoder_cross_attention_type \
	decoder_cross_attention_feature=$decoder_cross_attention_feature \
	decoder_cross_attention_dynamic=$decoder_cross_attention_dynamic \
	decoder_cross_attention_dynamic_threshold=$decoder_cross_attention_dynamic_threshold \
	decoder_cross_attention_dynamic_ratio=$decoder_cross_attention_dynamic_ratio \
	decoder_cross_attention_dynamic_loss=$decoder_cross_attention_dynamic_loss \
	r_max=${r_max} \
	lora_alpha=${lora_alpha} \
	output_dir=${output_dir} \
	cache_dir=${cache_dir} \
	precision=$precision \
	pretrain_separator_path=${pretrain_model_path} \
	eval_steps=${eval_steps} \
	virtual_env=${virtual_env}

