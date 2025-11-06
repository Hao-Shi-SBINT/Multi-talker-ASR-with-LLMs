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
epoch=50
corpus=libri2mix_mini
encoder=wavlm
decoder=Llama-3.2-1B
# decoder=Llama-3.2-1B-Instruct

encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=false
train_mode=hybrid # mode can be choose from ctc/hybrid/attention
instruct=false
talker_ctc=true
talker_numbers=2
eval_steps=1600
virtual_env=/lustre/users/shi/toolkits/m_speaker_llm/venv


cache_dir=/lustre/users/shi/.hf_cache

output_dir=exp

partial_encoder_unfreeze=""
# partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_decoder_unfreeze=""
partial_others_unfreeze="down_sampling,separator,serialized_ctc"

#pretrain_model_path=exp_finished/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
pretrain_model_path=""
per_device_train_batch_size=16
per_device_eval_batch_size=16

pmp=exp_finished/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy


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
	instruct=$instruct \
	talker_ctc=${talker_ctc} \
	talker_numbers=$talker_numbers \
	output_dir=${output_dir} \
	cache_dir=${cache_dir} \
	pretrain_model_path=${pretrain_model_path} \
	eval_steps=${eval_steps} \
	virtual_env=${virtual_env}

