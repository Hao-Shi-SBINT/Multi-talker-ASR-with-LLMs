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
stage=5
stop_stage=5
epoch=50
corpus=libri3mix_clean
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
train_mode=ctc # mode can be choose from ctc/hybrid/attention

instruct=false
# instruct=true


talker_ctc=true
talker_numbers=3
eval_steps=16
virtual_env=/lustre/users/shi/toolkits/m_speaker_llm/venv


cache_dir=/lustre/users/shi/.hf_cache

output_dir=exp

partial_encoder_unfreeze=""
# partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_decoder_unfreeze=""
partial_others_unfreeze="separator,serialized_ctc"

#pretrain_model_path=exp_finished/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
pretrain_model_path=""
per_device_train_batch_size=16
per_device_eval_batch_size=16

separator_hidden=796

# pmp=exp_finished/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
# pmp=exp_finished/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_noisy
# pmp=exp_finished/wavlm-Llama-3.2-1B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy

pt_separator=exp_separator/libri3mix_noisy_llama-ins-1b.pt
# pt_separator="${pretrain_separator_path:-None}"


precision=fp32

seed=1220

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
        separator_hidden=$separator_hidden \
	output_dir=${output_dir} \
	cache_dir=${cache_dir} \
	precision=$precision \
	seed=${seed} \
	pretrain_model_path=${pretrain_model_path} \
        pretrain_separator_path="${pt_separator}" \
	eval_steps=${eval_steps} \
	virtual_env=${virtual_env}

