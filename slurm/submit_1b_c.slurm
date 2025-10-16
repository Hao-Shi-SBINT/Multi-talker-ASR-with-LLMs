#!/bin/bash
#SBATCH --job-name=1b-2spk_c
#SBATCH --partition=002-partition-all
#SBATCH --gpus=8
#SBATCH --container-image=/lustre/users/shi/audio_llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

# source /lustre/users/shi/toolkits/m_speaker_llm/Multi-Speaker-ASR-with-LLM/venv/bin/activate
cd /lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs/slurm

# export PATH="/lustre/users/shi/toolkits/m_speaker_llm/Multi-Speaker-ASR-with-LLM/venv/bin:$PATH"


stage=3
stop_stage=3
epoch=40
corpus=libri2mix_clean
encoder=wavlm
decoder=Llama-3.2-1B

encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=false
talker_ctc=true
talker_numbers=2
eval_steps=1600
virtual_env=/lustre/users/shi/toolkits/m_speaker_llm/venv

output_dir=exp

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

bash ../run.sh \
	stage=$stage \
	stop_stage=$stop_stage \
	epoch=$epoch \
       	corpus=$corpus \
	encoder=$encoder \
	decoder=$decoder \
	encoder_freeze=$encoder_freeze \
	decoder_freeze=$decoder_freeze \
	partial_encoder_unfreeze=$partial_encoder_unfreeze \
	partial_decoder_unfreeze=$partial_decoder_unfreeze \
	partial_others_unfreeze=$partial_others_unfreeze \
	adapter_only_decoder=$adapter_only_decoder \
	instruct=$instruct \
	talker_ctc=$talker_ctc \
	talker_numbers=$talker_numbers \
	output_dir=${output_dir} \
	eval_steps=${eval_steps} \
	virtual_env=${virtual_env}

