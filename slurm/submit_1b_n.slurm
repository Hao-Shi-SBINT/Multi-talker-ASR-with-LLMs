#!/bin/bash
#SBATCH --job-name=1b-2spk_n
#SBATCH --partition=002-partition-massive
#SBATCH --gpus=8
#SBATCH --container-image=/lustre/container_cache/slam-llm/slam-llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

cd /lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs/slurm

stage=3
stop_stage=3
epoch=40
corpus=libri2mix_mini
encoder=wavlm
decoder=Llama-3.2-1B
encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=false
talker_ctc=true
talker_numbers=2
eval_steps=1600

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
	eval_steps=${eval_steps}

