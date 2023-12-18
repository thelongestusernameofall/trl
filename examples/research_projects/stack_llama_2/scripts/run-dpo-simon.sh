#!/bin/bash

unset http_proxy https_proxy ftp_proxy

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_name=/home/huangshaomang/research/AI/GPT/LLama/Vicuna/llama-2-zh/chinese-alpaca-2-13b-16k-inf1120-v17
output_dir=/home/huangshaomang/research/AI/GPT/LLama/Vicuna/llama-2-zh/chinese-alpaca-2-13b-16k-inf1120-v17-dpov1
data_path=/data0/research/AI/GPT/RLHF/trl/data/using
#target_modules="q_proj,v_proj,k_proj,out_proj,fc_in,fc_out,wte"
target_modules="q_proj,v_proj,k_proj"

load_in_4bit=0
# deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 8 dpo_llama2_simon.py \
    --model_name_or_path ${model_name} \
    --output_dir ${output_dir} \
    --load_in_4bit ${load_in_4bit} \
    --data_path ${data_path} \
    --target_modules ${target_modules}
