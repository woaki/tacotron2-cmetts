# !/bin/bash

gpu_ids="3"
export CUDA_VISIBLE_DEVICES=$gpu_ids

train_script="train.py"

# log file
log_file="231204_ta.log"

# cmd="nohup torchrun --nproc_per_node=${nproc_per_node} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost ${train_script} > ${log_file} 2>&1 &"
cmd="python ${train_script}"

echo "Running command: ${cmd}"

eval ${cmd}
