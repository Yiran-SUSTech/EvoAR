# !/bin/bash
set -x

export nnodes=${nnodes:-1}              # 节点数，单机=1
export nproc_per_node=${nproc_per_node:-1}  # 每节点GPU数
export node_rank=${node_rank:-0}        # 当前节点rank，单机=0
export master_addr=${master_addr:-127.0.0.1}  # master地址，单机=localhost
export master_port=${master_port:-12355}      # master端口
export PYTHONPATH=/mnt/afs/zhengmingkai/raozf/zyr/LlamaGen_ori2/:$PYTHONPATH


torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py "$@"