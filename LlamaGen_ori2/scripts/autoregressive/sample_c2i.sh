# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=29500 \
autoregressive/sample/sample_c2i_ddp.py \
"$@"
