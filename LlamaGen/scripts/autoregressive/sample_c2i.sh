# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
"$@"

# Example:
# bash scripts/autoregressive/sample_c2i.sh \
#   --gpt-ckpt /path/to/ckpt.pt \
#   --schedule-index 0
