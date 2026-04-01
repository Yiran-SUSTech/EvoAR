cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nnodes=1 \
nproc_per_node=8 \
node_rank=0 \
master_addr=127.0.0.1 \
master_port=29502 \
bash scripts/autoregressive/train_c2i_grouped.sh \
  --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
  --cloud-save-path /mnt/afs/zhengmingkai/zyr/EvoAR/results_cloud/c2i_nsga2 \
  --results-dir /mnt/afs/zhengmingkai/zyr/EvoAR/results_local/c2i_nsga2 \
  --dataset imagenet_code \
  --gpt-model GPT-B \
  --gpt-type c2i \
  --image-size 256 \
  --downsample-size 16 \
  --global-batch-size 256 \
  --num-workers 8 \
  --mixed-precision bf16 \
  --no-compile \
  --epochs 50 \
  --max-schedule-groups 256 \
  --schedule-population 128 \
  --schedule-mutation-prob 0.8 \
  --schedule-crossover-prob 0.5 \
  --schedule-shift-radius 4 \
  --schedule-block-max-size 8 \
  --schedule-split-prob 0.15 \
  --schedule-trend-weight 0.25 \
  --schedule-final-loss-weight 0.75 \
  --evolve-every 300 \
  --latency-proxy-mode stepwise_surrogate