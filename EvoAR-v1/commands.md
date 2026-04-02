export MACA_PATH=/opt/maca &&
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH} &&
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin

cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/autoregressive/run_c2i_platform.sh \
  --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
  --cloud-save-path /mnt/afs/zhengmingkai/zyr/EvoAR/results_cloud/c2i_nsga2 \
  --results-dir /mnt/afs/zhengmingkai/zyr/EvoAR/results_local/c2i_nsga2 \
  --dataset imagenet_code \
  --gpt-model GPT-B \
  --gpt-type c2i \
  --image-size 256 \
  --downsample-size 16 \
  --global-batch-size 32 \
  --num-workers 4 \
  --mixed-precision bf16 \
  --no-compile \
  --epochs 1 \
  --max-schedule-groups 256 \
  --schedule-population 8 \
  --schedule-mutation-prob 0.8 \
  --schedule-crossover-prob 0.5 \
  --schedule-shift-radius 4 \
  --schedule-block-max-size 8 \
  --schedule-split-prob 0.15 \
  --schedule-trend-weight 0.25 \
  --schedule-final-loss-weight 0.75 \
  --evolve-every 50 \
  --latency-proxy-mode stepwise_surrogate



cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/autoregressive/run_c2i_platform.sh \
  --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
  --cloud-save-path /mnt/afs/zhengmingkai/zyr/EvoAR/results_cloud/c2i_nsga2 \
  --results-dir /mnt/afs/zhengmingkai/zyr/EvoAR/results_local/c2i_nsga2 \
  --dataset imagenet_code \
  --gpt-model GPT-B \
  --gpt-type c2i \
  --image-size 256 \
  --downsample-size 16 \
  --global-batch-size 64 \
  --num-workers 4 \
  --mixed-precision bf16 \
  --no-compile \
  --epochs 1 \
  --max-schedule-groups 256 \
  --schedule-population 8 \
  --evolve-every 50 \
  --latency-proxy-mode stepwise_surrogate