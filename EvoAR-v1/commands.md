export MACA_PATH=/opt/maca &&
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH} &&
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin



# train with GA
cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/autoregressive/run_c2i_platform.sh \
  --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
  --results-dir /mnt/afs/zhengmingkai/zyr/EvoAR/results_local/c2i_nsga2 \
  --dataset imagenet_code \
  --gpt-model GPT-B \
  --gpt-type c2i \
  --image-size 256 \
  --downsample-size 16 \
  --global-batch-size 256 \
  --num-workers 4 \
  --mixed-precision bf16 \
  --no-compile \
  --epochs 100 \
  --max-schedule-groups 256 \
  --schedule-population 64 \
  --schedule-mutation-prob 0.8 \
  --schedule-crossover-prob 0.5 \
  --schedule-shift-radius 4 \
  --schedule-block-max-size 8 \
  --schedule-split-prob 0.15 \
  --schedule-trend-weight 0.25 \
  --schedule-final-loss-weight 0.75 \
  --evolve-every 500 \
  --ckpt-every 50000 \
  --latency-proxy-mode stepwise_surrogate

# train with fixed schedule
cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/autoregressive/run_c2i_platform.sh \
  --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
  --cloud-save-path /mnt/afs/zhengmingkai/zyr/EvoAR/results_cloud/c2i_fixed_schedule \
  --results-dir /mnt/afs/zhengmingkai/zyr/EvoAR/results_local/c2i_fixed_schedule \
  --dataset imagenet_code \
  --gpt-model GPT-B \
  --gpt-type c2i \
  --image-size 256 \
  --downsample-size 16 \
  --global-batch-size 128 \
  --num-workers 4 \
  --mixed-precision bf16 \
  --no-compile \
  --epochs 1 \
  --gpt-ckpt /path/to/your_checkpoint.pt \
  --fixed-schedule-index 0 \
  --fixed-schedule-ckpt /path/to/your_checkpoint.pt \
  --fixed-schedule-source pareto_front \
  --log-every 100 \
  --ckpt-every 5000

# sample c2i with schedule
cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
PROJECT_ROOT=/mnt/afs/zhengmingkai/zyr/EvoAR \
LLAMAGEN_DIR=/mnt/afs/zhengmingkai/zyr/EvoAR/LlamaGen \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/autoregressive/run_sample_c2i_platform.sh \
  --gpt-ckpt /mnt/afs/zhengmingkai/zyr/EvoAR/results_local/c2i_nsga2/000-GPT-B/checkpoints/0025000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --image-size-eval 256 \
  --cfg-scale 2.0 \
  --num-fid-samples 256 \
  --schedule-index 0

# sample t2i coco with schedule
cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
PROJECT_ROOT=/mnt/afs/zhengmingkai/zyr/EvoAR \
LLAMAGEN_DIR=/mnt/afs/zhengmingkai/zyr/EvoAR/LlamaGen \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/autoregressive/run_sample_t2i_coco_platform.sh \
  --gpt-ckpt /path/to/t2i_checkpoint.pt \
  --gpt-model GPT-XL \
  --image-size 512 \
  --cfg-scale 7.5 \
  --schedule-index 0

# sample t2i parti with schedule
cd /mnt/afs/zhengmingkai/zyr/EvoAR/EvoAR-v1 && \
PROJECT_ROOT=/mnt/afs/zhengmingkai/zyr/EvoAR \
LLAMAGEN_DIR=/mnt/afs/zhengmingkai/zyr/EvoAR/LlamaGen \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/autoregressive/run_sample_t2i_parti_platform.sh \
  --gpt-ckpt /path/to/t2i_checkpoint.pt \
  --gpt-model GPT-XL \
  --image-size 512 \
  --cfg-scale 7.5 \
  --schedule-index 0

# evaluate
cd /mnt/afs/zhengmingkai/zyr/EvoAR/LlamaGen && 
python evaluations/c2i/evaluator.py /mnt/afs/zhengmingkai/zyr/EvoAR/LlamaGen/VIRTUAL_imagenet256_labeled.npz /mnt/afs/zhengmingkai/zyr/EvoAR/LlamaGen/samples/GPT-B-0025000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz