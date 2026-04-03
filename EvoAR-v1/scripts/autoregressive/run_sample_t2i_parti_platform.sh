#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/zhengmingkai/zyr/EvoAR}"
LLAMAGEN_DIR="${LLAMAGEN_DIR:-$PROJECT_ROOT/LlamaGen}"
EVOAR_TRAIN_ENV="${EVOAR_TRAIN_ENV:-/mnt/afs/zhengmingkai/zyr/EvoGen_env}"
PYTHON_BIN="${PYTHON_BIN:-$EVOAR_TRAIN_ENV/bin/python}"

if [ -n "${CONDA_ENV:-}" ]; then
  _CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
  source "$_CONDA_SH"
  conda activate "$CONDA_ENV"
  PYTHON_BIN="$(command -v python)"
  echo "[run_sample_t2i_parti_platform] conda activate: $CONDA_ENV -> PYTHON_BIN=$PYTHON_BIN"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python not found: $PYTHON_BIN" >&2
  exit 1
fi

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_HOME="${MACA_HOME:-$MACA_PATH}"
export CUDA_PATH="${CUDA_PATH:-$MACA_PATH/tools/cu-bridge}"
export CUCC_PATH="${CUCC_PATH:-$CUDA_PATH}"

export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:${LIBRARY_PATH:-}"
export CPATH="$MACA_PATH/include:${CPATH:-}"
export PATH="$MACA_PATH/bin:$MACA_PATH/tools/cu-bridge/bin:${PATH:-}"

if [ -z "${MUSA_VISIBLE_DEVICES:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export MUSA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

cd "$LLAMAGEN_DIR"

echo "[run_sample_t2i_parti_platform] using python: $PYTHON_BIN"
echo "[run_sample_t2i_parti_platform] project dir: $LLAMAGEN_DIR"
echo "[run_sample_t2i_parti_platform] MACA_PATH=$MACA_PATH"
echo "[run_sample_t2i_parti_platform] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[run_sample_t2i_parti_platform] MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-<unset>}"

NUM_GPUS="${SENSECORE_ACCELERATE_DEVICE_COUNT:-${NUM_GPUS:-1}}"
NNODES="${SENSECORE_PYTORCH_NNODES:-${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}}"
NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${MLP_ROLE_INDEX:-${MLP_WORKER_RANK:-${RANK:-0}}}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29505}}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "${ENABLE_MCCL_IB_HCA_HINT:-0}" = "1" ]; then
  export MCCL_IB_HCA="${MCCL_IB_HCA:-mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0}"
fi

if [ "${DIST_DEBUG:-0}" = "1" ]; then
  export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-INFO}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"
  export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-/tmp/nccl.%h.%p.log}"
fi

echo "[run_sample_t2i_parti_platform] dist env:"
echo "  NUM_GPUS=$NUM_GPUS"
echo "  NNODES=$NNODES"
echo "  NODE_RANK=$NODE_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "  MCCL_IB_HCA=${MCCL_IB_HCA:-<unset>}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  VISIBLE_COUNT="${#_visible_devices[@]}"
  if [ "$VISIBLE_COUNT" -gt 0 ]; then
    NUM_GPUS="$VISIBLE_COUNT"
  fi
fi

_DIST_LAUNCHER=("$PYTHON_BIN" -m torch.distributed.run)
_SAMPLE_SCRIPT="$LLAMAGEN_DIR/autoregressive/sample/sample_t2i_ddp.py"
_PROMPT_CSV="${PROMPT_CSV:-$LLAMAGEN_DIR/evaluations/t2i/PartiPrompts.tsv}"
_SAMPLE_DIR="${SAMPLE_DIR:-samples_parti}"
_VQ_CKPT="${VQ_CKPT:-$LLAMAGEN_DIR/pretrained_models/vq_ds16_t2i.pt}"

if [ "$NNODES" -gt 1 ]; then
  exec "${_DIST_LAUNCHER[@]}" \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$_SAMPLE_SCRIPT" \
    --prompt-csv "$_PROMPT_CSV" \
    --sample-dir "$_SAMPLE_DIR" \
    --vq-ckpt "$_VQ_CKPT" \
    "$@"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  exec "${_DIST_LAUNCHER[@]}" \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$_SAMPLE_SCRIPT" \
    --prompt-csv "$_PROMPT_CSV" \
    --sample-dir "$_SAMPLE_DIR" \
    --vq-ckpt "$_VQ_CKPT" \
    "$@"
fi

exec "$PYTHON_BIN" "$_SAMPLE_SCRIPT" --prompt-csv "$_PROMPT_CSV" --sample-dir "$_SAMPLE_DIR" --vq-ckpt "$_VQ_CKPT" "$@"
