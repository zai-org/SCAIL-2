#! /usr/bin/env bash
set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-35447}"
CONFIG="${CONFIG:-configs/training/wan_pose_14Bsc_xc_latent_fsdp_704p.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml}"
SEED="${SEED:-$RANDOM}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
if [[ -n "${SCAIL2_INIT_CKPT:-}" ]]; then
  EXTRA_ARGS+=(--load "${SCAIL2_INIT_CKPT}")
fi

python -m torch.distributed.run \
  --nnodes "${NNODES}" \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  train_video.py \
  --base "${MODEL_CONFIG}" "${CONFIG}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
