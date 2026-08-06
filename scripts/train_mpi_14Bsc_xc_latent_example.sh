#! /usr/bin/env bash
set -euo pipefail

MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
MLP_GPU="${MLP_GPU:-8}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-35448}}"
JOB_NAME="${JOB_NAME:-scail2-deepspeed-example}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mpi_logs}"
MPI_HOSTFILE="${MPI_HOSTFILE:-/root/mpi_hostfile}"
CONFIG="${CONFIG:-configs/training/wan_pose_14Bsc_xc_latent_example.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml}"
SEED="${SEED:-$RANDOM}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_ADDR MASTER_PORT LOCAL_WORLD_SIZE="${MLP_GPU}"

TASK_ID="${MLP_TASK_ID:-local}"
ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
LOG_DIR="${OUTPUT_ROOT}/${TASK_ID}_${JOB_NAME}"
mkdir -p "${LOG_DIR}"

TRAIN_ARGS=(
  train_video.py
  --base "${MODEL_CONFIG}" "${CONFIG}"
  --seed "${SEED}"
)
if [[ -n "${SCAIL2_INIT_CKPT:-}" ]]; then
  TRAIN_ARGS+=(--load "${SCAIL2_INIT_CKPT}")
fi
TRAIN_ARGS+=("$@")

MPI_ARGS=(
  -np "$((MLP_WORKER_NUM * MLP_GPU))"
  --allow-run-as-root
  -oversubscribe
  -map-by "ppr:${MLP_GPU}:node"
  -mca pml ob1
  -mca btl ^openib
  --output-filename "${LOG_DIR}"
  -x NCCL_PXN_DISABLE=0
  -x NCCL_IB_GID_INDEX=3
  -x NCCL_NET_GDR_LEVEL=4
  -x NCCL_IB_RETRY_CNT=7
  -x NCCL_IB_TIMEOUT=25
  -x NCCL_IB_QPS_PER_CONNECTION=8
  -x NCCL_P2P_LEVEL=NVL
  -x NCCL_DEBUG=VERSION
  -x NCCL_IB_TC=106
  -x NCCL_ALGO=TREE
  -x MASTER_ADDR
  -x MASTER_PORT
  -x LOCAL_WORLD_SIZE
  -x CUDA_DEVICE_MAX_CONNECTIONS
  -x PYTORCH_CUDA_ALLOC_CONF
)

if [[ -f "${MPI_HOSTFILE}" ]]; then
  MPI_ARGS+=(--hostfile "${MPI_HOSTFILE}")
fi
if [[ -n "${MLP_SOCKET_IFNAME:-}" ]]; then
  MPI_ARGS+=(
    -x "OMPI_MCA_btl_tcp_if_include=${MLP_SOCKET_IFNAME}"
    -x "GLOO_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}"
    -x "NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}"
  )
fi
if [[ -n "${http_proxy:-}" ]]; then
  MPI_ARGS+=(-x http_proxy)
fi
if [[ -n "${https_proxy:-}" ]]; then
  MPI_ARGS+=(-x https_proxy)
fi

printf 'Launching MPI training: mpirun %q ' "${MPI_ARGS[@]}"
printf 'python %q ' "${TRAIN_ARGS[@]}"
printf '\n'

mpirun "${MPI_ARGS[@]}" python "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/${ROLE_INDEX}.log"
