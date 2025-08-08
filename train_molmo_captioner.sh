#!/bin/bash

set -ex

SCRIPT_NAME=train_captioner
MIXTURE=pixmo_cap_with_transcripts
NUM_GPUS=8
CURRENT_DATE_TIME=$(date +"%m-%d-%H-%M-%S")
# BEAKER_IMAGE="sanghol/molmo2-torch2.7.1-cuda12.8" #"rohunt/molmo-torch2.7.0-cuda12.8-video-v3" #  "chrisc/molmo-torch2.6.0-cuda12.4"
BEAKER_IMAGE="01K24G5CS5RGPGK3VT043MJ5BX"
PRIORITY="high"
CLUSTER=ai2/jupiter-cirrascale-2
CLUSTER2=ai2/ceres-cirrascale # 
WORKSPACE=ai2/mm-olmo
BUDGET=ai2/oe-mm

LLM=olmoe
VISION_BACKBONE=openai
RUN_NAME="train_molmo_${LLM}_${CURRENT_DATE_TIME}"

DATA_DIR=/weka/oe-training-default/mm-olmo
SAVE_FOLDER=/weka/oe-training-default/jasonr/checkpoints/molmo/${RUN_NAME}

gantry run \
  --allow-dirty \
  --budget ${BUDGET} \
  --workspace ${WORKSPACE} \
  --not-preemptible \
  --name "${RUN_NAME}" \
  --task-name "${RUN_NAME}" \
  --description "${RUN_NAME}" \
  --gpus ${NUM_GPUS} \
  --priority ${PRIORITY} \
  --cluster "${CLUSTER}" \
  --cluster "${CLUSTER2}" \
  --beaker-image "${BEAKER_IMAGE}" \
  --weka oe-training-default:/weka/oe-training-default \
  --shared-memory 128GiB \
  --env HF_DATASETS_OFFLINE=1 \
  --env OLMO_SHARED_FS=1 \
  --env OMP_NUM_THREADS=8 \
  --env-secret WANDB_API_KEY=JASONR_WANDB_API_KEY \
  --env-secret GITHUB_TOKEN=JASONR_GITHUB_TOKEN \
  --env-secret HF_ACCESS_TOKEN=JASONR_HF_ACCESS_TOKEN \
  --env-secret OPENAI_API_KEY=JASONR_OPENAI_API_KEY \
  --no-python \
  --venv base \
  -- /bin/bash -c "WANDB_ENTITY=prior-ai2 WANDB_PROJECT=jasonr_exps DATA_DIR=${DATA_DIR} MOLMO_DATA_DIR=${DATA_DIR} \
    torchrun -m --nproc-per-node ${NUM_GPUS} \
    launch_scripts.${SCRIPT_NAME} ${LLM} \
    --dataset ${MIXTURE} \
    --vision_backbone ${VISION_BACKBONE} \
    --save_folder=${SAVE_FOLDER} \
    --run_name=${RUN_NAME}"
