#!/bin/bash
# --- Slurm directives ---
#SBATCH --job-name=phasenet-tune-a100
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phasenet_tune_%A_%a.out
#SBATCH --error=logs/phasenet_tune_%A_%a.err
#SBATCH --array=0-5

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

VENV_PATH=/home/skataoka26/ann_final/venv
PROJECT_DIR=/home/skataoka26/ann_final

source ${VENV_PATH}/bin/activate
if [ $? -ne 0 ]; then
  echo "Failed to activate venv. Exiting."
  exit 1
fi
echo "Virtual environment activated."

cd ${PROJECT_DIR} || { echo "Failed to cd to ${PROJECT_DIR}"; exit 1; }
echo "Current directory: $(pwd)"

# Hyperparameters
LEARNING_RATES=(1e-2 1e-3 1e-4)
SIGMAS=(20 30)
NUM_LRS=${#LEARNING_RATES[@]}
NUM_SIGMAS=${#SIGMAS[@]}
LR_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SIGMAS % NUM_LRS))
SIGMA_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SIGMAS))
CURRENT_LR=${LEARNING_RATES[$LR_IDX]}
CURRENT_SIGMA=${SIGMAS[$SIGMA_IDX]}

DATASET_NAME=Iquique
EPOCHS_TRAIN=40
BATCH_SIZE_TRAIN=128
NUM_WORKERS_TRAIN=4
PRETRAINED=false

OUTPUT_DIR_BASE=tuning_runs/phasenet_${DATASET_NAME}
RUN_TAG=lr${CURRENT_LR}_s${CURRENT_SIGMA}_e${EPOCHS_TRAIN}
MODEL_OUTPUT_DIR=${OUTPUT_DIR_BASE}/${RUN_TAG}
MODEL_FILENAME=phasenet_best_${RUN_TAG}.pth

mkdir -p ${MODEL_OUTPUT_DIR} logs

echo "--- Run ${SLURM_ARRAY_TASK_ID}: LR=${CURRENT_LR}, σ=${CURRENT_SIGMA} ---"
echo "Model path: ${MODEL_OUTPUT_DIR}/${MODEL_FILENAME}"

# If model already exists, skip training
if [ -f ${MODEL_OUTPUT_DIR}/${MODEL_FILENAME} ]; then
  echo "Model already trained; skipping training."
else
  echo "--- Starting Training ---"
  TRAIN_CMD="python gpt_cnn.py \
    --dataset_name ${DATASET_NAME} \
    --learning_rate ${CURRENT_LR} \
    --epochs ${EPOCHS_TRAIN} \
    --batch_size ${BATCH_SIZE_TRAIN} \
    --sigma ${CURRENT_SIGMA} \
    --num_workers ${NUM_WORKERS_TRAIN} \
    --output_dir ${MODEL_OUTPUT_DIR} \
    --model_filename ${MODEL_FILENAME} \
    --log_interval 20"
  if [ "$PRETRAINED" = true ]; then
    TRAIN_CMD+=" --use_pretrained_weights --pretrained_model_name stead"
  fi
  echo "${TRAIN_CMD}"
  eval ${TRAIN_CMD}
  if [ $? -ne 0 ]; then
    echo "Training FAILED for ${RUN_TAG}"
    exit 1
  fi
  echo "Training completed."
fi

# Evaluation
echo "--- Starting Evaluation ---"
EVAL_OUTPUT_DIR=${MODEL_OUTPUT_DIR}/evaluation
mkdir -p ${EVAL_OUTPUT_DIR}
EVAL_CMD="python evaluate.py \
    --model_path ${MODEL_OUTPUT_DIR}/${MODEL_FILENAME} \
    --dataset_name ${DATASET_NAME} \
    --sigma ${CURRENT_SIGMA} \
    --batch_size 256 \
    --num_workers ${NUM_WORKERS_TRAIN} \
    --detection_threshold 0.3 \
    --output_dir ${EVAL_OUTPUT_DIR} \
    --num_plot_examples 3"
echo "${EVAL_CMD}"
eval ${EVAL_CMD}
if [ $? -ne 0 ]; then
  echo "Evaluation FAILED for ${RUN_TAG}"
else
  echo "Evaluation completed."
fi

echo "--- Task ${SLURM_ARRAY_TASK_ID} done ---"
