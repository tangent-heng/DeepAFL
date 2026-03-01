#!/bin/bash

# ======= Modify parameters here =======
# Basic Setup
DATASET="cifar100"
BACKBONE="resnet18"
BATCH_SIZE=512
WORKERS=0
GPU=0
PRETRAINED=true
MODEL_DIR="./weights"

# Data Configuration
DATA_DIR="./data"
DOWNLOAD=true
SEED=1



# Federated Learning Configuration
NUM_CLIENTS=10
PARTITION="dir"
ALPHA=0.1
MIN_SAMPLES=10


# Model Configuration
FE=1024
N_LAYERS=20
HIDDEN_DIM=1024
REG=1
REG_SAND=0.05
ACTIVATION="gelu"


# ======= End of parameters =======

# Create log directory structure
LOG_BASE_DIR="logs"
mkdir -p "${LOG_BASE_DIR}/${DATASET}/${PARTITION}/alpha${ALPHA}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_BASE_DIR}/${DATASET}/${PARTITION}/alpha${ALPHA}/${NUM_CLIENTS}_${TIMESTAMP}.log"

if [ "$DATASET" == "cifar100" ]; then
  NUM_CLASSES=100
elif [ "$DATASET" == "imagenet-r" ]; then
  NUM_CLASSES=200
elif [ "$DATASET" == "tinyimagenet" ]; then
  NUM_CLASSES=200
elif [ "$DATASET" == "cifar10" ]; then
  NUM_CLASSES=10
else
  echo "Warning: Unknown dataset, using default class count of 100"
  NUM_CLASSES=100
fi


# Convert boolean values to command line arguments
PRETRAINED_ARG=""
if [ "$PRETRAINED" = true ]; then
  PRETRAINED_ARG="--pretrained"
fi

DOWNLOAD_ARG=""
if [ "$DOWNLOAD" = true ]; then
  DOWNLOAD_ARG="--download"
fi


# Build complete command
CMD="python -u main.py \
  --dataset $DATASET \
  --backbone $BACKBONE \
  --workers $WORKERS \
  --batch-size $BATCH_SIZE \
  --gpu $GPU \
  --data_dir $DATA_DIR \
  --seed $SEED \
  --num_classes $NUM_CLASSES \
  --model_dir $MODEL_DIR \
  --fe $FE \
  --n_layers $N_LAYERS \
  --hidden_dim $HIDDEN_DIM \
  --reg $REG \
  --reg_sand $REG_SAND \
  --partition $PARTITION \
  --alpha $ALPHA \
  --min_samples $MIN_SAMPLES \
  --num_clients $NUM_CLIENTS \
  --activation $ACTIVATION \
  $PRETRAINED_ARG $DOWNLOAD_ARG"

# Display configuration
echo "====== Experiment Configuration ======" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET (Classes: $NUM_CLASSES)" | tee -a "$LOG_FILE"
echo "Model: $BACKBONE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=====================================" | tee -a "$LOG_FILE"

# Execute command
echo "Executing: $CMD" | tee -a "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"

echo "Experiment completed. Log saved to: $LOG_FILE"