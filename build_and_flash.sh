#!/usr/bin/env bash
# esp32-edge-ai: train, export, build and flash for Ubuntu 24.04 LTS.
# Requires: ESP32-S3 N16R8 (16MB flash, 8MB PSRAM).
#
# Usage:
#   ./build_and_flash.sh               # full run: data prep, train, export, build and flash
#   ./build_and_flash.sh --skip-train   # reuse an existing firmware/model/model.bin, just build + flash
set -euo pipefail

SKIP_TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --skip-train) SKIP_TRAIN=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# =============================================================================
# STEP 0: Setup toolchain and dependencies
# =============================================================================
echo "Setting up toolchain and dependencies..."

#!/bin/bash

if ! command -v arduino-cli &> /dev/null; then 
    echo "Installing arduino-cli..." 
    curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh 
    export PATH="$PATH:$PWD/bin" 
fi
arduino-cli config init || echo "Config already exists, continuing..." 
arduino-cli config set network.connection_timeout 10m
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json 
arduino-cli core update-index 
arduino-cli core install esp32:esp32@3.3.10 
arduino-cli lib install "Adafruit SH110X"


#VENV_activating
VENV_DIR="venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment found. Activating..."
else
  echo "Virtual environment not found. Creating it now..."
  python3 -m venv "$VENV_DIR"
  echo "Virtual environment created. Activating..."
fi

source "$VENV_DIR/bin/activate"

pip install -r requirements.txt
pip install esptool tokenizers==0.23.1

echo "Looking for a connected ESP32-S3..."
CANDIDATE_PORTS=(/dev/ttyACM* /dev/ttyUSB*)

if [ ! -e "${CANDIDATE_PORTS[0]}" ]; then
  echo "No /dev/ttyACM* or /dev/ttyUSB* device found. Is the board plugged in?"
  exit 1
elif [ "${#CANDIDATE_PORTS[@]}" -eq 1 ]; then
  ESP32_PORT="${CANDIDATE_PORTS[0]}"
  echo "Found one port: $ESP32_PORT"
else
  echo "Found multiple candidate ports:"
  select choice in "${CANDIDATE_PORTS[@]}"; do
    if [ -n "$choice" ]; then
      ESP32_PORT="$choice"
      break
    fi
  done
fi
export ESP32_PORT
echo "Using port: $ESP32_PORT"

python3 -m esptool -p "$ESP32_PORT" flash_id | head -n 20



# =============================================================================
# STEP 1: Data preparation
# =============================================================================
if [ "$SKIP_TRAIN" -eq 1 ]; then
  echo "--skip-train set, skipping data preparation..."
else
  echo "Preparing TinyStories data..."
  python3 -m research.tinystories.prepare --vocab 32768
fi

# =============================================================================
# STEP 2: Training
# =============================================================================
if [ "$SKIP_TRAIN" -eq 1 ]; then
  echo "--skip-train set, skipping training..."
else
  echo "Training PLE model..."
  python3 -m research.tinystories.train \
    --arm ple \
    --vocab 32768 \
    --d-model 96 \
    --n-layers 6 \
    --ple-dim 128 \
    --target-core 560000 \
    --batch-size 4 \
    --seq-len 256 \
    --steps 5000 \
    --seed 0 \
    --tag cleandeploy \
    --use-gradient-checkpointing \
    --use-cpu-offload \
    --memory-threshold 0.75 \
    --print-memory-every 100 
fi

# =============================================================================
# STEP 3: Export
# =============================================================================
if [ "$SKIP_TRAIN" -eq 1 ]; then
  if [ ! -f "artifacts/tinystories/model.bin" ]; then
    echo "--skip-train set but model.bin not found, running export..."
    TOKENIZER="data/tinystories/vocab-32768/tokenizer.json"
    python3 -m research.tinystories.export ple-cleandeploy-s0 \
      --tokenizer "$TOKENIZER"
  fi
  echo "--skip-train set, reusing existing artifacts/tinystories/model.bin"
else
  echo "Exporting model to firmware format..."
  TOKENIZER="data/tinystories/vocab-32768/tokenizer.json"
  python3 -m research.tinystories.export ple-cleandeploy-s0 \
    --tokenizer "$TOKENIZER"

  echo "Verifying C implementation against PyTorch golden..."
  cc -O3 -Wall -Wextra -o /tmp/verify runtime/host_verify/verify.c -lm
  /tmp/verify artifacts/tinystories/model.bin artifacts/tinystories/golden.txt
fi

# =============================================================================
# STEP 4: Generate vocab.h
# =============================================================================
echo "Generating vocab.h from tokenizer..."
python3 firmware/esp32_tinystories/tools/generate_vocab.py \
  --tokenizer artifacts/tinystories/tokenizer.json \
  --out firmware/esp32_tinystories/generated/vocab.h

# =============================================================================
# STEP 5: Build firmware
# =============================================================================
echo "Building firmware..."
SKETCH_FILE="firmware/esp32_tinystories/esp32_tinystories.ino"
sed -i.bak -E 's/(#define[[:space:]]+USE_DISPLAY[[:space:]]+)1/\10/' "$SKETCH_FILE"

FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=default,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'

arduino-cli compile \
  -b "$FQBN" \
  --build-path /tmp/esp32-llm-build \
  --build-property "compiler.optimization_flags=-O3" \
  firmware/esp32_tinystories

# =============================================================================
# STEP 6: Detect ESP32 port
# =============================================================================
echo "Looking for ESP32-S3..."
CANDIDATE_PORTS=(/dev/ttyACM* /dev/ttyUSB*)
if [ ! -e "${CANDIDATE_PORTS[0]}" ]; then
  echo "No /dev/ttyACM* or /dev/ttyUSB* found. Is the board plugged in?"
  exit 1
elif [ "${#CANDIDATE_PORTS[@]}" -eq 1 ]; then
  ESP32_PORT="${CANDIDATE_PORTS[0]}"
else
  echo "Found multiple ports:"
  select choice in "${CANDIDATE_PORTS[@]}"; do ESP32_PORT="$choice"; break; done
fi
echo "Using port: $ESP32_PORT"

# =============================================================================
# STEP 7: Flash firmware
# =============================================================================
echo "Flashing firmware to ESP32..."
arduino-cli upload -p "$ESP32_PORT" \
  --fqbn "$FQBN" \
  --input-dir /tmp/esp32-llm-build \
  firmware/esp32_tinystories

# =============================================================================
# STEP 8: Flash model partition
# =============================================================================
echo "Flashing model to flash partition at 0x110000..."
if [ ! -f "artifacts/tinystories/model.bin" ]; then
  echo "Error: model.bin not found at artifacts/tinystories/"
  exit 1
fi
python3 -m esptool -c esp32s3 \
  -p "$ESP32_PORT" -b 921600 \
  write_flash 0x110000 artifacts/tinystories/model.bin

# =============================================================================
# STEP 9: Monitor
# =============================================================================
echo ""
echo "Ready! Press the RESET button on the board, then monitor output:"
echo "    ./bin/arduino-cli monitor -p /dev/ttyACM0 -c baudrate=115200"