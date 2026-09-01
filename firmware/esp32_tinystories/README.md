# ESP32-S3 on-chip inference

This sketch runs the 28.9M-parameter PLE TinyLM on an ESP32-S3 N16R8. The model
lives in the custom `model` flash partition at `0x110000`; the tied
embedding/output head is staged in PSRAM at boot.

## Build and flash

```bash
scripts/fetch_model.sh tinystories   # download and verify the artifacts
scripts/deploy.sh tinystories        # gates, compile, flash model, flash firmware
```

`deploy.sh` is the authoritative procedure. It regenerates the decode header from
the tokenizer being deployed, runs both host gates, compiles at global `-O3`,
writes the model partition and then the firmware, and prints the fingerprint the
board should report back. Compilation happens before either flash, so a build
failure cannot leave new weights under old firmware.

The model argument is mandatory: the board holds one model at a time, and
deploying replaces whichever one is installed.

## Running a step by hand

These are the pieces worth running on their own. For flashing, use the script:
regenerating the header and writing the two images in order are part of what
makes the result correct.

Host gates:

```bash
cc -O3 -Wall -Wextra -o /tmp/verify runtime/host_verify/verify.c -lm
/tmp/verify artifacts/tinystories/model.bin artifacts/tinystories/golden.txt

cc -O3 -Wall -Wextra -DLLM_INT8_ACT=1 -o /tmp/staging runtime/host_verify/staging_verify.c -lm
/tmp/staging artifacts/tinystories/model.bin
```

The first checks the exact int4 path against the PyTorch golden; the second the
staged int8 kernel and platform hooks, which is what the device runs. Every host
tool takes explicit paths; none assumes a model location.

Rebuild the decode header from a tokenizer:

```bash
uv run python firmware/esp32_tinystories/tools/generate_vocab.py \
  --tokenizer artifacts/tinystories/tokenizer.json \
  --out firmware/esp32_tinystories/generated/vocab.h
```

Regenerate the model artifacts from a checkpoint:

```bash
TAG=ple-cleandeploy-s0                              # a checkpoint under runs/
TOKENIZER=data/tinystories/vocab-32768/tokenizer.json   # the one it trained on
uv run python -m research.tinystories.export "$TAG" --tokenizer "$TOKENIZER"
```

Watch the board:

```bash
PORT=/dev/cu.usbmodemNNNN   # the port deploy.sh printed
arduino-cli monitor -p "$PORT" --config baudrate=115200
```

The model payload only needs reflashing after a new export. Firmware-only changes
can be uploaded without rewriting the model partition.

## Expected boot output

The deployed model has SHA-256:

```text
1d8326c05c383ccfa615f5455575802817cb453dbc7ab28875d41a9dbb45477e
```

```text
=== ESP32-S3 PLE TinyLM ===
model: Vin=32768 Vout=25353 D=96 L=6 H=4 F=66 P=128  (mapped 15.6 MB)
norms  -> SRAM   20 vectors
hot set-> SRAM   21128 B dynamic + 8192 B static = 29320 B managed
weights-> PSRAM  44 tensors int8, 4.19 MB allocated
build: bytes=14912348 fp=a9bdd778 sram=29320B psram=4.19MB
free: sram 294 KB | psram 3.74 MB
```

`fp` is FNV-1a over the mapped image; `deploy.sh` prints the same value for the
file it flashed and the two must agree. A `FATAL:` line means an allocation
missed its intended tier; initialization stops rather than run elsewhere.

## Measured

On this board, 200 tokens:

| | |
|---|---:|
| compute | 94.9 ms/token |
| attached serial | 9.88 tok/s |
| output head | 59.4 ms/token |
| attention | 20.5 |
| PLE path | 6.4 |
| FFN | 6.4 |
| input | 2.2 |

The head is staged int8 and split across both LX7 cores, and is
PSRAM-bandwidth-bound. int8 activations cost +0.0003 nats of validation CE over
32,768 predictions (2.4793 -> 2.4796, ppl 11.93 / 11.94). The fp32 host golden
matches PyTorch to 1e-5.
