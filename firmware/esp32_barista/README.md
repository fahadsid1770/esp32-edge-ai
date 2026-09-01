# ESP32-S3 espresso question answering

This sketch runs the 8.9M-parameter Barista model on an ESP32-S3 N16R8. A
question arrives over USB serial, the answer streams back there and, when a
panel is wired, to an OLED as well. The model lives in the custom `model` flash
partition at `0x110000`.

What makes it different from the TinyStories sketch is the vocabulary: Barista
**reads** 8,057 input tokens so it can take varied ASCII questions, and
**writes** only 854 output classes. The head is therefore its own tensor rather
than a view of the embedding, and a sampled index is an output class, not a
token id.

## Build and flash

```bash
scripts/fetch_model.sh barista   # download and verify the artifacts
scripts/deploy.sh barista        # gates, compile, flash model, flash firmware
```

`deploy.sh` is the authoritative procedure. Compilation happens before either
flash, so a build failure cannot leave new weights under old firmware.

## The artifact set

Barista needs four files, against TinyStories' two. All four are frozen with the
trained model and are published together:

| file | why the device needs it |
|---|---|
| `model.bin` | the weights |
| `tokenizer.json` | encodes the question on-device |
| `vocab.json` | turns an output class into its output text |
| `layout.json` | maps an output class to the input token id to feed forward |

`golden.txt` is produced by the exporter and is not distributed; the golden gate
is skipped when it is absent.

## Generated headers

`deploy.sh` rebuilds all three from the artifacts being deployed, every run. They
are gitignored, so a fresh clone has none until the first deploy.

| header | from | contents |
|---|---|---|
| `generated/barista_words.h` | `vocab.json` | the 854 output classes as C strings |
| `generated/barista_out2in.h` | `layout.json` | class to input token id |
| `generated/tokenizer_encoder.h` | `tokenizer.json` | 43,056 B BTK1 encoder asset |

## The feedback path

The head emits a class. Feeding it back in needs the input token id that class
corresponds to, which is what `out2in` holds:

```c
llm_forward(&model, BARISTA_OUT2IN[best], pos++, &scratch);
```

Three boot guards refuse to run rather than answer wrongly if the tables and the
weights disagree: `out_vocab` against `BARISTA_WORD_COUNT`, the encoder's widest
id against the model's input vocabulary, and `max(BARISTA_OUT2IN) + 1` against
the same. Flashing Barista firmware over the TinyStories model produces
`word table mismatch: model 25353 vs table 854` and stops.

## Output

Serial is always on. Type a question, press return. The answer streams a class at
a time and ends with a timing line.

The OLED is optional and is probed once at startup. If nothing answers on the
bus, one line is printed and every later draw becomes a no-op, so an absent or
unplugged panel costs a single probe rather than a failed write per class.

128x64 I2C mono OLED, four wires:

```
GND -> GND    VCC -> 3V3    SCL -> GPIO46    SDA -> GPIO18
```

Set `OLED_CONTROLLER` to match the panel: 1.3" is usually SH1106, 0.96" usually
SSD1306. The panel shows the question, a rule, then the answer building up one
output piece at a time, scrolling once it fills.

## Compile switches

All are `#ifndef`-guarded and set the same way:

```bash
arduino-cli compile --build-property compiler.cpp.extra_flags=-DUSE_DISPLAY=0 ...
```

| switch | default | effect |
|---|---|---|
| `USE_DISPLAY` | 1 | 0 builds serial-only, with no display code at all |
| `BARISTA_DUAL_CORE` | 1 | 0 runs every matvec on one core |
| `BARISTA_PROFILE` | 0 | 1 prints where the time goes, per answer |
| `OLED_CONTROLLER` | `OLED_SH1106` | or `OLED_SSD1306` |
| `OLED_ADDR` | `0x3C` | some panels are `0x3D` |

## Expected boot output

The deployed model has SHA-256:

```text
1359a1cb74de4143d630c2c192990de814cd47255bcdfa9cc135f07ef0a39fc4
```

```text
=== ESP32 BARISTA ===
ask an espresso question; the model writes the answer.
model: Vin=8057 Vout=854 D=128 L=6 H=4 F=384 P=128
scratch in SRAM: 20940 B
norms in SRAM: 20/20 vectors, 10656 B
sram free 288 KB
int8-staged 44 tensors | psram free 5.55 MB
build: magic=00454c50 bytes=4600186 fp=e602146b scratch_sram=1 fallbacks=0
config: profile=0 dual_core_requested=1 dual_core_active=1 display_enabled=1 display_present=1
READY>
```

`build:` identifies the weights: `fp` is FNV-1a over the mapped image, and
`deploy.sh` prints the same value for the file it flashed. The two must agree.

`config:` identifies the build switches, so a measurement can require the
configuration it claims rather than trusting a label.
`benchmark_device.py --expect key=value` checks against this line and refuses on
a mismatch.

## Measured

On this board, eight fixed prompts, two passes per mode.

An output piece is one emitted class. Punctuation is a class, so pieces are not
readable words: over the benchmark set, 253 pieces render as 213 words.

| | |
|---|---:|
| serial only | 60.2 ms/piece |
| with the OLED | 88.8 ms/piece |
| panel redraw | +28.6 ms/piece |
| per forward | 49.6 ms |
| dual core against single | 1.75x |

Measure with `USE_DISPLAY=0`, or the per-piece panel redraw lands in the total.

## Running a step by hand

```bash
PORT=/dev/cu.usbmodemNNNN   # the port deploy.sh printed
arduino-cli monitor -p "$PORT" --config baudrate=115200
```

`arduino-cli monitor` holds the port; nothing else can attach until it is
stopped.

The display layout is checked on the host, with no board and no Arduino:

```bash
c++ -std=c++17 -O2 -Wall -Wextra \
  -I firmware/esp32_barista/host_verify/stub \
  -o /tmp/dv firmware/esp32_barista/host_verify/display_verify.cpp
/tmp/dv
```

The model payload only needs reflashing after a new export. Firmware-only
changes can be uploaded without rewriting the model partition.
