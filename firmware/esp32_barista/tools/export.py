"""Export the Barista checkpoint to the unified PLE binary, plus a golden logits
reference so the C runtime can be proven correct before it touches hardware.

Header, 56 bytes:
  magic "PLE\0", version, header_bytes, flags, input_vocab, output_vocab,
  then d_model, n_layers, n_heads, ffn_hidden, ple_dim, seq_len, group,
  then rope_theta.

flags bit 0 is TIED_HEAD. Barista does not set it: it reads a BPE vocabulary and
writes its own set of curated words, so the head is a separate tensor appended
after out_norm, which is where llm_load binds it.

Tensors follow in a fixed order the C reader hardcodes, each either int4 codes
packed two per byte followed by fp16 group scales, or raw fp32 for the norms.
The golden logits are the 4-bit model's, so comparing C against them isolates
port correctness from quantization error.

  uv run python firmware/esp32_barista/tools/export.py --checkpoint /path/to/checkpoint.pt
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from model import Config, TinyLM

TOOLS = Path(__file__).resolve().parent
# The word-table generator already validates this pair in full. Reusing it keeps
# one definition of a well-formed vocabulary instead of a weaker copy here.
sys.path.insert(0, str(TOOLS))
from generate_vocab_headers import load as load_word_tables  # noqa: E402
ROOT = TOOLS.parent.parent.parent
OUT = ROOT / "artifacts" / "barista"
DEFAULT_VOCAB = OUT / "vocab.json"
DEFAULT_LAYOUT = OUT / "layout.json"

MAGIC = 0x00454C50  # "PLE\0"
FORMAT_VERSION = 1
HEADER_BYTES = 56
FLAG_TIED_HEAD = 1 << 0
GROUP = 128

# Config keys this format cannot express. Null means nothing was configured, so
# dropping it changes nothing; any other value would be silently ignored, which
# is the case worth failing on.
UNSUPPORTED_CONFIG_KEYS = ("active_vocab_size",)


def quant_pack(w, group=GROUP):
    """Group-wise symmetric int4, ragged (no padding) with fp16 scales.

    Returns (packed_uint8[rows, ceil(cols/2)], scales_fp16[rows, n_groups], dq).
    The last group of each row may be shorter than `group`; the C reader derives
    n_groups=ceil(cols/group) and row_bytes=ceil(cols/2). Scales are rounded to
    fp16 *before* computing dq so the golden matches the C exactly. dq is the
    dequantized fp32 tensor the C reconstructs.
    """
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    rows, cols = x.shape
    n_groups = (cols + group - 1) // group
    q = torch.zeros(rows, cols)
    dq = torch.zeros(rows, cols)
    scales = torch.zeros(rows, n_groups)
    for gi in range(n_groups):
        a, b = gi * group, min((gi + 1) * group, cols)
        seg = x[:, a:b]
        amax = seg.abs().amax(dim=1, keepdim=True)
        sc = torch.where(amax == 0, torch.zeros_like(amax),
                         (amax / 7).clamp_min(2**-24))
        sc = sc.half().float()  # fp16-round without underflowing nonzero groups
        scales[:, gi] = sc.squeeze(1)
        divisor = torch.where(sc == 0, torch.ones_like(sc), sc)
        qi = torch.clamp(torch.round(seg / divisor), -7, 7)
        q[:, a:b] = qi
        dq[:, a:b] = qi * sc
    dq = dq.reshape(out_shape)

    codes = (q.to(torch.int16) + 8).to(torch.uint8).numpy()  # rows x cols, 0..15
    row_bytes = (cols + 1) // 2
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed[:, : lo.shape[1]] = lo
    packed[:, : hi.shape[1]] |= (hi << 4)
    return packed.reshape(-1), scales.numpy().astype(np.float16).reshape(-1), dq


def load_config(raw):
    """Build a Config from a stored cfg dict, refusing anything it cannot honour."""
    raw = dict(raw)
    for key in UNSUPPORTED_CONFIG_KEYS:
        if key in raw:
            value = raw.pop(key)
            if value is not None:
                raise SystemExit(
                    f"checkpoint sets {key}={value!r}, which this format does "
                    f"not express; exporting it would silently ignore the setting"
                )
    return Config(**raw)


def check_vocabulary(cfg, vocab_path, layout_path):
    """Both sides of the model must match the tables the firmware will carry.

    load_word_tables covers the tables' internal consistency. What it cannot
    know is the checkpoint, so the two size relationships are checked here:
    otherwise an export would pair the checkpoint weights with incompatible
    class-to-token tables.
    """
    words, out2in, layout = load_word_tables(vocab_path, layout_path)
    if layout["total"] != cfg.vocab_size:
        raise SystemExit(
            f"layout describes {layout['total']} input tokens but the "
            f"checkpoint embedding has {cfg.vocab_size} rows"
        )
    if len(out2in) != cfg.resolved_out_vocab_size:
        raise SystemExit(
            f"layout maps {len(out2in)} output classes but the head has "
            f"{cfg.resolved_out_vocab_size} rows"
        )
    return words, out2in, layout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint to export; no default, so the artifact "
                         "always records which weights it came from")
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(ck["cfg"])
    if cfg.arm != "ple":
        raise SystemExit(f"checkpoint is arm={cfg.arm}; this exporter writes PLE models")
    if cfg.head_is_tied:
        raise SystemExit(
            "checkpoint has a tied head; this exporter writes the untied "
            "Barista layout"
        )
    out_vocab = cfg.resolved_out_vocab_size
    check_vocabulary(cfg, args.vocab, args.layout)

    model = TinyLM(cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    print(f"input_vocab={cfg.vocab_size}  output_vocab={out_vocab}  untied head")

    # Ordered list: (name, tensor, is_quant). C reads in exactly this order.
    plan = []
    sd = model.state_dict()

    def add(name, quant):
        plan.append((name, sd[name], quant))

    add("tok_emb.weight", True)           # input embedding only, the head is separate
    add("ple_model_proj.weight", True)
    add("ple_proj_norm.weight", False)
    add("ple_table.weight", True)         # sparse per-token table
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        add(p + "attn_norm.weight", False)
        add(p + "attn.qkv.weight", True)
        add(p + "attn.proj.weight", True)
        add(p + "ffn_norm.weight", False)
        add(p + "ffn.gate.weight", True)
        add(p + "ffn.up.weight", True)
        add(p + "ffn.down.weight", True)
        add(p + "ple_gate.weight", True)
        add(p + "ple_proj.weight", True)
        add(p + "ple_norm.weight", False)
    add("out_norm.weight", False)
    add("head.weight", True)              # last: llm_load binds it after out_norm

    # Reconstruct a dequantized state dict to drive the golden forward.
    dq_sd = {k: v.clone() for k, v in sd.items()}
    blobs = []
    for name, t, quant in plan:
        if quant:
            packed, scales, dq = quant_pack(t)
            dq_sd[name] = dq
            blobs.append(("Q", packed, scales))
        else:
            blobs.append(("F", t.contiguous().numpy().astype(np.float32), None))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)   # artifacts/ is gitignored
    path = out_dir / "model.bin"
    with open(path, "wb") as f:
        flags = 0  # untied: no TIED_HEAD
        f.write(struct.pack("<IIII", MAGIC, FORMAT_VERSION, HEADER_BYTES, flags))
        f.write(struct.pack("<II", cfg.vocab_size, out_vocab))
        for v in [cfg.d_model, cfg.n_layers, cfg.n_heads,
                  cfg.ffn_hidden, cfg.ple_dim, cfg.seq_len, GROUP]:
            f.write(struct.pack("<i", v))
        f.write(struct.pack("<f", cfg.rope_theta))
        for entry in blobs:
            if entry[0] == "Q":
                _, packed, scales = entry
                f.write(struct.pack("<i", GROUP))   # per-tensor group
                f.write(packed.tobytes())           # ragged int4 codes
                f.write(scales.tobytes())           # fp16 scales
            else:
                f.write(entry[1].tobytes())
    size = path.stat().st_size
    print(f"wrote {path}  ({size/1e6:.2f} MB)  {len(plan)} tensors")

    metadata = {
        "metadata_schema_version": 1,
        "model_binary_format": "PLE",
        "model_binary_version": FORMAT_VERSION,
        "model_binary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_binary_bytes": size,
        "quantization_group_size": GROUP,
        "input_vocab_size": cfg.vocab_size,
        "output_vocab_size": out_vocab,
        "head_is_tied": False,
        # Identity, not location: a path would carry whatever directory the
        # checkpoint happened to live in, which is neither portable nor ours to
        # publish.
        "source_checkpoint_name": checkpoint.name,
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Golden: load dequantized weights, forward a fixed prompt, dump last-pos
    # logits. These are the 4-bit model's logits, so comparing C against them
    # isolates port correctness from quantization error.
    gold = TinyLM(cfg)
    gold.load_state_dict(dq_sd)
    gold.eval()
    prompt = [1, 500, 1000, 200, 42, 777, 13, 99]  # arbitrary fixed token ids
    with torch.no_grad():
        logits, _ = gold(torch.tensor([prompt]))
    last = logits[0, -1].numpy().astype(np.float32)
    assert last.shape == (out_vocab,), last.shape
    np.savez(out_dir / "golden.npz",
             prompt=np.array(prompt, dtype=np.int32), logits=last)
    # Text form for the C host verifier: plen, prompt ids, then out_vocab logits.
    with open(out_dir / "golden.txt", "w") as gf:
        gf.write(f"{len(prompt)}\n")
        gf.write(" ".join(str(t) for t in prompt) + "\n")
        gf.write("\n".join(f"{v:.6f}" for v in last) + "\n")
    print(f"golden: prompt={prompt}")
    print(f"golden: last-pos top5 output classes = {last.argsort()[-5:][::-1].tolist()}")
    print(f"golden: logit range [{last.min():.3f}, {last.max():.3f}]")
    print(f"checkpoint sha256 {metadata['source_checkpoint_sha256']}")


if __name__ == "__main__":
    main()
