"""Time the flashed board and prove a change did not alter what it says.

Every run records the generated text as well as the timing, and --baseline fails
if a single piece of output moved. A pass is accepted only when every expected
prompt produced an answer, a timing line, and a profile line if profiling was
requested; a partial pass is rejected rather than averaged.

The two models are driven differently. Barista answers questions, so it is given
a fixed prompt set and timed per answer. TinyStories generates once per boot from
a fixed prompt and is timed by the throughput line it prints when it finishes.

Barista's counter increments per emitted output class, and punctuation is a
class, so its unit is an output piece rather than a readable word. Both counts
are recorded.

    uv run python scripts/benchmark_device.py barista --out runs/before.json \\
        --expect dual_core_active=1 --expect display_enabled=0
    uv run python scripts/benchmark_device.py barista --out runs/after.json \\
        --expect dual_core_active=1 --baseline runs/before.json

Needs pyserial, which is an optional extra: uv sync --extra device.
Receipts are written under runs/, which is gitignored.
"""

import argparse
import glob
import json
import re
import statistics
import sys
import time
from pathlib import Path

# Varied on purpose: short and long answers, a decline, a safety route, and both
# taste directions.
BARISTA_PROMPTS = [
    "my espresso tastes sour",
    "my espresso tastes bitter and burnt",
    "the shot gushes out in 8 seconds",
    "almost nothing comes out",
    "theres a burning smell",
    "whats the weather tomorrow",
    "i went finer and now every shot is different",
    "how do i clean my machine",
]

# "[32 pieces, 1937 ms, 16.5 pieces/s]" -- a piece is one emitted output class,
# and punctuation is a class.
BARISTA_TIMING = re.compile(r"\[(\d+) pieces, (\d+) ms, [\d.]+ pieces/s\]")
BARISTA_PROFILE = re.compile(
    r"\[profile (\d+) fwd \| input ([\d.]+)ms .*?\| attn ([\d.]+)ms .*?\| "
    r"ffn ([\d.]+)ms .*?\| ple ([\d.]+)ms .*?\| head ([\d.]+)ms .*?\| "
    r"accounted ([\d.]+)%\]")

TINY_TOTAL = re.compile(r"---\s+(\d+) tokens in ([\d.]+) s\s+---")
TINY_TIMING = re.compile(r"throughput: ([\d.]+) tok/s\s+\(([\d.]+) ms/token\)")
TINY_PROFILE = re.compile(
    r"profile ms/token: input ([\d.]+) \| attn ([\d.]+) \| ffn ([\d.]+) \| "
    r"ple ([\d.]+) \| head ([\d.]+)")

FINGERPRINT = re.compile(r"fp=([0-9a-f]+)")
TINY_GRACE_S = 1.0
CONFIG = re.compile(r"^config: (.+)$", re.M)


def parse_config(text):
    """The board's build switches, or None when the sketch printed none."""
    m = CONFIG.search(text)
    if not m:
        return None
    out = {}
    for pair in m.group(1).split():
        if "=" in pair:
            key, value = pair.split("=", 1)
            out[key] = value
    return out


def parse_barista_answer(text, prompt):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if (not line or line == "READY>" or line == prompt
                or line.startswith("[") or line.startswith("config:")):
            continue
        out.append(line[3:] if line.startswith("A: ") else line)
    return " ".join(out).strip()


def parse_barista(text, prompt):
    m = BARISTA_TIMING.search(text)
    answer = parse_barista_answer(text, prompt)
    row = {
        "prompt": prompt,
        "answer": answer,
        "units": int(m.group(1)) if m else 0,     # emitted output classes
        "words": len(answer.split()),             # readable, punctuation attached
        "ms": int(m.group(2)) if m else 0,
        "complete": text.rstrip().endswith("READY>"),
    }
    p = BARISTA_PROFILE.search(text)
    if p:
        row["profile"] = {
            "forwards": int(p.group(1)), "input_ms": float(p.group(2)),
            "attn_ms": float(p.group(3)), "ffn_ms": float(p.group(4)),
            "ple_ms": float(p.group(5)), "head_ms": float(p.group(6)),
            "accounted_pct": float(p.group(7)),
        }
    return row


def parse_tinystories(text, _prompt=None):
    total, timing = TINY_TOTAL.search(text), TINY_TIMING.search(text)
    body = text.split(">>>", 1)[1] if ">>>" in text else text
    body = TINY_TOTAL.split(body)[0]
    story = " ".join(l.strip() for l in body.splitlines() if l.strip())
    row = {
        "prompt": "<fixed boot prompt>",
        "answer": story,
        "units": int(total.group(1)) if total else 0,
        "words": len(story.split()),
        "ms": round(float(total.group(2)) * 1000) if total else 0,
        "complete": bool(timing),
    }
    if timing:
        row["tok_per_s"] = float(timing.group(1))
        row["ms_per_token_compute"] = float(timing.group(2))
    p = TINY_PROFILE.search(text)
    if p:
        row["profile"] = {
            "input_ms": float(p.group(1)), "attn_ms": float(p.group(2)),
            "ffn_ms": float(p.group(3)), "ple_ms": float(p.group(4)),
            "head_ms": float(p.group(5)),
        }
    return row


def barista_done(buf):
    return buf.rstrip().endswith("READY>")


def make_tinystories_done():
    """Complete once the profile line lands. A build that emits no profile ends
    on the throughput line plus a short grace, rather than waiting out the
    timeout. State is per-read, so a second pass does not inherit the first's.
    """
    state = {}

    def done(buf):
        if TINY_PROFILE.search(buf):
            return True
        if TINY_TIMING.search(buf):
            first = state.setdefault("seen_at", time.time())
            return time.time() - first > TINY_GRACE_S
        return False

    return done


MODELS = {
    "barista": {
        "prompts": BARISTA_PROMPTS,
        "parse": parse_barista,
        "done": lambda: barista_done,
        "unit": "piece",
        "requires_config": True,
        "boot_wait": 8.0,
        "reply_wait": 90.0,
    },
    "tinystories": {
        "prompts": None,
        "parse": parse_tinystories,
        "done": make_tinystories_done,
        "unit": "token",
        "boot_wait": 8.0,
        "reply_wait": 180.0,
    },
}


def open_port(port):
    try:
        import serial
    except ImportError:
        sys.exit("pyserial is required: uv sync --extra device")
    if not port:
        found = sorted(glob.glob("/dev/cu.usbmodem*"))
        if not found:
            sys.exit("no board on /dev/cu.usbmodem*; pass --port")
        port = found[0]
    return serial.Serial(port, 115200, timeout=1), port


def read_until(s, done, timeout):
    """Read until done(buf) or the timeout. done is a predicate, not a suffix:
    TinyStories has no prompt marker to wait for."""
    buf, start = "", time.time()
    while time.time() - start < timeout:
        chunk = s.read(4096).decode("utf-8", "replace")
        if chunk:
            buf += chunk
        else:
            time.sleep(0.05)
        if done and done(buf):
            return buf
    return buf


def collect(kind, port):
    spec = MODELS[kind]
    s, port_used = open_port(port)
    s.setDTR(False)
    time.sleep(0.3)
    s.setDTR(True)
    time.sleep(spec["boot_wait"])

    done = spec["done"]()
    if spec["prompts"] is None:
        text = read_until(s, done, spec["reply_wait"])
        banner_text, rows = text, [spec["parse"](text)]
    else:
        banner_text = read_until(s, done, 25)
        rows = []
        for prompt in spec["prompts"]:
            s.reset_input_buffer()
            s.write((prompt + "\n").encode())
            s.flush()
            rows.append(spec["parse"](
                read_until(s, done, spec["reply_wait"]), prompt))
    s.close()
    banner = [l.strip() for l in banner_text.splitlines() if l.strip()]
    return banner, parse_config(banner_text), rows, port_used


def build_line(banner):
    return next((l for l in banner if l.startswith("build:")), "")


def fingerprint(build):
    m = FINGERPRINT.search(build or "")
    return m.group(1) if m else None


def check_pass(kind, banner, config, rows, expect):
    """Everything that must hold before a pass may be recorded at all."""
    spec = MODELS[kind]
    problems = []
    # A board that reports profiling on must emit profile lines, whether or not
    # the caller asked for it.
    want_profile = bool(config and config.get("profile") == "1")

    build = build_line(banner)
    if not build:
        problems.append("board printed no build: line, so the weights are unidentified")
    elif not fingerprint(build):
        problems.append(f"build line carries no fingerprint: {build}")
    if "fallbacks=" in build and "fallbacks=0" not in build:
        problems.append(f"SRAM allocation fell back to PSRAM: {build}")

    # Barista always publishes its switches; a build that does not is too old to
    # benchmark, because its provenance cannot be established.
    if config is None and (expect or spec.get("requires_config")):
        problems.append("board printed no config: line, so the build switches "
                        "cannot be confirmed")
    if expect and config is not None:
        for key, value in expect.items():
            if key not in config:
                problems.append(f"board reports no config key {key!r}; "
                                f"it reports {sorted(config)}")
            elif config[key] != value:
                problems.append(f"config {key}={config[key]}, expected {value}")

    expected_rows = len(spec["prompts"]) if spec["prompts"] else 1
    if len(rows) != expected_rows:
        problems.append(f"{len(rows)} rows, expected {expected_rows}")
    for r in rows:
        where = r["prompt"][:40]
        if not r["complete"]:
            problems.append(f"never reached the end marker: {where}")
        if not r["units"] or not r["ms"]:
            problems.append(f"no timing line: {where}")
        if not r["answer"]:
            problems.append(f"no generated text: {where}")
        if want_profile and "profile" not in r:
            problems.append(f"profiling requested but no profile line: {where}")
    return problems


def check_passes_agree(passes, what="run"):
    """Passes may only be aggregated when they describe the same thing: same
    weights, same build switches, and the same generated text."""
    problems = []
    fps = {fingerprint(build_line(p.get("banner", []))) or p.get("fingerprint")
           for p in passes}
    fps.discard(None)
    if len(fps) > 1:
        problems.append(f"{what}: passes disagree on the weights fingerprint: "
                        f"{sorted(fps)}")
    configs = {json.dumps(p.get("config"), sort_keys=True) for p in passes}
    if len(configs) > 1:
        problems.append(f"{what}: passes disagree on the build configuration")
    answers = {}
    for p in passes:
        for r in p["rows"]:
            answers.setdefault(r["prompt"], set()).add(r["answer"])
    unstable = sorted(k for k, v in answers.items() if len(v) > 1)
    if unstable:
        problems.append(f"{what}: generated text differs between passes for "
                        f"{len(unstable)} prompt(s): {unstable}")
    return problems


def summarise(rows, unit):
    usable = [r for r in rows if r["units"]]
    if not usable:
        return None
    per_unit = [r["ms"] / r["units"] for r in usable]
    total_ms = sum(r["ms"] for r in usable)
    total_units = sum(r["units"] for r in usable)
    return {
        f"ms_per_{unit}": total_ms / total_units,
        f"{unit}s_per_s": 1000.0 * total_units / total_ms,
        f"median_ms_per_{unit}": statistics.median(per_unit),
        f"total_{unit}s": total_units,
        "total_readable_words": sum(r["words"] for r in usable),
        "total_ms": total_ms,
    }


def aggregate(passes, unit):
    values = [p["summary"][f"ms_per_{unit}"] for p in passes]
    lo, hi = min(values), max(values)
    return {
        f"min_ms_per_{unit}": lo,
        f"median_ms_per_{unit}": statistics.median(values),
        f"max_ms_per_{unit}": hi,
        "spread_pct": 100.0 * (hi - lo) / lo if lo else 0.0,
        "passes": len(values),
    }


def compare(kind, current, baseline_path, unit, allow_different_weights):
    base = json.loads(Path(baseline_path).read_text())
    problems = []
    if base.get("model") != kind:
        problems.append(f"baseline is {base.get('model')}, this run is {kind}")

    b_fp, c_fp = fingerprint(base.get("build", "")), fingerprint(current["build"])
    if not b_fp:
        problems.append(f"baseline {baseline_path} carries no fingerprint; "
                        f"it cannot be compared against")
    elif not c_fp:
        problems.append("this run carries no fingerprint")
    elif b_fp != c_fp:
        msg = (f"different weights: baseline fp={b_fp}, now fp={c_fp}. "
               f"ms/{unit} is comparable only across identical weights.")
        if allow_different_weights:
            print(f"  WARNING: {msg}")
        else:
            problems.append(msg + " Pass --allow-different-weights to override.")

    b_cfg, c_cfg = base.get("config"), current.get("config")
    if b_cfg and c_cfg and b_cfg != c_cfg:
        differing = {k: (b_cfg.get(k), c_cfg.get(k))
                     for k in set(b_cfg) | set(c_cfg) if b_cfg.get(k) != c_cfg.get(k)}
        print(f"  NOTE: build configuration differs from the baseline: {differing}")

    problems += check_passes_agree(base["passes"], f"baseline {baseline_path}")

    # Collect every answer the baseline recorded per prompt rather than keeping
    # the last one, so a differing earlier pass cannot be overwritten.
    base_answers = {}
    for p in base["passes"]:
        for r in p["rows"]:
            base_answers.setdefault(r["prompt"], set()).add(r["answer"])
    changed = sorted({r["prompt"] for p in current["passes"] for r in p["rows"]
                      if base_answers.get(r["prompt"]) != {r["answer"]}})

    b_agg, c_agg = base.get("aggregate"), current["aggregate"]
    if b_agg and c_agg:
        b_ms = b_agg[f"median_ms_per_{unit}"]
        c_ms = c_agg[f"median_ms_per_{unit}"]
        print(f"\n  baseline {b_ms:.2f} ms/{unit}  (spread {b_agg['spread_pct']:.1f}%, "
              f"{b_agg['passes']} pass(es))")
        print(f"  now      {c_ms:.2f} ms/{unit}  (spread {c_agg['spread_pct']:.1f}%, "
              f"{c_agg['passes']} pass(es))")
        print(f"  ratio    {b_ms / c_ms:.3f}x  ({100 * (b_ms - c_ms) / b_ms:+.1f}% time)")

    if changed:
        problems.append(f"generated text changed on {len(changed)} prompt(s): {changed}")
    else:
        n = len({r["prompt"] for p in current["passes"] for r in p["rows"]})
        print(f"  text identical on all {n} prompt(s) across every pass "
              f"(behavioural equivalence on these prompts, not proof of "
              f"bit-identical logits)")
    return problems


def parse_expect(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            sys.exit(f"--expect wants key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--port", default="")
    ap.add_argument("--repeat", type=int, default=1,
                    help="passes to run; every pass is validated and recorded")
    ap.add_argument("--expect", action="append", metavar="KEY=VALUE",
                    help="require this value in the board's config: line, "
                         "repeatable (e.g. --expect dual_core_active=1)")
    ap.add_argument("--allow-different-weights", action="store_true")
    args = ap.parse_args()

    if args.repeat < 1:
        sys.exit(f"--repeat must be at least 1, got {args.repeat}")
    expect = parse_expect(args.expect)
    spec = MODELS[args.model]
    unit = spec["unit"]
    passes, port_used, banner, config = [], "", [], None
    for i in range(args.repeat):
        banner, config, rows, port_used = collect(args.model, args.port)
        problems = check_pass(args.model, banner, config, rows, expect)
        if problems:
            for p in problems:
                print(f"  REFUSING pass {i + 1}: {p}", file=sys.stderr)
            sys.exit(1)
        summary = summarise(rows, unit)
        passes.append({"rows": rows, "summary": summary, "banner": banner,
                       "config": config})
        print(f"  pass {i + 1}/{args.repeat}: "
              f"{summary[f'ms_per_{unit}']:.2f} ms/{unit}")

    problems = check_passes_agree(passes)
    if problems:
        for p in problems:
            print(f"  REFUSING: {p}", file=sys.stderr)
        sys.exit(1)

    # Every pass agreed, so one canonical build and config describe the receipt.
    build = build_line(banner)
    for line in banner:
        if any(k in line for k in ("model:", "build:", "config:", "int8-staged",
                                   "norms in SRAM", "scratch in SRAM", "SRAM full",
                                   "no display", "dual-core")):
            print("  " + line)

    agg = aggregate(passes, unit)
    last = passes[-1]
    print(f"\n{args.label or args.out}")
    for r in last["rows"]:
        print(f"  {r['units']:4} {unit}s {r['ms']:7} ms  "
              f"{r['ms'] / r['units']:7.2f} ms/{unit}  {r['prompt'][:40]}")
    s = last["summary"]
    print(f"  ---- {s[f'total_{unit}s']} {unit}s ({s['total_readable_words']} readable words) "
          f"/ {s['total_ms']} ms = {s[f'ms_per_{unit}']:.2f} ms/{unit}")
    print(f"  ---- across {agg['passes']} pass(es): min {agg[f'min_ms_per_{unit}']:.2f}, "
          f"median {agg[f'median_ms_per_{unit}']:.2f}, "
          f"max {agg[f'max_ms_per_{unit}']:.2f}, spread {agg['spread_pct']:.1f}%")

    profiles = [r["profile"] for p in passes for r in p["rows"] if "profile" in r]
    if profiles:
        keys = sorted(profiles[0])
        print("  profile mean:", "  ".join(
            f"{k}={statistics.mean(pr[k] for pr in profiles):.1f}" for k in keys))

    receipt = {
        "model": args.model, "label": args.label, "port": port_used,
        "build": build, "config": config, "expect": expect,
        "aggregate": agg, "passes": passes,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"\n  receipt: {args.out}")

    if args.baseline:
        problems = compare(args.model, receipt, args.baseline, unit,
                           args.allow_different_weights)
        if problems:
            for p in problems:
                print(f"\n  REFUSING: {p}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
