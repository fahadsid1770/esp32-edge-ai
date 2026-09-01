"""scripts/benchmark_device.py: parsing, pass validation, and comparison rules.

No board and no pyserial. The fixtures are verbatim serial captures from this
project's two sketches, so a change to a sketch's output format fails here rather
than producing a receipt full of zeros.

  uv run python -m unittest discover -s tests
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_device.py"
_spec = importlib.util.spec_from_file_location("benchmark_device", SCRIPT)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

# Captured from the board, BARISTA_PROFILE=1 and USE_DISPLAY=0.
BARISTA_PROFILED = """
=== ESP32 BARISTA ===
ask an espresso question; the model writes the answer.
model: Vin=8057 Vout=854 D=128 L=6 H=4 F=384 P=128
scratch in SRAM: 20940 B
norms in SRAM: 20/20 vectors, 10704 B
sram free 292 KB
int8-staged 44 tensors | psram free 5.55 MB
build: magic=00454c50 bytes=4600186 fp=e602146b scratch_sram=1 fallbacks=0
config: profile=1 dual_core_requested=1 dual_core_active=1 display_enabled=0 display_present=0
READY>
A: if the cup reads bitter, coarsen a step. if it reads sour, go finer instead. which of those does your drink read as?
[28 pieces, 1623 ms, 17.2 pieces/s]
[profile 33 fwd | input 87ms 5% | attn 423ms 26% | ffn 826ms 51% | ple 198ms 12% | head 86ms 5% | accounted 100%]
READY>
"""

# The same sketch built without profiling, which is the default.
BARISTA_PLAIN = """
A: if the flow is quicker than your usual pull, the setting is likely too coarse. confirm the basket and that the grinder is adjustable, then move one step.
[32 pieces, 1937 ms, 16.5 pieces/s]
READY>
"""

BARISTA_REFUSAL = """
(ascii only)
READY>
"""

# Captured from the board running esp32_tinystories.
TINYSTORIES = """
=== ESP32-S3 PLE TinyLM ===
model: Vin=32768 Vout=25353 D=96 L=6 H=4 F=66 P=128  (mapped 15.6 MB)
norms  -> SRAM   20 vectors
hot set-> SRAM   21128 B dynamic + 8192 B static = 29320 B managed
weights-> PSRAM  44 tensors int8, 4.19 MB allocated
build: bytes=14912348 fp=a9bdd778 sram=29320B psram=4.19MB
free: sram 294 KB | psram 3.74 MB

>>> Once upon a time, there was a little girl named Lily. She loved to play
outside in the sunshine.

--- 200 tokens in 20.24 s ---
throughput: 9.88 tok/s   (94.9 ms/token)
profile ms/token: input 2.2 | attn 20.5 | ffn 6.4 | ple 6.4 | head 59.4
"""


class TestBaristaParsing(unittest.TestCase):
    def test_timing_and_answer(self):
        row = bench.parse_barista(BARISTA_PLAIN, "is my puck too wet?")
        self.assertEqual(row["units"], 32)
        self.assertEqual(row["ms"], 1937)
        self.assertTrue(row["answer"].startswith("if the flow is quicker"))
        self.assertTrue(row["answer"].endswith("move one step."))

    def test_answer_excludes_markers(self):
        row = bench.parse_barista(BARISTA_PLAIN, "is my puck too wet?")
        for marker in ("READY>", "[32 pieces", "A: "):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, row["answer"])

    def test_profile_is_parsed_when_present(self):
        row = bench.parse_barista(BARISTA_PROFILED, "my espresso tastes sour")
        p = row["profile"]
        self.assertEqual(p["forwards"], 33)
        self.assertEqual(p["input_ms"], 87.0)
        self.assertEqual(p["attn_ms"], 423.0)
        self.assertEqual(p["ffn_ms"], 826.0)
        self.assertEqual(p["ple_ms"], 198.0)
        self.assertEqual(p["head_ms"], 86.0)
        self.assertEqual(p["accounted_pct"], 100.0)

    def test_profile_components_sum_to_the_reported_total(self):
        row = bench.parse_barista(BARISTA_PROFILED, "my espresso tastes sour")
        p = row["profile"]
        parts = sum(p[k] for k in ("input_ms", "attn_ms", "ffn_ms", "ple_ms", "head_ms"))
        self.assertAlmostEqual(parts, row["ms"], delta=0.01 * row["ms"])

    def test_no_profile_key_without_a_profile_line(self):
        self.assertNotIn("profile", bench.parse_barista(BARISTA_PLAIN, "x"))

    def test_a_refusal_yields_no_timing(self):
        row = bench.parse_barista(BARISTA_REFUSAL, "why is my café bitter")
        self.assertEqual(row["units"], 0)
        self.assertEqual(row["ms"], 0)


class TestTinyStoriesParsing(unittest.TestCase):
    def test_tokens_and_duration(self):
        row = bench.parse_tinystories(TINYSTORIES)
        self.assertEqual(row["units"], 200)
        self.assertEqual(row["ms"], 20240)

    def test_throughput_line(self):
        row = bench.parse_tinystories(TINYSTORIES)
        self.assertEqual(row["tok_per_s"], 9.88)
        self.assertEqual(row["ms_per_token_compute"], 94.9)

    def test_profile_line(self):
        p = bench.parse_tinystories(TINYSTORIES)["profile"]
        self.assertEqual(p["head_ms"], 59.4)
        self.assertEqual(p["attn_ms"], 20.5)

    def test_generated_text_excludes_banner_and_summary(self):
        row = bench.parse_tinystories(TINYSTORIES)
        self.assertIn("Once upon a time", row["answer"])
        for marker in ("build:", "throughput:", "tokens in", "profile ms/token"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, row["answer"])


class TestSummary(unittest.TestCase):
    def test_totals_and_median(self):
        rows = [{"units": 10, "ms": 1000, "words": 9},
                {"units": 20, "ms": 1000, "words": 18},
                {"units": 30, "ms": 3000, "words": 27}]
        s = bench.summarise(rows, "piece")
        self.assertEqual(s["total_pieces"], 60)
        self.assertEqual(s["total_readable_words"], 54)
        self.assertEqual(s["total_ms"], 5000)
        self.assertAlmostEqual(s["ms_per_piece"], 5000 / 60)
        self.assertAlmostEqual(s["median_ms_per_piece"], 100.0)

    def test_rows_without_timing_are_excluded(self):
        s = bench.summarise([{"units": 0, "ms": 0, "words": 0},
                             {"units": 10, "ms": 500, "words": 9}], "piece")
        self.assertEqual(s["total_pieces"], 10)

    def test_no_usable_rows_returns_none(self):
        self.assertIsNone(bench.summarise([{"units": 0, "ms": 0, "words": 0}], "piece"))


class TestConfigLine(unittest.TestCase):
    def test_switches_are_parsed(self):
        cfg = bench.parse_config(BARISTA_PROFILED)
        self.assertEqual(cfg["profile"], "1")
        self.assertEqual(cfg["dual_core_active"], "1")
        self.assertEqual(cfg["display_enabled"], "0")

    def test_absent_config_is_none(self):
        self.assertIsNone(bench.parse_config(BARISTA_PLAIN))


class TestFingerprint(unittest.TestCase):
    GOOD = "build: magic=00454c50 bytes=4600186 fp=e602146b scratch_sram=1 fallbacks=0"

    def test_extracted(self):
        self.assertEqual(bench.fingerprint(self.GOOD), "e602146b")

    def test_absent_is_none(self):
        self.assertIsNone(bench.fingerprint(""))
        self.assertIsNone(bench.fingerprint("build: bytes=1 sram=2"))


class TestPassValidation(unittest.TestCase):
    BANNER = ["build: magic=00454c50 bytes=4600186 fp=e602146b scratch_sram=1 fallbacks=0"]
    CONFIG = {"profile": "0", "dual_core_requested": "1", "dual_core_active": "1",
              "display_enabled": "1", "display_present": "1"}

    def rows(self, n=8, **override):
        out = []
        for i, prompt in enumerate(bench.BARISTA_PROMPTS[:n]):
            row = {"prompt": prompt, "answer": "an answer", "units": 30,
                   "words": 28, "ms": 1800, "complete": True}
            if i == 0:
                row.update(override)
            out.append(row)
        return out

    UNSET = object()

    def check(self, rows=None, banner=None, config=UNSET, expect=None):
        return bench.check_pass("barista", banner if banner is not None else self.BANNER,
                                self.CONFIG if config is self.UNSET else config,
                                rows if rows is not None else self.rows(),
                                expect or {})

    def test_a_complete_pass_is_accepted(self):
        self.assertEqual(self.check(), [])

    def test_a_missing_prompt_is_rejected(self):
        problems = self.check(rows=self.rows(7))
        self.assertTrue(any("7 rows, expected 8" in p for p in problems))

    def test_a_prompt_without_timing_is_rejected(self):
        problems = self.check(rows=self.rows(units=0, ms=0))
        self.assertTrue(any("no timing line" in p for p in problems))

    def test_a_prompt_that_never_finished_is_rejected(self):
        problems = self.check(rows=self.rows(complete=False))
        self.assertTrue(any("never reached the end marker" in p for p in problems))

    def test_an_empty_answer_is_rejected(self):
        problems = self.check(rows=self.rows(answer=""))
        self.assertTrue(any("no generated text" in p for p in problems))

    def test_a_board_reporting_profiling_must_emit_profile_lines(self):
        # Inferred from the board, not from the caller's expectations.
        on = dict(self.CONFIG, profile="1")
        problems = self.check(config=on)
        self.assertTrue(any("no profile line" in p for p in problems))

    def test_missing_build_line_is_rejected(self):
        problems = self.check(banner=["model: Vin=8057"])
        self.assertTrue(any("no build: line" in p for p in problems))

    def test_build_without_fingerprint_is_rejected(self):
        problems = self.check(banner=["build: bytes=4600186 scratch_sram=1"])
        self.assertTrue(any("no fingerprint" in p for p in problems))

    def test_sram_fallback_is_rejected(self):
        banner = [self.BANNER[0].replace("fallbacks=0", "fallbacks=3")]
        self.assertTrue(any("fell back" in p for p in self.check(banner=banner)))

    def test_expected_config_must_match(self):
        problems = self.check(expect={"dual_core_active": "0"})
        self.assertTrue(any("dual_core_active=1, expected 0" in p for p in problems))

    def test_expected_config_key_must_exist(self):
        problems = self.check(expect={"nonesuch": "1"})
        self.assertTrue(any("no config key" in p for p in problems))

    def test_expectations_without_a_config_line_are_rejected(self):
        problems = self.check(config=None, expect={"dual_core_active": "1"})
        self.assertTrue(any("no config: line" in p for p in problems))

    def test_barista_requires_a_config_line_even_without_expectations(self):
        problems = self.check(config=None)
        self.assertTrue(any("no config: line" in p for p in problems))


class TestAggregate(unittest.TestCase):
    def make(self, values):
        return [{"summary": {"ms_per_piece": v}} for v in values]

    def test_min_median_max_and_spread(self):
        agg = bench.aggregate(self.make([100.0, 110.0, 105.0]), "piece")
        self.assertEqual(agg["min_ms_per_piece"], 100.0)
        self.assertEqual(agg["median_ms_per_piece"], 105.0)
        self.assertEqual(agg["max_ms_per_piece"], 110.0)
        self.assertAlmostEqual(agg["spread_pct"], 10.0)
        self.assertEqual(agg["passes"], 3)


class FakeSerial:
    """Hands out captured output in chunks, and records how long it was read."""

    def __init__(self, text, chunk=64):
        self.data = text.encode()
        self.chunk = chunk
        self.reads = 0

    def read(self, _n):
        if not self.data:
            return b""
        self.reads += 1
        out, self.data = self.data[:self.chunk], self.data[self.chunk:]
        return out


class TestCompletionPredicates(unittest.TestCase):
    def test_barista_stops_at_the_prompt_marker(self):
        # Byte at a time, so the predicate is evaluated exactly at the marker
        # rather than after a chunk that already swept past it.
        s = FakeSerial(BARISTA_PLAIN + "trailing junk that must not be read", chunk=1)
        buf = bench.read_until(s, bench.barista_done, timeout=5)
        self.assertTrue(buf.rstrip().endswith("READY>"))
        self.assertNotIn("trailing junk", buf)
        self.assertTrue(s.data, "it should have stopped with data still pending")

    def test_tinystories_stops_at_the_profile_line(self):
        # The defect this guards: without a predicate the pass ran until the
        # full timeout even though the board had finished.
        s = FakeSerial(TINYSTORIES)
        started = time.time()
        buf = bench.read_until(s, bench.make_tinystories_done(), timeout=30)
        self.assertLess(time.time() - started, 2.0)
        self.assertIn("profile ms/token", buf)

    def test_tinystories_without_a_profile_line_still_finishes(self):
        no_profile = TINYSTORIES[:TINYSTORIES.index("profile ms/token")]
        s = FakeSerial(no_profile)
        started = time.time()
        buf = bench.read_until(s, bench.make_tinystories_done(), timeout=30)
        elapsed = time.time() - started
        self.assertIn("throughput:", buf)
        self.assertGreaterEqual(elapsed, bench.TINY_GRACE_S)
        self.assertLess(elapsed, bench.TINY_GRACE_S + 3.0)

    def test_a_timeout_returns_what_arrived(self):
        s = FakeSerial("nothing conclusive here")
        buf = bench.read_until(s, bench.barista_done, timeout=0.3)
        self.assertIn("nothing conclusive", buf)


class TestCrossPassAgreement(unittest.TestCase):
    def passes(self, answers, fps=None, configs=None):
        out = []
        for i, answer in enumerate(answers):
            out.append({
                "rows": [{"prompt": "q", "answer": answer, "units": 10,
                          "words": 2, "ms": 1000}],
                "banner": [f"build: fp={(fps or ['aaaaaaaa'] * len(answers))[i]}"],
                "config": (configs or [{"dual_core_active": "1"}] * len(answers))[i],
            })
        return out

    def test_agreeing_passes_are_accepted(self):
        self.assertEqual(bench.check_passes_agree(self.passes(["x", "x"])), [])

    def test_differing_text_between_passes_is_rejected(self):
        problems = bench.check_passes_agree(self.passes(["x", "y"]))
        self.assertTrue(any("differs between passes" in p for p in problems))

    def test_differing_fingerprints_are_rejected(self):
        problems = bench.check_passes_agree(
            self.passes(["x", "x"], fps=["aaaaaaaa", "bbbbbbbb"]))
        self.assertTrue(any("disagree on the weights fingerprint" in p for p in problems))

    def test_differing_configuration_is_rejected(self):
        problems = bench.check_passes_agree(self.passes(
            ["x", "x"], configs=[{"dual_core_active": "1"}, {"dual_core_active": "0"}]))
        self.assertTrue(any("disagree on the build configuration" in p for p in problems))


class TestComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def receipt(self, model="barista", fp="e602146b", answer="one two", ms=1000,
                passes=1):
        rows = [{"prompt": "q", "answer": answer, "units": 10, "words": 2, "ms": ms}]
        pass_list = [{"rows": rows, "summary": bench.summarise(rows, "piece")}
                     for _ in range(passes)]
        return {
            "model": model,
            "build": f"build: fp={fp} fallbacks=0" if fp else "build: fallbacks=0",
            "config": {"dual_core_active": "1"},
            "passes": pass_list,
            "aggregate": bench.aggregate(pass_list, "piece"),
        }

    def write(self, receipt):
        path = self.dir / "base.json"
        path.write_text(json.dumps(receipt))
        return path

    def compare(self, current, baseline, allow=False, model="barista"):
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.compare(model, current, baseline, "piece", allow)

    def test_identical_runs_compare_cleanly(self):
        self.assertEqual(self.compare(self.receipt(), self.write(self.receipt())), [])

    def test_changed_text_is_refused(self):
        base = self.write(self.receipt(answer="one two"))
        problems = self.compare(self.receipt(answer="one three"), base)
        self.assertTrue(any("changed" in p for p in problems))

    def test_changed_text_in_any_pass_is_refused(self):
        # The defect this guards: a differing pass must not be averaged away.
        base = self.write(self.receipt(answer="one two", passes=3))
        current = self.receipt(answer="one two", passes=3)
        current["passes"][1]["rows"][0]["answer"] = "one three"
        self.assertTrue(any("changed" in p for p in self.compare(current, base)))

    def test_different_weights_are_refused(self):
        base = self.write(self.receipt(fp="aaaaaaaa"))
        problems = self.compare(self.receipt(fp="bbbbbbbb"), base)
        self.assertTrue(any("different weights" in p for p in problems))

    def test_different_weights_may_be_overridden(self):
        base = self.write(self.receipt(fp="aaaaaaaa"))
        self.assertEqual(self.compare(self.receipt(fp="bbbbbbbb"), base, allow=True), [])

    def test_baseline_without_a_fingerprint_is_refused(self):
        base = self.write(self.receipt(fp=""))
        problems = self.compare(self.receipt(), base)
        self.assertTrue(any("no fingerprint" in p for p in problems))

    def test_current_run_without_a_fingerprint_is_refused(self):
        base = self.write(self.receipt())
        problems = self.compare(self.receipt(fp=""), base)
        self.assertTrue(any("no fingerprint" in p for p in problems))

    def test_comparing_across_models_is_refused(self):
        base = self.write(self.receipt(model="tinystories"))
        problems = self.compare(self.receipt(), base)
        self.assertTrue(any("baseline is tinystories" in p for p in problems))

    def test_a_speedup_does_not_excuse_changed_text(self):
        base = self.write(self.receipt(answer="one two", ms=2000))
        problems = self.compare(self.receipt(answer="one three", ms=1000), base)
        self.assertTrue(any("changed" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
