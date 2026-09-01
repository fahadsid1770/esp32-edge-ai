"""Quantization argument bounds and PTQ checkpoint provenance.

The bounds exist because bits=1 produced NaN and group=0 divided a row into
nothing, both silently. The provenance checks exist because a filename is not
an identity: a checkpoint copied over another name, or trained on a different
schedule, otherwise compares as if it belonged.

  uv run python -m unittest discover -s tests
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import torch

from quantize import quantize_groupwise, quantize_model
from research.tinystories import quantize_eval


class TestQuantizationBounds(unittest.TestCase):
    def setUp(self):
        self.w = torch.randn(4, 128)

    def test_valid_arguments_round_trip(self):
        out = quantize_groupwise(self.w, bits=4, group=64)
        self.assertEqual(out.shape, self.w.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_one_bit_rejected(self):
        # Produced NaN rather than an error: one bit leaves no magnitude range.
        with self.assertRaises(ValueError):
            quantize_groupwise(self.w, bits=1, group=64)

    def test_bits_above_eight_rejected(self):
        with self.assertRaises(ValueError):
            quantize_groupwise(self.w, bits=9, group=64)

    def test_zero_group_rejected(self):
        with self.assertRaises(ValueError):
            quantize_groupwise(self.w, bits=4, group=0)

    def test_negative_group_rejected(self):
        with self.assertRaises(ValueError):
            quantize_groupwise(self.w, bits=4, group=-8)

    def test_quantize_model_validates_too(self):
        model = torch.nn.Linear(8, 8)
        with self.assertRaises(ValueError):
            quantize_model(model, bits=1, group=64)

    def test_fp16_scales_stay_finite(self):
        out = quantize_groupwise(self.w, bits=4, group=64, fp16_scales=True)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())


class TestRetainedEdge(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(quantize_eval.format_retained(0.05, 0.10), "50%")

    def test_zero_fp_gap_is_undefined_not_a_crash(self):
        # Equal baseline and PLE results are a real outcome, not a bug to guard.
        self.assertIn("undefined", quantize_eval.format_retained(0.01, 0.0))


def checkpoint(arm, seed=0, tag="t", name=None, vocab=4096, seq_len=512,
               tokenizer="a" * 64, **sched):
    training = {"batch_size": 32, "steps": 3000, "lr": 1e-3, "seed": seed}
    training.update(sched)
    return {
        "cfg": {"arm": arm, "vocab_size": vocab, "d_model": 128, "n_layers": 6,
                "n_heads": 4, "ffn_hidden": 415, "seq_len": seq_len,
                "ple_dim": 64, "rope_theta": 10000.0},
        "state": {},
        "seed": seed, "tag": tag,
        "name": name if name is not None else f"{arm}-{tag}-s{seed}",
        "tokenizer_sha256": tokenizer,
        "training": training,
    }


class RunsDir:
    """Three checkpoints named as quantize_eval expects to find them."""

    def __init__(self, overrides=None):
        self.tmp = tempfile.TemporaryDirectory()
        for arm in ("baseline", "ple", "fatembed"):
            ck = checkpoint(arm)
            for k, v in (overrides or {}).get(arm, {}).items():
                if k == "training":
                    ck["training"].update(v)
                else:
                    ck[k] = v
            torch.save(ck, os.path.join(self.tmp.name, f"{arm}-t-s0.pt"))

    def __enter__(self):
        return self.tmp.name

    def __exit__(self, *a):
        self.tmp.cleanup()


class ReachedEvaluation(Exception):
    """Raised by a patched load() once provenance validation has completed."""


def run_eval(runs_dir, stop_after_validation=False):
    """Drive main() far enough to hit the provenance checks. They run before any
    model is loaded, so the empty state dicts above are never touched.

    With stop_after_validation=True, load() is replaced by one that raises a
    unique sentinel, and ONLY that sentinel is caught. Catching Exception
    instead would let a crash before validation look like success: the warnings
    under test would never be printed and the assertions would still pass.
    """
    old_runs, old_argv = quantize_eval.RUNS, sys.argv
    old_load = quantize_eval.load
    quantize_eval.RUNS = runs_dir
    sys.argv = ["quantize_eval", "--tag", "t", "--seed", "0"]

    def stop(*_a, **_k):
        raise ReachedEvaluation

    if stop_after_validation:
        quantize_eval.load = stop
    buf = io.StringIO()
    reached = False
    try:
        with contextlib.redirect_stdout(buf):
            quantize_eval.main()
    except ReachedEvaluation:
        reached = True
    finally:
        quantize_eval.RUNS, sys.argv = old_runs, old_argv
        quantize_eval.load = old_load

    if stop_after_validation and not reached:
        raise AssertionError(
            "main() returned without reaching evaluation; provenance validation "
            "did not complete as the test assumes")
    return buf.getvalue()


class TestCheckpointProvenance(unittest.TestCase):
    def test_wrong_arm_in_the_file_rejected(self):
        with RunsDir({"baseline": {"cfg": checkpoint("ple")["cfg"]}}) as d:
            with self.assertRaises(SystemExit) as e:
                run_eval(d)
        self.assertIn("arm=", str(e.exception))

    def test_wrong_seed_rejected(self):
        with RunsDir({"ple": {"seed": 1}}) as d, self.assertRaises(SystemExit) as e:
            run_eval(d)
        self.assertIn("seed", str(e.exception))

    def test_wrong_tag_rejected(self):
        with RunsDir({"ple": {"tag": "other"}}) as d, self.assertRaises(SystemExit) as e:
            run_eval(d)
        self.assertIn("tag", str(e.exception))

    def test_wrong_name_rejected(self):
        with RunsDir({"ple": {"name": "ple-elsewhere-s0"}}) as d:
            with self.assertRaises(SystemExit) as e:
                run_eval(d)
        self.assertIn("name", str(e.exception))

    def test_mismatched_schedule_rejected_and_names_the_field(self):
        with RunsDir({"ple": {"training": {"steps": 9999}}}) as d:
            with self.assertRaises(SystemExit) as e:
                run_eval(d)
        msg = str(e.exception)
        self.assertIn("training.steps", msg)
        self.assertIn("9999", msg)

    def test_mismatched_tokenizer_rejected(self):
        with RunsDir({"ple": {"tokenizer_sha256": "b" * 64}}) as d:
            with self.assertRaises(SystemExit) as e:
                run_eval(d)
        self.assertIn("tokenizer", str(e.exception))

    def test_mismatched_vocab_rejected(self):
        cfg = checkpoint("ple", vocab=32768)["cfg"]
        with RunsDir({"ple": {"cfg": cfg}}) as d, self.assertRaises(SystemExit) as e:
            run_eval(d)
        self.assertIn("vocab_size", str(e.exception))

    def test_missing_checkpoint_rejected(self):
        with RunsDir() as d:
            os.remove(os.path.join(d, "fatembed-t-s0.pt"))
            with self.assertRaises(SystemExit) as e:
                run_eval(d)
        self.assertIn("missing checkpoints", str(e.exception))

    def test_legacy_gaps_are_reported_by_kind(self):
        """Identity can be inferred from the filename; tokenizer and schedule
        cannot be recovered at all. The warnings must not conflate them."""
        with RunsDir() as d:
            for arm in ("baseline", "ple", "fatembed"):
                path = os.path.join(d, f"{arm}-t-s0.pt")
                ck = torch.load(path, map_location="cpu", weights_only=False)
                for k in ("name", "seed", "tag", "tokenizer_sha256", "training"):
                    ck.pop(k, None)
                torch.save(ck, path)
            out = run_eval(d, stop_after_validation=True)

        self.assertIn("taken from the filename", out)
        self.assertIn("cannot be verified at all", out)
        # and the two kinds are in separate messages, not one list
        from_name = [l for l in out.splitlines() if "taken from the filename" in l]
        unrecoverable = [l for l in out.splitlines() if "cannot be verified at all" in l]
        self.assertEqual(len(from_name), 1)
        self.assertEqual(len(unrecoverable), 1)
        self.assertNotIn("tokenizer_sha256", from_name[0])
        self.assertNotIn("seed", unrecoverable[0])

    def test_complete_provenance_produces_no_warnings(self):
        """The counterpart: a fully recorded cohort must warn about nothing, or
        the warnings above prove only that the code prints something."""
        with RunsDir() as d:
            out = run_eval(d, stop_after_validation=True)
        self.assertNotIn("WARNING", out)


if __name__ == "__main__":
    unittest.main()
