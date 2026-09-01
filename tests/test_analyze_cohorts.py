"""The analyzer's safety checks, which are the only thing standing between a
mixed set of runs and a confident wrong number.

Each test asserts a specific way a cohort can be invalid. They exist because
every one of these checks was, at some point, present in the code but not
actually firing.

  uv run python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from research.tinystories import analyze


def record(arm, seed, tag="clean", vocab=4096, seq_len=512, **training):
    sched = {"batch_size": 32, "steps": 3000, "lr": 1e-3, "warmup": 200,
             "eval_every": 250, "eval_iters": 40, "target_core": 1500000,
             "fixed_ffn": None, "seed": seed}
    sched.update(training)
    return {
        "arm": arm, "seed": seed, "tag": tag,
        "config": {"arm": arm, "vocab_size": vocab, "seq_len": seq_len,
                   "d_model": 128, "n_layers": 6, "n_heads": 4,
                   "ffn_hidden": 415, "ple_dim": 64, "rope_theta": 10000.0},
        "training": sched,
        "params": {"core": 1_499_328, "table": 1_572_864, "stream": 0, "total": 0},
        "final_val": 2.0 + seed * 0.01,
        "tokenizer_sha256": "a" * 64,
        "steps": sched["steps"],
        "tokens_seen": sched["steps"] * sched["batch_size"] * seq_len,
    }


class CohortDir:
    def __init__(self, records):
        self.tmp = tempfile.TemporaryDirectory()
        for i, r in enumerate(records):
            name = f"{r['arm']}-{r['tag']}-s{r.get('seed', 'x')}-{i}.json"
            Path(self.tmp.name, name).write_text(json.dumps(r))

    def __enter__(self):
        return self.tmp.name

    def __exit__(self, *a):
        self.tmp.cleanup()


def run(runs_dir, tag="clean", arms="baseline,ple", seeds=2):
    by_arm, _ = analyze.load(runs_dir, tag)
    analyze.check_comparable(by_arm)
    analyze.check_complete(by_arm, {a for a in arms.split(",")}, seeds)
    return by_arm


class TestComparability(unittest.TestCase):
    def test_valid_cohort_passes(self):
        recs = [record(a, s) for a in ("baseline", "ple") for s in (0, 1)]
        with CohortDir(recs) as d:
            self.assertEqual(set(run(d)), {"baseline", "ple"})

    def test_mixed_vocabulary_rejected(self):
        recs = [record("baseline", s) for s in (0, 1)]
        recs += [record("ple", s, vocab=32768) for s in (0, 1)]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("vocab_size", str(e.exception))

    def test_mixed_schedule_rejected(self):
        recs = [record("baseline", s) for s in (0, 1)]
        recs += [record("ple", s, steps=9999) for s in (0, 1)]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("steps", str(e.exception))

    def test_field_present_in_only_some_runs_rejected(self):
        """The bug this file exists for: a field one arm records and another
        does not was reported as verified from the subset that had it."""
        recs = [record("baseline", s) for s in (0, 1)]
        ple = [record("ple", s) for s in (0, 1)]
        for r in ple:
            del r["training"]["lr"]
        with CohortDir(recs + ple) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("training.lr", str(e.exception))
        self.assertIn("2 of 4", str(e.exception))

    def test_recorded_null_is_a_value_not_an_absence(self):
        """fixed_ffn is written as null when the arm solved its own ffn.
        Treating that as missing would compare null against 256 and pass."""
        recs = [record("baseline", s) for s in (0, 1)]
        recs += [record("ple", s, fixed_ffn=256) for s in (0, 1)]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("fixed_ffn", str(e.exception))

    def test_absent_from_every_run_is_unverifiable_not_an_error(self):
        recs = [record(a, s) for a in ("baseline", "ple") for s in (0, 1)]
        for r in recs:
            del r["tokenizer_sha256"]
        with CohortDir(recs) as d:
            by_arm, _ = analyze.load(d, "clean")
            _, unverifiable = analyze.check_comparable(by_arm)
        self.assertIn("tokenizer_sha256", unverifiable)


class TestCompleteness(unittest.TestCase):
    def test_missing_arm_rejected(self):
        recs = [record("baseline", s) for s in (0, 1)]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("missing ple", str(e.exception))

    def test_unexpected_arm_rejected(self):
        recs = [record(a, s) for a in ("baseline", "ple", "bigcore") for s in (0, 1)]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("unexpected", str(e.exception))

    def test_duplicate_records_at_one_seed_rejected(self):
        recs = [record(a, s) for a in ("baseline", "ple") for s in (0, 1)]
        recs.append(record("ple", 0))
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("distinct seeds", str(e.exception))

    def test_arms_on_different_seed_sets_rejected(self):
        recs = [record("baseline", s) for s in (0, 1)]
        recs += [record("ple", s) for s in (0, 2)]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("same seeds", str(e.exception))

    def test_missing_seed_rejected(self):
        recs = [record(a, s) for a in ("baseline", "ple") for s in (0, 1)]
        del recs[-1]["seed"]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d)
        self.assertIn("seed", str(e.exception))

    def test_wrong_seed_count_rejected(self):
        recs = [record(a, 0) for a in ("baseline", "ple")]
        with CohortDir(recs) as d, self.assertRaises(SystemExit) as e:
            run(d, seeds=2)
        self.assertIn("expected 2 seeds", str(e.exception))


class TestSelection(unittest.TestCase):
    def test_other_tags_are_excluded(self):
        recs = [record(a, s) for a in ("baseline", "ple") for s in (0, 1)]
        recs += [record(a, s, tag="cleandeploy", vocab=32768)
                 for a in ("baseline", "ple") for s in (0, 1)]
        with CohortDir(recs) as d:
            by_arm, skipped = analyze.load(d, "clean")
        self.assertEqual(skipped, 4)
        self.assertEqual(sum(len(v) for v in by_arm.values()), 4)

    def test_records_without_an_arm_are_ignored(self):
        recs = [record(a, s) for a in ("baseline", "ple") for s in (0, 1)]
        with CohortDir(recs) as d:
            Path(d, "bench_device.json").write_text(
                json.dumps({"ms_per_token": 94.9, "tag": "clean"}))
            by_arm, _ = analyze.load(d, "clean")
        self.assertEqual(set(by_arm), {"baseline", "ple"})


if __name__ == "__main__":
    unittest.main()
