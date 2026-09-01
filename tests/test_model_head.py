"""The shared model's output head, whose width can differ from the input side.

Two independent questions decide the head's behaviour:

  output_ids_are_input_ids   can a sampled index be fed straight back in
  head_is_tied               does the head share storage with tok_emb

They differ for fatembed, which widens tok_emb and so keeps a separate head
while still writing the vocabulary it reads.

Fixtures are small and synthetic, except the per-arm parameter budgets, which
are pinned at the default shape so that a change to the head cannot move them.

  uv run python -m unittest discover -s tests
"""

import unittest

import torch

from model import Config, TinyLM

# A small asymmetric model: reads 61 tokens, writes 17 classes.
# ple_dim is chosen so table_width (n_layers * ple_dim = 24) differs from
# d_model (16), which is the condition that keeps fatembed's head separate.
ASYMMETRIC = dict(arm="ple", vocab_size=61, out_vocab_size=17, d_model=16,
                  n_layers=2, n_heads=2, ffn_hidden=24, seq_len=32, ple_dim=12)
SYMMETRIC = {**ASYMMETRIC, "out_vocab_size": None}

# Parameter budgets of each arm at the default shape. Integers derived from the
# architecture, so exact.
EXPECTED_ARM_BUDGETS = {
    "baseline":    {"core": 984704,  "stream": 524288, "table": 0,       "total": 1508992},
    "ple":         {"core": 1132992, "stream": 524288, "table": 1572864, "total": 3230144},
    "ple_notable": {"core": 1132992, "stream": 524288, "table": 0,       "total": 1657280},
    "fatembed":    {"core": 1033856, "stream": 524288, "table": 1572864, "total": 3131008},
}


def build(seed=0, **kwargs):
    torch.manual_seed(seed)
    return TinyLM(Config(**kwargs))


def forward(model, vocab, seed=1, batch=2, length=8):
    torch.manual_seed(seed)
    with torch.no_grad():
        logits, _ = model(torch.randint(0, vocab, (batch, length)))
    return logits


class TestConfigProperties(unittest.TestCase):
    def test_default_head_spans_the_input_vocabulary(self):
        cfg = Config(**SYMMETRIC)
        self.assertEqual(cfg.resolved_out_vocab_size, cfg.vocab_size)
        self.assertTrue(cfg.output_ids_are_input_ids)
        self.assertTrue(cfg.head_is_tied)

    def test_stating_the_same_size_is_the_same_as_omitting_it(self):
        cfg = Config(**{**SYMMETRIC, "out_vocab_size": SYMMETRIC["vocab_size"]})
        self.assertTrue(cfg.output_ids_are_input_ids)
        self.assertTrue(cfg.head_is_tied)

    def test_distinct_alphabet_is_neither_tied_nor_feedable(self):
        cfg = Config(**ASYMMETRIC)
        self.assertEqual(cfg.resolved_out_vocab_size, 17)
        self.assertFalse(cfg.output_ids_are_input_ids)
        self.assertFalse(cfg.head_is_tied)

    def test_fatembed_writes_what_it_reads_but_cannot_share_storage(self):
        # The two properties disagree here, which is the whole reason they are
        # two properties: tok_emb is table_width wide, the head is d_model wide.
        cfg = Config(**{**SYMMETRIC, "arm": "fatembed"})
        self.assertTrue(cfg.output_ids_are_input_ids)
        self.assertFalse(cfg.head_is_tied)


class TestHeadConstruction(unittest.TestCase):
    def test_symmetric_head_shares_storage(self):
        model = build(**SYMMETRIC)
        self.assertIs(model.head.weight, model.tok_emb.weight)

    def test_asymmetric_head_is_its_own_tensor(self):
        model = build(**ASYMMETRIC)
        self.assertIsNot(model.head.weight, model.tok_emb.weight)
        self.assertEqual(tuple(model.tok_emb.weight.shape), (61, 16))
        self.assertEqual(tuple(model.head.weight.shape), (17, 16))

    def test_fatembed_head_is_separate_because_the_widths_differ(self):
        cfg = Config(**{**SYMMETRIC, "arm": "fatembed"})
        model = build(**{**SYMMETRIC, "arm": "fatembed"})
        self.assertNotEqual(cfg.table_width, cfg.d_model)
        self.assertIsNot(model.head.weight, model.tok_emb.weight)
        self.assertEqual(tuple(model.head.weight.shape), (cfg.vocab_size, cfg.d_model))
        self.assertEqual(tuple(model.tok_emb.weight.shape),
                         (cfg.vocab_size, cfg.table_width))

    def test_logits_have_the_head_width(self):
        logits = forward(build(**ASYMMETRIC), 61, batch=3, length=5)
        self.assertEqual(tuple(logits.shape), (3, 5, 17))


class TestLoss(unittest.TestCase):
    def test_loss_reshapes_by_the_head_width(self):
        # 17 columns do not divide into 61, so reshaping by the read vocabulary
        # cannot even be attempted here.
        model = build(**ASYMMETRIC)
        torch.manual_seed(2)
        logits, loss = model(torch.randint(0, 61, (2, 8)),
                             targets=torch.randint(0, 17, (2, 8)))
        self.assertEqual(tuple(logits.shape), (2, 8, 17))
        self.assertTrue(torch.isfinite(loss))

    def test_ignored_targets_stay_ignored(self):
        model = build(**ASYMMETRIC)
        torch.manual_seed(3)
        targets = torch.full((2, 8), -1)
        targets[0, 0] = 5
        _, loss = model(torch.randint(0, 61, (2, 8)), targets=targets)
        self.assertTrue(torch.isfinite(loss))


class TestParameterBudget(unittest.TestCase):
    def test_symmetric_embedding_is_not_double_counted(self):
        cfg = Config(**SYMMETRIC)
        budget = build(**SYMMETRIC).param_budget()
        self.assertEqual(budget["stream"], cfg.vocab_size * cfg.d_model)
        self.assertEqual(budget["table"], cfg.vocab_size * cfg.n_layers * cfg.ple_dim)

    def test_standalone_embedding_counts_as_table_not_core(self):
        # Once tok_emb stops doubling as the head it is read one row per token,
        # which is table traffic, not dense core.
        cfg = Config(**ASYMMETRIC)
        budget = build(**ASYMMETRIC).param_budget()
        self.assertEqual(budget["stream"], cfg.resolved_out_vocab_size * cfg.d_model)
        self.assertEqual(
            budget["table"],
            cfg.vocab_size * cfg.n_layers * cfg.ple_dim + cfg.vocab_size * cfg.d_model)
        self.assertEqual(budget["core"],
                         budget["total"] - budget["table"] - budget["stream"])

    def test_arm_budgets(self):
        for arm, want in EXPECTED_ARM_BUDGETS.items():
            with self.subTest(arm=arm):
                self.assertEqual(build(arm=arm, vocab_size=4096).param_budget(), want)

    def test_stating_the_default_explicitly_changes_nothing(self):
        for arm in ("baseline", "ple", "fatembed"):
            with self.subTest(arm=arm):
                implicit = build(arm=arm, vocab_size=4096)
                explicit = build(arm=arm, vocab_size=4096, out_vocab_size=4096)
                self.assertEqual(implicit.param_budget(), explicit.param_budget())
                self.assertTrue(torch.equal(forward(implicit, 4096),
                                            forward(explicit, 4096)))


class TestGenerate(unittest.TestCase):
    def test_generate_works_when_output_ids_are_input_ids(self):
        for arm in ("baseline", "ple", "fatembed"):
            with self.subTest(arm=arm):
                model = build(**{**SYMMETRIC, "arm": arm})
                out = model.generate(torch.zeros(1, 4, dtype=torch.long), 3)
                self.assertEqual(tuple(out.shape), (1, 7))

    def test_generate_refuses_a_distinct_output_alphabet(self):
        model = build(**ASYMMETRIC)
        with self.assertRaisesRegex(ValueError, "span the input vocabulary"):
            model.generate(torch.zeros(1, 4, dtype=torch.long), 3)

    def test_the_refusal_names_both_sizes(self):
        model = build(**ASYMMETRIC)
        for pattern in ("17 classes", "61 tokens"):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    model.generate(torch.zeros(1, 4, dtype=torch.long), 1)

    def test_refusal_is_keyed_on_the_alphabet_not_on_weight_sharing(self):
        # fatembed does not share storage, but its output ids are token ids, so
        # generate is valid for it. The guard therefore keys on the id space,
        # not on whether the tensors are shared.
        model = build(**{**SYMMETRIC, "arm": "fatembed"})
        self.assertFalse(model.cfg.head_is_tied)
        self.assertEqual(tuple(model.generate(
            torch.zeros(1, 2, dtype=torch.long), 2).shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
