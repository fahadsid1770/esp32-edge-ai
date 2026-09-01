"""The Barista exporter's input checks.

The canonical checkpoint is not committed, so the byte comparison against the
released artifact is a local receipt rather than a test. What is covered here is
the part that runs without it: refusing a checkpoint or a pair of tables the
format cannot honour.

Both size relationships matter and neither is visible from the tables alone.
Without them an export would pair the checkpoint weights with incompatible
class-to-token tables.

  uv run python -m unittest discover -s tests
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "firmware" / "esp32_barista" / "tools"
sys.path.insert(0, str(TOOLS))
_spec = importlib.util.spec_from_file_location("barista_export", TOOLS / "export.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

from model import Config  # noqa: E402

SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")
N_WORDS = 854          # the word tables' own fixed size
BPE_VOCAB = 100
N_REUSED = 4
TOTAL = BPE_VOCAB + N_WORDS - N_REUSED


def tables(n_words=N_WORDS, bpe_vocab=BPE_VOCAB, n_reused=N_REUSED):
    """A consistent (vocab, layout) pair, shaped like the canonical one."""
    words = list(SPECIALS) + [f"w{i}" for i in range(n_words - len(SPECIALS))]
    out2in = list(range(n_reused))
    out2in += list(range(bpe_vocab, bpe_vocab + n_words - n_reused))
    vocab = {"version": "test", "total": n_words,
             "tokens": [{"token": w, "tier": "test"} for w in words]}
    layout = {"bpe_vocab": bpe_vocab, "total": bpe_vocab + n_words - n_reused,
              "n_words": n_words, "out2in": out2in}
    return vocab, layout


def config(vocab_size=TOTAL, out_vocab_size=N_WORDS):
    return Config(arm="ple", vocab_size=vocab_size, out_vocab_size=out_vocab_size,
                  d_model=16, n_layers=2, n_heads=2, ffn_hidden=24,
                  seq_len=32, ple_dim=12)


class ExportCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, vocab, layout):
        vocab_path = self.dir / "vocab.json"
        layout_path = self.dir / "layout.json"
        vocab_path.write_text(json.dumps(vocab))
        layout_path.write_text(json.dumps(layout))
        return vocab_path, layout_path

    def check(self, cfg, vocab, layout):
        return export.check_vocabulary(cfg, *self.write(vocab, layout))


class TestUnsupportedConfigKeys(unittest.TestCase):
    def test_null_is_dropped(self):
        cfg = export.load_config({"arm": "ple", "vocab_size": 61,
                                  "active_vocab_size": None, "out_vocab_size": 17})
        self.assertEqual(cfg.vocab_size, 61)
        self.assertEqual(cfg.resolved_out_vocab_size, 17)

    def test_non_null_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "active_vocab_size=8"):
            export.load_config({"arm": "ple", "vocab_size": 61,
                                "active_vocab_size": 8, "out_vocab_size": 17})

    def test_absent_key_is_fine(self):
        cfg = export.load_config({"arm": "ple", "vocab_size": 61})
        self.assertEqual(cfg.vocab_size, 61)


class TestVocabularyAgreement(ExportCase):
    def test_matching_sizes_accepted(self):
        words, out2in, layout = self.check(config(), *tables())
        self.assertEqual(len(words), N_WORDS)
        self.assertEqual(len(out2in), N_WORDS)
        self.assertEqual(layout["total"], TOTAL)

    def test_input_vocabulary_mismatch_rejected(self):
        # The head is the right width, so only the input side disagrees.
        with self.assertRaisesRegex(SystemExit, "input tokens but the checkpoint"):
            self.check(config(vocab_size=TOTAL + 1), *tables())

    def test_output_vocabulary_mismatch_rejected(self):
        with self.assertRaisesRegex(SystemExit, "output classes but the head"):
            self.check(config(out_vocab_size=N_WORDS - 1), *tables())

    def test_table_defects_are_still_caught(self):
        # The shared loader owns internal consistency; this only confirms the
        # exporter routes through it rather than around it.
        vocab, layout = tables()
        layout["out2in"][20] = layout["out2in"][19]
        with self.assertRaisesRegex(SystemExit, "reuses input id"):
            self.check(config(), vocab, layout)


if __name__ == "__main__":
    unittest.main()
