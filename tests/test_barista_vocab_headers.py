"""The Barista vocabulary header generator, the only thing keeping the two
device headers in step with the frozen output vocabulary.

These cover the JSON to C conversion contract, not the model: nothing here
knows what the trained head contains, only that the two canonical inputs agree
with each other and survive the trip into C.

Every failure these tests describe is silent on device. A class mapped to the
wrong input id still prints the right word, then feeds the wrong token into the
next decoding step, so the damage surfaces later in the answer; a table one row
short still compiles. So each test names one specific way vocab.json and
layout.json can disagree while both remain valid JSON.

The canonical inputs are distributed on Hugging Face and are not committed to
Git, so they are absent until fetched. Every fixture here is therefore
synthetic and the tests do not require fetched HF assets. Each rejection is
matched on its message, because a test that accepts any SystemExit would also
pass if the generator failed while opening the file.

  uv run python -m unittest discover -s tests
"""

import contextlib
import importlib.util
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

GENERATOR = (Path(__file__).resolve().parents[1] / "firmware" / "esp32_barista"
             / "tools" / "generate_vocab_headers.py")
# The sketch tools are deliberately not packaged: they are build steps for one
# board, not a library. Load by path rather than inventing a package for them.
_spec = importlib.util.spec_from_file_location("barista_generate_vocab_headers",
                                               GENERATOR)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def assets(n_words=gen.EXPECTED_WORDS, bpe_vocab=100, n_reused=4):
    """A consistent (vocab, layout) pair, shaped like the canonical one.

    n_reused classes keep a low id that already existed in the BPE vocabulary;
    the rest get fresh rows appended above it, exactly as the real layout does.
    """
    words = list(gen.SPECIALS)
    words += [f"w{i}" for i in range(n_words - len(gen.SPECIALS))]
    out2in = list(range(n_reused))
    out2in += list(range(bpe_vocab, bpe_vocab + n_words - n_reused))
    vocab = {"version": "test-output-vocab", "total": n_words,
             "tokens": [{"token": w, "tier": "test"} for w in words]}
    layout = {"bpe_vocab": bpe_vocab, "total": bpe_vocab + n_words - n_reused,
              "n_words": n_words, "out2in": out2in}
    return vocab, layout


class GeneratorCase(unittest.TestCase):
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

    def run_generator(self, vocab, layout, out_dir=None):
        vocab_path, layout_path = self.write(vocab, layout)
        # The generator reports what it wrote; keep that off the test output.
        with contextlib.redirect_stdout(io.StringIO()):
            return gen.generate(vocab_path, layout_path,
                                out_dir or self.dir / "generated")

    def reject(self, pattern, vocab, layout):
        with self.assertRaisesRegex(SystemExit, pattern):
            self.run_generator(vocab, layout)


class TestValidInput(GeneratorCase):
    def test_writes_both_headers(self):
        words_h, out2in_h = self.run_generator(*assets())
        self.assertTrue(words_h.exists() and out2in_h.exists())
        text = words_h.read_text()
        self.assertIn(f"#define BARISTA_WORD_COUNT {gen.EXPECTED_WORDS}", text)
        self.assertIn(f"BARISTA_WORDS[{gen.EXPECTED_WORDS}]", text)
        self.assertIn(f"BARISTA_OUT2IN[{gen.EXPECTED_WORDS}]", out2in_h.read_text())

    def test_creates_a_missing_output_directory(self):
        # firmware/**/generated/ is ignored, so it does not exist in a clone.
        out_dir = self.dir / "absent" / "generated"
        words_h, _ = self.run_generator(*assets(), out_dir=out_dir)
        self.assertTrue(words_h.exists())

    def test_output_is_deterministic(self):
        vocab, layout = assets()
        first = [p.read_bytes() for p in self.run_generator(vocab, layout)]
        second = [p.read_bytes() for p in
                  self.run_generator(vocab, layout, out_dir=self.dir / "again")]
        self.assertEqual(first, second)

    def test_word_and_id_order_survives_the_round_trip(self):
        vocab, layout = assets()
        words_h, out2in_h = self.run_generator(vocab, layout)
        body = words_h.read_text().split("= {", 1)[1].split("};", 1)[0]
        emitted = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        self.assertEqual(emitted, [t["token"] for t in vocab["tokens"]])
        body = out2in_h.read_text().split("= {", 1)[1].split("};", 1)[0]
        self.assertEqual([int(v) for v in re.findall(r"\d+", body)], layout["out2in"])

    def test_special_constants_are_zero_to_three(self):
        words_h, _ = self.run_generator(*assets())
        text = words_h.read_text()
        for i, name in enumerate(("PAD", "BOS", "EOS", "UNK")):
            self.assertIn(f"#define BARISTA_{name} {i}", text)

    def test_reused_id_count_is_reported_in_the_header(self):
        # The comment explains why the map is not a uniform offset. If it were
        # a constant it would quietly lie the first time the layout changed.
        _, out2in_h = self.run_generator(*assets(n_reused=7))
        self.assertIn("// 7 writable words already existed", out2in_h.read_text())


class TestCStringSafety(GeneratorCase):
    def test_quote_and_backslash_are_escaped(self):
        vocab, layout = assets()
        vocab["tokens"][9]["token"] = 'a"b\\c'
        words_h, _ = self.run_generator(vocab, layout)
        self.assertIn(r'"a\"b\\c",', words_h.read_text())

    def test_non_ascii_word_is_rejected(self):
        vocab, layout = assets()
        vocab["tokens"][9]["token"] = "café"
        self.reject("outside printable ASCII", vocab, layout)

    def test_control_character_is_rejected(self):
        vocab, layout = assets()
        vocab["tokens"][9]["token"] = "a\nb"
        self.reject("outside printable ASCII", vocab, layout)


class TestSizeAgreement(GeneratorCase):
    def test_wrong_word_count_rejected(self):
        self.reject(f"expected {gen.EXPECTED_WORDS}",
                    *assets(n_words=gen.EXPECTED_WORDS - 1))

    def test_n_words_disagreeing_with_the_token_list_rejected(self):
        vocab, layout = assets()
        layout["n_words"] -= 1
        self.reject("size disagreement", vocab, layout)

    def test_out2in_shorter_than_the_token_list_rejected(self):
        vocab, layout = assets()
        layout["out2in"] = layout["out2in"][:-1]
        self.reject("size disagreement", vocab, layout)


class TestVocabularyMetadata(GeneratorCase):
    """vocab.json states its size twice and must not contradict itself."""

    def test_total_disagreeing_with_the_token_list_rejected(self):
        vocab, layout = assets()
        vocab["total"] = 999
        self.reject("self-contradictory", vocab, layout)

    def test_duplicate_output_word_rejected(self):
        # Two classes emitting one word: the head can pick either, decoding
        # cannot tell them apart, and one of the 854 rows is wasted.
        vocab, layout = assets()
        vocab["tokens"][9]["token"] = vocab["tokens"][8]["token"]
        self.reject("repeats the word", vocab, layout)


class TestIntegerTypes(GeneratorCase):
    """JSON has one number type, so a size can arrive as a float or a bool and
    satisfy every range check before C truncates it."""

    def test_fractional_input_id_rejected(self):
        # 100.5 reaches the header, the compiler truncates it to 100, and that
        # can then collide with a class that legitimately owns 100.
        vocab, layout = assets()
        layout["out2in"][20] = 100.5
        self.reject("is not an integer", vocab, layout)

    def test_boolean_input_id_rejected(self):
        vocab, layout = assets()
        layout["out2in"][20] = True
        self.reject("is not an integer", vocab, layout)

    def test_string_input_id_rejected(self):
        vocab, layout = assets()
        layout["out2in"][20] = "116"
        self.reject("is not an integer", vocab, layout)

    def test_fractional_layout_scalar_rejected(self):
        for field in ("bpe_vocab", "total", "n_words"):
            with self.subTest(field=field):
                vocab, layout = assets()
                layout[field] = layout[field] + 0.5
                self.reject(f"layout {field} must be an integer", vocab, layout)

    def test_boolean_layout_scalar_rejected(self):
        for field in ("bpe_vocab", "total", "n_words"):
            with self.subTest(field=field):
                vocab, layout = assets()
                layout[field] = True
                self.reject(f"layout {field} must be an integer", vocab, layout)

    def test_fractional_vocab_total_rejected(self):
        vocab, layout = assets()
        vocab["total"] = 854.0
        self.reject("total must be an integer", vocab, layout)


class TestLayoutConsistency(GeneratorCase):
    def test_duplicate_input_id_rejected(self):
        # Both words still print. What collapses is the next-step state: after
        # either class the model sees the same token, so whatever follows is
        # identical no matter which of the two it chose.
        vocab, layout = assets()
        layout["out2in"][20] = layout["out2in"][19]
        self.reject("reuses input id", vocab, layout)

    def test_swapped_appended_ids_rejected(self):
        # The gap every other check misses. Both ids stay unique, both stay in
        # range, the totals still agree. On the canonical layout this is the
        # 'adjustment' and 'adjustments' pair trading 7348 and 7349: each class
        # prints its own word and then feeds the other one's token forward.
        vocab, layout = assets()
        layout["out2in"][20], layout["out2in"][21] = (
            layout["out2in"][21], layout["out2in"][20])
        self.reject("not in class order", vocab, layout)

    def test_id_at_or_above_total_rejected(self):
        vocab, layout = assets()
        layout["out2in"][20] = layout["total"]
        self.reject(r"outside \[0,", vocab, layout)

    def test_negative_id_rejected(self):
        vocab, layout = assets()
        layout["out2in"][20] = -1
        self.reject(r"outside \[0,", vocab, layout)

    def test_total_not_equal_to_bpe_vocab_plus_new_rows_rejected(self):
        vocab, layout = assets()
        layout["total"] += 1
        self.reject("does not add up", vocab, layout)

    def test_read_vocabulary_too_large_for_uint16_rejected(self):
        vocab, layout = assets(bpe_vocab=0x10000)
        self.reject("does not fit uint16_t", vocab, layout)


class TestMissingInputs(GeneratorCase):
    """The defaults live under the ignored artifacts/, so absent is the normal
    state of a fresh clone rather than an exotic error."""

    def test_missing_vocab_names_what_was_expected(self):
        _, layout_path = self.write(*assets())
        with self.assertRaisesRegex(SystemExit, "output vocabulary not found"):
            gen.generate(self.dir / "absent.json", layout_path, self.dir / "out")

    def test_missing_layout_names_what_was_expected(self):
        vocab_path, _ = self.write(*assets())
        with self.assertRaisesRegex(SystemExit, "layout not found"):
            gen.generate(vocab_path, self.dir / "absent.json", self.dir / "out")


class TestSpecialTokens(GeneratorCase):
    def test_missing_special_rejected(self):
        vocab, layout = assets()
        vocab["tokens"][1]["token"] = "beans"
        self.reject("must be", vocab, layout)

    def test_reordered_specials_rejected(self):
        vocab, layout = assets()
        vocab["tokens"][1]["token"], vocab["tokens"][2]["token"] = "<eos>", "<bos>"
        self.reject("must be", vocab, layout)


if __name__ == "__main__":
    unittest.main()
