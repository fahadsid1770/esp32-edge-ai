"""The Barista tokenizer header generator.

A wrong encoder table can silently change the token ids presented to the model,
without crashing or logging. So these tests are about the asset being exactly
what runtime/bpe_tokenizer.h expects, and about rejecting any tokenizer
configuration the device cannot reproduce.

The canonical tokenizer.json is distributed on Hugging Face and is not committed
to Git, so every fixture here is synthetic and the tests do not require fetched
HF assets. The byte mapping is checked against known GPT-2 values rather than
against the generator's own function, so the fixture cannot certify itself.

  uv run python -m unittest discover -s tests
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

GENERATOR = (Path(__file__).resolve().parents[1] / "firmware" / "esp32_barista"
             / "tools" / "generate_tokenizer_header.py")
# Sketch tools are build steps for one board, not a library, so they are not
# packaged. Load by path.
_spec = importlib.util.spec_from_file_location("barista_generate_tokenizer_header",
                                               GENERATOR)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# Byte -> symbol for the 256 single-byte tokens, taken from the generator so the
# fixture is shaped like a real ByteLevel vocabulary. TestByteMapping pins the
# mapping itself against known GPT-2 values, so this is not self-certifying.
SYMBOL = gen.BYTE_SYMBOLS


def byte_level(**overrides):
    pre = {"type": "ByteLevel", "add_prefix_space": False,
           "trim_offsets": True, "use_regex": True}
    pre.update(overrides)
    return pre


def tokenizer(merges=None, vocab_extra=None, drop_symbol=None, model=None,
              **top_level):
    """A minimal but structurally valid ByteLevel BPE tokenizer.json.

    Ids are kept dense, which the generator requires, so a test that wants a
    gap has to introduce one deliberately through vocab_extra.
    """
    if merges is None:
        merges = ["a b", "ab c", "d e"]
    symbols = [SYMBOL[b] for b in range(256) if b != drop_symbol]
    vocab = {symbol: i for i, symbol in enumerate(symbols)}
    for rank, merge in enumerate(merges):
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        vocab.setdefault(left + right, len(symbols) + rank)
    if vocab_extra:
        vocab.update(vocab_extra)

    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": byte_level(),
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": vocab, "merges": merges,
        },
    }
    if model:
        config["model"].update(model)
    config.update(top_level)
    return config


def parse(header_text):
    declared = int(re.search(r"TOKENIZER_ENCODER_ASSET\[(\d+)\]", header_text).group(1))
    body = header_text.split("= {", 1)[1].split("};", 1)[0]
    return declared, bytes(int(b) for b in re.findall(r"\d+", body))


def fields(asset):
    magic, version, vocab_size, merges, reserved, base, sha = struct.unpack(
        gen.HEADER_STRUCT, asset[:struct.calcsize(gen.HEADER_STRUCT)])
    return dict(magic=magic, version=version, vocab=vocab_size, merges=merges,
                reserved=reserved, base=base, sha=sha)


class GeneratorCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, config):
        path = self.dir / "tokenizer.json"
        path.write_text(json.dumps(config))
        return path

    def run_generator(self, config, out=None):
        path = self.write(config)
        with contextlib.redirect_stdout(io.StringIO()):
            return gen.generate(path, out or self.dir / "generated" / "tok.h")

    def build(self, config, out=None):
        return parse(self.run_generator(config, out).read_text())

    def reject(self, pattern, config):
        with self.assertRaisesRegex(SystemExit, pattern):
            self.run_generator(config)


class TestByteMapping(unittest.TestCase):
    """Pinned against published GPT-2 values, not against the generator."""

    def test_known_symbols(self):
        # Printable ASCII maps to itself; everything else is lifted above 0xFF
        # in byte order, which is why space becomes the familiar U+0120.
        self.assertEqual(SYMBOL[0x21], "!")
        self.assertEqual(SYMBOL[0x7E], "~")
        self.assertEqual(SYMBOL[0x20], "Ġ")
        self.assertEqual(SYMBOL[0x00], "Ā")
        self.assertEqual(SYMBOL[0x7F], "ġ")
        self.assertEqual(SYMBOL[0xAD], "Ń")

    def test_is_a_bijection_over_every_byte(self):
        self.assertEqual(len(SYMBOL), 256)
        self.assertEqual(len(set(SYMBOL.values())), 256)


class TestAssetLayout(GeneratorCase):
    def test_declared_size_matches_the_payload(self):
        declared, asset = self.build(tokenizer())
        self.assertEqual(declared, len(asset))

    def test_size_follows_the_layout(self):
        _, asset = self.build(tokenizer())
        head = struct.calcsize(gen.HEADER_STRUCT)
        self.assertEqual(head, 52)
        self.assertEqual(len(asset), head + 512 + 6 * fields(asset)["merges"])

    def test_header_fields(self):
        _, asset = self.build(tokenizer())
        f = fields(asset)
        self.assertEqual(f["magic"], b"BTK1")
        self.assertEqual(f["version"], 2)
        self.assertEqual(f["vocab"], 256 + 3)
        self.assertEqual(f["merges"], 3)
        self.assertEqual(f["base"], 256)

    def test_offset_16_is_reserved_and_zero(self):
        # Not an end-of-sequence id. Barista's end of answer is output class 2
        # in the word tables, and canonical token id 0 is "!".
        _, asset = self.build(tokenizer())
        self.assertEqual(fields(asset)["reserved"], 0)
        self.assertEqual(gen.RESERVED, 0)

    def test_sha256_records_the_source_file(self):
        config = tokenizer()
        path = self.write(config)
        _, asset = parse(self.run_generator(config).read_text())
        self.assertEqual(fields(asset)["sha"],
                         hashlib.sha256(path.read_bytes()).digest()[:28])

    def test_byte_table_maps_every_byte_to_its_id(self):
        _, asset = self.build(tokenizer())
        head = struct.calcsize(gen.HEADER_STRUCT)
        table = struct.unpack("<256H", asset[head:head + 512])
        self.assertEqual(list(table), list(range(256)))

    def test_merge_entries_are_sorted_and_unique(self):
        # The runtime binary searches this table. Unsorted, it silently misses
        # merges and the device tokenizes differently from Python.
        _, asset = self.build(tokenizer())
        start = struct.calcsize(gen.HEADER_STRUCT) + 512
        keys = [struct.unpack("<I", asset[start + i * 6:start + i * 6 + 4])[0]
                for i in range(fields(asset)["merges"])]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))

    def test_rank_is_preserved_through_the_sort(self):
        _, asset = self.build(tokenizer())
        start = struct.calcsize(gen.HEADER_STRUCT) + 512
        found = {}
        for i in range(fields(asset)["merges"]):
            key, rank = struct.unpack("<IH", asset[start + i * 6:start + i * 6 + 6])
            found[rank] = (key >> 16, key & 0xFFFF)
        self.assertEqual(found[0], (ord("a"), ord("b")))
        self.assertEqual(found[1], (256, ord("c")))


class TestHeaderText(GeneratorCase):
    def test_progmem_fallback_is_emitted(self):
        # Defined by the Arduino core, absent on a host compiler, and this
        # header is included by both.
        text = self.run_generator(tokenizer()).read_text()
        self.assertIn("#ifndef PROGMEM\n#define PROGMEM\n#endif\n", text)

    def test_declares_the_size_constant(self):
        path = self.run_generator(tokenizer())
        _, asset = parse(path.read_text())
        self.assertIn(f"#define TOKENIZER_ENCODER_ASSET_SIZE {len(asset)}",
                      path.read_text())

    def test_creates_a_missing_output_directory(self):
        out = self.dir / "absent" / "generated" / "tok.h"
        self.assertTrue(self.run_generator(tokenizer(), out=out).exists())

    def test_output_is_deterministic(self):
        config = tokenizer()
        first = self.run_generator(config).read_bytes()
        second = self.run_generator(config, out=self.dir / "again" / "t.h").read_bytes()
        self.assertEqual(first, second)


class TestVocabularyIds(GeneratorCase):
    """Ids must be a bijection onto 0..active_vocab-1: the asset stores only a
    count, and merge results are derived from rank plus base."""

    def test_boolean_id_rejected(self):
        self.reject("not an integer", tokenizer(vocab_extra={"BOOL": True}))

    def test_fractional_id_rejected(self):
        self.reject("not an integer", tokenizer(vocab_extra={"FRAC": 1.5}))

    def test_duplicate_id_rejected(self):
        self.reject("shared by more than one symbol",
                    tokenizer(vocab_extra={"DUP": 0}))

    def test_gapped_ids_rejected(self):
        self.reject("dense range", tokenizer(vocab_extra={"GAP": 999}))

    def test_oversized_unused_id_rejected(self):
        self.reject("dense range", tokenizer(vocab_extra={"BIG": 70000}))

    def test_vocabulary_too_large_for_uint16(self):
        config = tokenizer(merges=["a b"])
        vocab = {SYMBOL[b]: b for b in range(256)}
        vocab["ab"] = 256
        vocab.update({f"f{i}": 257 + i for i in range(gen.UINT16_LIMIT - 256)})
        config["model"]["vocab"] = vocab
        self.reject("does not fit uint16_t", config)


class TestEncodingContract(GeneratorCase):
    """The device implements one configuration. Anything else would tokenize
    differently on device than in Python."""

    def test_non_bpe_model_rejected(self):
        self.reject("implements BPE only", tokenizer(model={"type": "WordPiece"}))

    def test_normalizer_rejected(self):
        self.reject("normalizer is configured", tokenizer(normalizer={"type": "NFD"}))

    def test_added_tokens_rejected(self):
        self.reject("added tokens are configured",
                    tokenizer(added_tokens=[{"id": 0, "content": "<pad>"}]))

    def test_truncation_rejected(self):
        self.reject("truncation is configured", tokenizer(truncation={"max_length": 128}))

    def test_padding_rejected(self):
        self.reject("padding is configured", tokenizer(padding={"strategy": "BatchLongest"}))

    def test_post_processor_rejected(self):
        self.reject("post_processor is configured",
                    tokenizer(post_processor={"type": "ByteLevel"}))

    def test_non_byte_level_pre_tokenizer_rejected(self):
        self.reject("hardcodes ByteLevel", tokenizer(pre_tokenizer={"type": "Whitespace"}))

    def test_missing_pre_tokenizer_rejected(self):
        self.reject("hardcodes ByteLevel", tokenizer(pre_tokenizer=None))

    def test_add_prefix_space_rejected(self):
        self.reject("add_prefix_space is True",
                    tokenizer(pre_tokenizer=byte_level(add_prefix_space=True)))

    def test_use_regex_disabled_rejected(self):
        self.reject("use_regex is False",
                    tokenizer(pre_tokenizer=byte_level(use_regex=False)))

    def test_dropout_rejected(self):
        self.reject("model dropout", tokenizer(model={"dropout": 0.1}))

    def test_unk_token_rejected(self):
        self.reject("model unk_token", tokenizer(model={"unk_token": "<unk>"}))

    def test_continuing_subword_prefix_rejected(self):
        self.reject("continuing_subword_prefix",
                    tokenizer(model={"continuing_subword_prefix": "##"}))

    def test_end_of_word_suffix_rejected(self):
        self.reject("end_of_word_suffix", tokenizer(model={"end_of_word_suffix": "</w>"}))

    def test_byte_fallback_rejected(self):
        self.reject("byte_fallback", tokenizer(model={"byte_fallback": True}))

    def test_ignore_merges_rejected(self):
        self.reject("ignore_merges", tokenizer(model={"ignore_merges": True}))

    def test_disabled_settings_spelled_as_empty_are_accepted(self):
        # tokenizers writes null, but an empty string means the same thing and
        # must not be mistaken for a configured prefix.
        self.build(tokenizer(model={"continuing_subword_prefix": "",
                                    "end_of_word_suffix": "", "dropout": 0}))


class TestRawAsset(GeneratorCase):
    """--asset writes the same bytes the header embeds, so the host check and
    the firmware cannot be looking at different tables."""

    def test_raw_asset_matches_the_embedded_bytes(self):
        path = self.write(tokenizer())
        header = self.dir / "generated" / "tok.h"
        raw = self.dir / "raw" / "tokenizer.btk"
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate(path, header, raw)
        declared, embedded = parse(header.read_text())
        self.assertTrue(raw.exists())            # its directory was created too
        self.assertEqual(raw.read_bytes(), embedded)
        self.assertEqual(len(raw.read_bytes()), declared)

    def test_asset_is_not_written_unless_asked(self):
        path = self.write(tokenizer())
        header = self.dir / "generated" / "tok.h"
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate(path, header)
        self.assertEqual(list(header.parent.iterdir()), [header])


class TestMergeForms(GeneratorCase):
    def test_merges_may_be_pairs_instead_of_strings(self):
        # Newer tokenizer.json files store merges as two-element lists. The
        # tables must come out identical; only the recorded sha differs, since
        # that covers the source text rather than the tables.
        def tables(asset):
            return asset[:24] + asset[struct.calcsize(gen.HEADER_STRUCT):]

        as_strings = self.build(tokenizer())[1]
        as_pairs = self.build(tokenizer(merges=[["a", "b"], ["ab", "c"], ["d", "e"]]))[1]
        self.assertNotEqual(as_strings, as_pairs)
        self.assertEqual(tables(as_strings), tables(as_pairs))


class TestMergeRejections(GeneratorCase):
    def test_missing_tokenizer_file(self):
        with self.assertRaisesRegex(SystemExit, "tokenizer not found"):
            gen.generate(self.dir / "absent.json", self.dir / "out.h")

    def test_missing_byte_symbol(self):
        self.reject("missing from the vocabulary", tokenizer(drop_symbol=0x00))

    def test_no_merges(self):
        self.reject("no merges", tokenizer(merges=[]))

    def test_repeated_pair_rejected_before_the_sequential_check(self):
        # One pair cannot carry two ranks. This is caught on the literal pair,
        # not as a downstream consequence of the result ids.
        with self.assertRaisesRegex(SystemExit, "cannot have two ranks"):
            self.run_generator(tokenizer(merges=["a b", "a b"]))

    def test_non_sequential_merge_results(self):
        config = tokenizer(merges=["a b", "d e"])
        config["model"]["vocab"]["ab"] = 257
        config["model"]["vocab"]["de"] = 256
        self.reject("must be sequential", config)

    def test_merge_result_absent_from_vocabulary(self):
        config = tokenizer(merges=["a b"])
        del config["model"]["vocab"]["ab"]
        self.reject("not in the vocabulary", config)

    def test_merge_operand_absent_from_vocabulary(self):
        self.reject("operand", tokenizer(merges=["a ZZZ"]))

    def test_merge_base_of_zero(self):
        # The runtime reads a stored base of 0 as "field absent" and uses 257.
        config = tokenizer(merges=["a b"])
        vocab = {SYMBOL[b]: b + 1 for b in range(256)}
        vocab["ab"] = 0
        config["model"]["vocab"] = vocab
        self.reject("merge base is 0", config)


if __name__ == "__main__":
    unittest.main()
