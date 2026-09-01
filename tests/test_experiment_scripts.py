"""Every published experiment names its own vocabulary.

RESULTS.md reports results at two vocabularies. A script that omits --vocab
takes train.py's default instead, which fails silently: no error, no missing
file, just valid numbers for a different configuration than its header claims.

Running these experiments takes GPU hours, so the scripts are read as text.
What is checked is that each one states a vocabulary, that the value is the
published one, and that its header says the same.

  uv run python -m unittest discover -s tests
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research" / "tinystories" / "scripts"

# Every script here trains, and each one belongs to a published table.
EXPECTED_VOCAB = {
    "run_small_vocab_ablation.sh": 4096,
    "run_table_sweep.sh": 4096,
    "run_deploy_ablation.sh": 32768,
}


def script_text(name):
    return (SCRIPTS / name).read_text()


def script_header(name):
    """The comment block between the shebang and the first command.

    That block is what a reader sees when deciding what a script produces. A
    comment further down does not serve that purpose, so it does not count."""
    lines = script_text(name).splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    header = []
    for line in lines:
        if not line.startswith("#"):
            break
        header.append(line)
    return "\n".join(header)


class ScriptsPinTheirVocabulary(unittest.TestCase):
    def test_every_training_script_is_covered(self):
        # A new script added without an entry here would otherwise be checked by
        # nothing at all.
        found = {p.name for p in SCRIPTS.glob("*.sh")
                 if "research.tinystories.train" in p.read_text()}
        self.assertEqual(found, set(EXPECTED_VOCAB),
                         "a training script is missing from EXPECTED_VOCAB")

    def test_vocabulary_is_explicit(self):
        for name in EXPECTED_VOCAB:
            with self.subTest(script=name):
                self.assertIn("--vocab", script_text(name),
                              f"{name} takes train.py's default vocabulary; a "
                              f"change to that default silently changes this "
                              f"experiment")

    def test_vocabulary_is_the_published_one(self):
        for name, vocab in EXPECTED_VOCAB.items():
            with self.subTest(script=name):
                found = set(re.findall(r"--vocab\s+(\d+)", script_text(name)))
                self.assertEqual(found, {str(vocab)},
                                 f"{name} should run at vocab {vocab}")

    def test_header_states_the_vocabulary(self):
        # The header has to agree with the flag rather than merely not
        # contradict it.
        for name, vocab in EXPECTED_VOCAB.items():
            with self.subTest(script=name):
                self.assertIn(str(vocab), script_header(name),
                              f"{name}'s header does not say it runs at vocab {vocab}")


class DocumentedCommandsPinTheirVocabulary(unittest.TestCase):
    def test_readme_train_examples_are_explicit(self):
        readme = (ROOT / "research" / "tinystories" / "README.md").read_text()
        blocks = re.findall(r"```bash\n(.*?)```", readme, re.S)
        examples = [b for b in blocks if "research.tinystories.train" in b]
        self.assertTrue(examples, "no train example found in the research README")
        for block in examples:
            with self.subTest(example=block.strip().splitlines()[0]):
                self.assertIn("--vocab", block)


if __name__ == "__main__":
    unittest.main()
