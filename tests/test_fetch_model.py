"""scripts/fetch_model.sh: which downloads are accepted, which are refused, and
what is left on disk afterwards.

Fixtures are synthetic and offline. Each test builds a throwaway repository
holding only the script, rewrites its pinned hashes to describe files it
generates, and puts a fake `hf` on PATH that copies a prepared directory into
--local-dir. TMPDIR is redirected per test so staging can be inspected.

  uv run python -m unittest discover -s tests
"""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_model.sh"

FAKE_HF = """#!/usr/bin/env bash
# Stands in for the huggingface CLI: copies a prepared directory into whatever
# --local-dir was asked for, so the script under test does real file handling.
dest=""; prev=""
for a in "$@"; do
  [ "$prev" = "--local-dir" ] && dest=$a
  prev=$a
done
mkdir -p "$dest"
cp -R "$FAKE_HF_SRC"/. "$dest"/ 2>/dev/null || true
echo "fake hf: copied into $dest"
"""


def pin_line(name, blob):
    return f'      "{name} {hashlib.sha256(blob).hexdigest()} {len(blob)}"'


def rewrite_pins(script, model, files):
    """Replace one model's PINNED block so it describes the synthetic files."""
    lines = script.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == f"{model})")
    open_at = next(i for i in range(start, len(lines)) if lines[i].strip() == "PINNED=(")
    close_at = next(i for i in range(open_at, len(lines)) if lines[i].strip() == ")")
    body = [pin_line(n, b) for n, b in files.items() if n != "metadata.json"]
    return "\n".join(lines[:open_at + 1] + body + lines[close_at:]) + "\n"


def metadata_for(files):
    return {
        "files": {
            n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
            for n, b in files.items() if n != "metadata.json"
        }
    }


class FetchCase(unittest.TestCase):
    MODEL = "barista"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.root = self.dir / "repo"
        (self.root / "scripts").mkdir(parents=True)
        self.remote = self.dir / "remote"
        self.remote.mkdir()
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self.tmpdir = self.dir / "tmp"
        self.tmpdir.mkdir()
        hf = self.bin / "hf"
        hf.write_text(FAKE_HF)
        hf.chmod(0o755)

    def publish(self, files, pins=None, metadata=None):
        """Write the fake remote, and pin the script to `pins` (default: files)."""
        full = dict(files)
        full["metadata.json"] = json.dumps(
            metadata if metadata is not None else metadata_for(files)).encode()
        for name, blob in full.items():
            (self.remote / name).write_bytes(blob)
        script = rewrite_pins(SCRIPT.read_text(), self.MODEL, pins or files)
        target = self.root / "scripts" / "fetch_model.sh"
        target.write_text(script)
        target.chmod(0o755)

    def run_fetch(self, *args):
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}:{env['PATH']}"
        env["FAKE_HF_SRC"] = str(self.remote)
        env["TMPDIR"] = str(self.tmpdir)
        return subprocess.run(
            ["bash", str(self.root / "scripts" / "fetch_model.sh"), *args],
            capture_output=True, text=True, env=env)

    def dest(self):
        return self.root / "artifacts" / self.MODEL

    def staging_dirs(self):
        return sorted(self.tmpdir.glob("esp32ai-fetch-*"))


class TestArguments(FetchCase):
    def setUp(self):
        super().setUp()
        self.publish({"model.bin": b"weights"})

    def test_missing_argument_exits_2(self):
        r = self.run_fetch()
        self.assertEqual(r.returncode, 2)
        self.assertIn("usage:", r.stderr)

    def test_unknown_model_exits_2(self):
        self.assertEqual(self.run_fetch("bogus").returncode, 2)

    def test_extra_arguments_exit_2(self):
        self.assertEqual(self.run_fetch("barista", "tinystories").returncode, 2)

    def test_a_rejected_argument_downloads_nothing(self):
        self.run_fetch("bogus")
        self.assertFalse(self.dest().exists())


class TestVerifiedDownload(FetchCase):
    def test_matching_files_are_installed(self):
        files = {"model.bin": b"weights", "tokenizer.json": b"{}"}
        self.publish(files)
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in (*files, "metadata.json"):
            with self.subTest(name=name):
                self.assertTrue((self.dest() / name).is_file())
        self.assertEqual((self.dest() / "model.bin").read_bytes(), b"weights")

    def test_it_points_at_the_deploy_step_rather_than_flashing(self):
        self.publish({"model.bin": b"weights"})
        r = self.run_fetch(self.MODEL)
        self.assertIn("scripts/deploy.sh barista", r.stdout)


class TestRefusals(FetchCase):
    def test_wrong_hash_is_refused(self):
        # Same length, different content: the size check passes, so only the
        # hash can reject this.
        self.publish({"model.bin": b"weights"}, pins={"model.bin": b"weightZ"})
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("sha256", r.stderr)
        self.assertNotIn("size", r.stderr.split("metadata")[0])

    def test_wrong_size_is_refused(self):
        # A size mismatch is reported as a size mismatch, not as a bad hash.
        self.publish({"model.bin": b"weights"}, pins={"model.bin": b"much longer weights"})
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("size", r.stderr)

    def test_missing_file_is_refused(self):
        # Pinned for two files, the remote serves one.
        self.publish({"model.bin": b"weights"},
                     pins={"model.bin": b"weights", "tokenizer.json": b"{}"})
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("MISSING", r.stderr)

    def test_missing_metadata_is_refused(self):
        files = {"model.bin": b"weights"}
        self.publish(files)
        (self.remote / "metadata.json").unlink()
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("metadata.json MISSING", r.stderr)

    def test_metadata_disagreeing_with_the_pins_is_refused(self):
        # The bytes verify against the pins, but the release's own record
        # describes something else.
        files = {"model.bin": b"weights"}
        wrong = {"files": {"model.bin": {"sha256": "0" * 64, "bytes": 7}}}
        self.publish(files, metadata=wrong)
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("disagrees", r.stderr)

    def test_metadata_omitting_a_file_is_refused(self):
        self.publish({"model.bin": b"weights"}, metadata={"files": {}})
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not describe", r.stderr)


class TestExistingArtifactsSurviveAFailure(FetchCase):
    def test_a_refused_download_changes_nothing(self):
        # A good install, then a failing fetch: the install must be unchanged.
        self.publish({"model.bin": b"good weights"})
        self.assertEqual(self.run_fetch(self.MODEL).returncode, 0)
        before = {p.name: p.read_bytes() for p in self.dest().iterdir()}

        self.publish({"model.bin": b"tampered"}, pins={"model.bin": b"good weights"})
        r = self.run_fetch(self.MODEL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("was not modified", r.stderr)

        after = {p.name: p.read_bytes() for p in self.dest().iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(after["model.bin"], b"good weights")


class TestStagingCleanup(FetchCase):
    def test_a_successful_fetch_leaves_no_staging_directory(self):
        self.publish({"model.bin": b"weights"})
        self.assertEqual(self.run_fetch(self.MODEL).returncode, 0)
        self.assertEqual(self.staging_dirs(), [])

    def test_a_refused_fetch_leaves_no_staging_directory(self):
        self.publish({"model.bin": b"weights"}, pins={"model.bin": b"weightZ"})
        self.assertEqual(self.run_fetch(self.MODEL).returncode, 1)
        self.assertEqual(self.staging_dirs(), [])

    def test_a_normal_run_trips_neither_guard(self):
        self.publish({"model.bin": b"weights"})
        r = self.run_fetch(self.MODEL)
        self.assertNotIn("not removing unexpected staging path", r.stderr)
        self.assertNotIn("could not remove", r.stderr)

    def test_an_unexpected_staging_path_is_left_alone(self):
        # Removal is keyed on the parent and the name. A directory that is not
        # the one mktemp was asked for must survive, whatever else happens.
        self.publish({"model.bin": b"weights"})
        unexpected = self.dir / "not-a-staging-dir"
        fake_mktemp = self.bin / "mktemp"
        fake_mktemp.write_text(
            "#!/usr/bin/env bash\n"
            f'mkdir -p "{unexpected}"\n'
            f'echo "{unexpected}"\n')
        fake_mktemp.chmod(0o755)
        r = self.run_fetch(self.MODEL)
        self.assertIn("not removing unexpected staging path", r.stderr)
        self.assertTrue(unexpected.is_dir(), "the guard removed a path it did not create")


class TestPinnedReleaseValues(unittest.TestCase):
    """Every pin must be a name, a 64-character digest, and a positive size."""

    def test_every_pin_is_a_name_a_sha256_and_a_size(self):
        found = 0
        for line in SCRIPT.read_text().splitlines():
            line = line.strip()
            if not (line.startswith('"') and line.endswith('"')):
                continue
            parts = line.strip('"').split()
            if len(parts) != 3:
                continue
            name, sha, size = parts
            with self.subTest(name=name):
                self.assertRegex(sha, r"^[0-9a-f]{64}$")
                self.assertTrue(size.isdigit() and int(size) > 0)
                found += 1
        self.assertGreaterEqual(found, 6)   # 2 tinystories + 4 barista


if __name__ == "__main__":
    unittest.main()
