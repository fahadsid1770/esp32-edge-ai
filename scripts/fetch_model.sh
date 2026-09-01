#!/usr/bin/env bash
# Download a released model's inference assets and verify them against the
# hashes recorded here, then install them under artifacts/<model>/.
#
# This script never compiles, flashes, or touches the board. Fetching and device
# mutation are kept apart so that deploy.sh never reaches the network and a
# download can be repeated or audited without risking what is installed.
# pipefail: the download is piped, and `set -e` alone does not see a failure on
# the left of a pipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat >&2 <<'EOF'
usage: scripts/fetch_model.sh <tinystories|barista>

Downloads that model's released files. The inference assets are checked against
a SHA-256 and byte size pinned in this script; metadata.json carries no pinned
hash, so it is parsed and cross-checked against those same pins instead. Only
then is anything installed into artifacts/<model>/.

Nothing is installed unless every check passes, so a failed or partial download
leaves whatever is already in artifacts/ untouched.

Then flash with:
  scripts/deploy.sh <tinystories|barista>
EOF
  exit 2
}

# Argument first, before any download, verification, or install.
[ $# -eq 1 ] || usage
MODEL_KIND=$1

# --- released files, pinned ---------------------------------------------------
# One line per file: name sha256 bytes. These are the values published with the
# release; a mismatch means the download is not the release, so it is refused
# rather than reported.
case "$MODEL_KIND" in
  tinystories)
    REPO=slvDev/esp32-ai-tinystories
    PINNED=(
      "model.bin 1d8326c05c383ccfa615f5455575802817cb453dbc7ab28875d41a9dbb45477e 14912348"
      "tokenizer.json 4e28163669f2249af31a528a54fc25064dcbd0a34edbfa7bedb16d2d600ec7ae 1788896"
    )
    ;;
  barista)
    REPO=slvDev/esp32-ai-barista
    PINNED=(
      "model.bin 1359a1cb74de4143d630c2c192990de814cd47255bcdfa9cc135f07ef0a39fc4 4600186"
      "tokenizer.json 0ad085811c949f35c5f5f15b555f2ff2d46ec1ec94a1650552416d06aaa19ee2 491735"
      "vocab.json 5a16d6224abf03265d69ebcccf121c8f8d2c222bfedd8274acbc0bbbe13e4eb7 50494"
      "layout.json 15036c5ee2b23b9b35404ef6422cb788bbede8cf2c269bae35d4dbf9a48a0b90 5142"
    )
    ;;
  *)
    usage
    ;;
esac

# metadata.json travels with the assets but carries no pinned hash: it records
# the release commit, which is not known when these values are written. It is
# fetched, parsed, and cross-checked against the pins below.
NAMES=()
for entry in "${PINNED[@]}"; do NAMES+=("${entry%% *}"); done
NAMES+=("metadata.json")

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need hf
need shasum
need python3   # the metadata cross-check below is a python snippet

DEST="artifacts/$MODEL_KIND"

# Download into a scratch directory so a failure cannot leave artifacts/ holding
# a half-written or unverified file.
STAGING_PARENT=${TMPDIR:-/tmp}
STAGING_PARENT=${STAGING_PARENT%/}
STAGING=$(mktemp -d "$STAGING_PARENT/esp32ai-fetch-XXXXXX")
# Removes only the directory this invocation created. The name and parent are
# checked first so the path can only be the one mktemp returned, and a failed
# removal is reported rather than swallowed.
cleanup() {
  case "$STAGING" in
    "$STAGING_PARENT"/esp32ai-fetch-??????) ;;
    *) echo "not removing unexpected staging path: $STAGING" >&2; return 0 ;;
  esac
  [ -d "$STAGING" ] || return 0
  rm -rf -- "$STAGING" || echo "could not remove staging directory: $STAGING" >&2
}
trap cleanup EXIT

echo "=== download $REPO ==="
hf download "$REPO" "${NAMES[@]}" --local-dir "$STAGING" 2>&1 | tail -3

echo "=== verify ==="
failed=0
for entry in "${PINNED[@]}"; do
  read -r name want_sha want_bytes <<<"$entry"
  path="$STAGING/$name"
  if [ ! -f "$path" ]; then
    echo "  $name MISSING from the download" >&2
    failed=1
    continue
  fi
  got_bytes=$(wc -c < "$path" | tr -d ' ')
  got_sha=$(shasum -a 256 "$path" | cut -d' ' -f1)
  if [ "$got_bytes" != "$want_bytes" ]; then
    echo "  $name size $got_bytes, expected $want_bytes" >&2
    failed=1
  elif [ "$got_sha" != "$want_sha" ]; then
    echo "  $name sha256 $got_sha, expected $want_sha" >&2
    failed=1
  else
    printf "  %-16s ok  %s B\n" "$name" "$got_bytes"
  fi
done

# The release's own metadata must agree with the pins above. The pinned bytes
# stay authoritative either way; a disagreement means the release is internally
# inconsistent, which is reported rather than worked around.
if [ -f "$STAGING/metadata.json" ]; then
  if ! python3 - "$STAGING/metadata.json" "${PINNED[@]}" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    meta = json.load(fh)
files = meta.get("files", {})
bad = 0
for entry in sys.argv[2:]:
    name, sha, size = entry.split()
    rec = files.get(name)
    if rec is None:
        print(f"  metadata.json does not describe {name}", file=sys.stderr); bad = 1
    elif rec.get("sha256") != sha or str(rec.get("bytes")) != size:
        print(f"  metadata.json disagrees with the pinned {name}", file=sys.stderr); bad = 1
sys.exit(bad)
PY
  then
    failed=1
  else
    echo "  metadata.json  ok  agrees with the pinned values"
  fi
else
  echo "  metadata.json MISSING from the download" >&2
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  echo "verification failed; $DEST/ was not modified" >&2
  exit 1
fi

# Everything verified: install. Existing files are replaced only at this point.
mkdir -p "$DEST"
for name in "${NAMES[@]}"; do
  mv "$STAGING/$name" "$DEST/$name"
done

echo
echo "installed into $DEST/:"
for name in "${NAMES[@]}"; do printf "  %s\n" "$name"; done
echo
echo "flash it with:"
echo "  scripts/deploy.sh $MODEL_KIND"
