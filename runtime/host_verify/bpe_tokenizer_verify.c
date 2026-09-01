// Verifies the BTK1 loader and the ASCII ByteLevel-BPE encoder against a
// hand-built asset, so the checks are readable and need no model or corpus.
//
// This covers the encoder's own contract: header validation, byte lookup,
// ranked merging, pre-token splitting and input rejection. Agreement with the
// Hugging Face tokenizer over the real vocabulary is a separate gate.
//
//   cc -O3 -Wall -Wextra -o /tmp/btk runtime/host_verify/bpe_tokenizer_verify.c
//   /tmp/btk

#include <stdio.h>
#include <string.h>
#include "../bpe_tokenizer.h"

static int failures = 0;

static void check(const char *what, int ok, const char *detail) {
  printf("  %-46s %s%s%s\n", what, ok ? "ok" : "FAIL",
         detail && *detail ? "  " : "", detail ? detail : "");
  if (!ok) failures++;
}

// --- fixture ----------------------------------------------------------------
// Byte symbol ids are the byte values themselves, so 'a' is 97. Three merges,
// deliberately not in rank order once sorted by key, so a search that returned
// table position instead of rank would be caught:
//
//   rank 0  'a' + 'b'   -> 256      key 0x00610062
//   rank 1  "ab" + 'c'  -> 257      key 0x01000063
//   rank 2  ' ' + 'a'   -> 258      key 0x00200061
//
// sorted by key: rank 2, rank 0, rank 1.
#define FIX_MERGES 3
#define FIX_BASE 256
#define FIX_BYTES (BTK_HEADER_BYTES + BTK_BYTE_TABLE_BYTES + \
                   FIX_MERGES * BTK_MERGE_ENTRY_BYTES)

static uint8_t fixture[FIX_BYTES];

static void put_u32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put_u16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}

static void build_fixture(void) {
  memset(fixture, 0, sizeof(fixture));
  memcpy(fixture, "BTK1", 4);
  put_u32(fixture + 4, BTK_FORMAT_VERSION);
  put_u32(fixture + 8, 259);              /* 256 bytes + 3 merge results */
  put_u32(fixture + 12, FIX_MERGES);
  put_u32(fixture + 16, 0xDEADBEEF);      /* reserved: must be ignored */
  put_u32(fixture + 20, FIX_BASE);
  for (int b = 0; b < 256; b++)
    put_u16(fixture + BTK_HEADER_BYTES + b * 2, (uint16_t)b);

  uint8_t *m = fixture + BTK_HEADER_BYTES + BTK_BYTE_TABLE_BYTES;
  struct { uint32_t key; uint16_t rank; } e[FIX_MERGES] = {
    {(' ' << 16) | 'a', 2},
    {('a' << 16) | 'b', 0},
    {(256u << 16) | 'c', 1},
  };
  for (int i = 0; i < FIX_MERGES; i++) {
    put_u32(m + i * BTK_MERGE_ENTRY_BYTES, e[i].key);
    put_u16(m + i * BTK_MERGE_ENTRY_BYTES + 4, e[i].rank);
  }
}

// --- helpers ----------------------------------------------------------------
static int encode(const BpeTokenizer *tok, const char *text, uint16_t *out) {
  return bpe_encode_ascii(tok, text, out, BTK_MAX_INPUT_BYTES);
}

static int ids_equal(const uint16_t *got, int n, const uint16_t *want, int wn) {
  return n == wn && memcmp(got, want, (size_t)n * sizeof(uint16_t)) == 0;
}

static void check_encode(const BpeTokenizer *tok, const char *what,
                         const char *text, const uint16_t *want, int wn) {
  uint16_t got[BTK_MAX_INPUT_BYTES];
  char detail[160];
  int n = encode(tok, text, got);
  int ok = n >= 0 && ids_equal(got, n, want, wn);
  int off = snprintf(detail, sizeof(detail), "got");
  for (int i = 0; i < n && i < 8 && off > 0 && off < (int)sizeof(detail); i++)
    off += snprintf(detail + off, sizeof(detail) - (size_t)off, " %u", got[i]);
  if (n < 0) snprintf(detail, sizeof(detail), "error %d", n);
  check(what, ok, ok ? "" : detail);
}

int main(void) {
  build_fixture();
  BpeTokenizer tok;
  printf("BTK1 loader and ASCII encoder\n");

  // --- header validation ----------------------------------------------------
  check("valid asset loads", bpe_tokenizer_load(fixture, sizeof(fixture), &tok) == 0, "");
  check("active vocab read", tok.active_vocab == 259, "");
  check("merge count read", tok.merge_count == FIX_MERGES, "");
  check("merge base read", tok.merge_base == FIX_BASE, "");

  uint8_t bad[FIX_BYTES];
  memcpy(bad, fixture, sizeof(bad));
  bad[0] = 'X';
  check("bad magic rejected", bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 4, 1);
  check("format version 1 rejected", bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");
  put_u32(bad + 4, 3);
  check("format version 3 rejected", bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 20, 0);
  check("merge base 0 rejected", bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 12, 0x20000);
  check("absurd merge count rejected", bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");
  put_u32(bad + 12, 0);
  check("zero merge count rejected", bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 8, 0x20000);
  check("absurd vocab size rejected", bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  // --- null and size guards -------------------------------------------------
  check("null asset rejected",
        bpe_tokenizer_load(NULL, sizeof(fixture), &tok) == -1, "");
  check("null tokenizer rejected",
        bpe_tokenizer_load(fixture, sizeof(fixture), NULL) == -1, "");
  check("truncated header rejected",
        bpe_tokenizer_load(fixture, BTK_HEADER_BYTES - 1, &tok) == -1, "");
  check("truncated byte table rejected",
        bpe_tokenizer_load(fixture, BTK_HEADER_BYTES + BTK_BYTE_TABLE_BYTES - 1,
                           &tok) == -1, "");
  check("truncated merge table rejected",
        bpe_tokenizer_load(fixture, sizeof(fixture) - 1, &tok) == -1, "");
  check("exact size accepted",
        bpe_tokenizer_load(fixture, sizeof(fixture), &tok) == 0, "");
  check("larger buffer accepted",
        bpe_tokenizer_load(fixture, sizeof(fixture) + 64, &tok) == 0, "");
  {
    // The case that motivated the length argument: a header-and-byte-table
    // asset claiming a full merge table. The counts alone look sane, so only
    // the size check stops btk_merge_rank reading past the end of the image.
    uint8_t stub[BTK_HEADER_BYTES + BTK_BYTE_TABLE_BYTES];
    memcpy(stub, fixture, sizeof(stub));
    put_u32(stub + 8, 0x10000);           /* active_vocab 65536 */
    put_u32(stub + 12, 0xFF00);           /* claims a table it does not carry */
    put_u32(stub + 20, 256);
    check("asset claiming an absent merge table rejected",
          bpe_tokenizer_load(stub, sizeof(stub), &tok) == -1, "");
  }

  // --- header relationships -------------------------------------------------
  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 8, 255);
  check("active vocab below 256 rejected",
        bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 20, 259);                 /* equal to active_vocab */
  check("merge base at active vocab rejected",
        bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");
  put_u32(bad + 20, 300);                 /* beyond active_vocab */
  check("merge base beyond active vocab rejected",
        bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 8, 258);                  /* base 256 plus 3 merges needs 259 */
  check("merge results past active vocab rejected",
        bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  memcpy(bad, fixture, sizeof(bad));
  put_u32(bad + 20, 257);                 /* 257 + 3 exceeds 259 */
  check("merge base leaving too little room rejected",
        bpe_tokenizer_load(bad, sizeof(bad), &tok) == -1, "");

  // A rejected asset must not leave anything behind.
  {
    BpeTokenizer before, after;
    memset(&before, 0xAA, sizeof(before));
    memcpy(&after, &before, sizeof(after));
    memcpy(bad, fixture, sizeof(bad));
    put_u32(bad + 12, 0);                 /* fails after the header is read */
    check("rejected asset leaves the tokenizer untouched",
          bpe_tokenizer_load(bad, sizeof(bad), &after) == -1 &&
          memcmp(&before, &after, sizeof(before)) == 0, "");
  }

  // The reserved word must not reach the struct at all. Load two assets that
  // differ only there and require the results to be byte-identical, padding
  // included, which a stored copy of the field would break.
  {
    BpeTokenizer a, b;
    memset(&a, 0xAA, sizeof(a));   /* so padding is identical in both */
    memset(&b, 0xAA, sizeof(b));
    memcpy(bad, fixture, sizeof(bad));
    /* One buffer, mutated in place, so every derived pointer is identical and
     * the only difference between the two loads is the reserved word. */
    put_u32(bad + 16, 0xDEADBEEF);
    int ok_a = bpe_tokenizer_load(bad, sizeof(bad), &a) == 0;
    put_u32(bad + 16, 0);
    int ok_b = bpe_tokenizer_load(bad, sizeof(bad), &b) == 0;
    check("reserved word never reaches the struct",
          ok_a && ok_b && memcmp(&a, &b, sizeof(a)) == 0, "");
  }

  if (bpe_tokenizer_load(fixture, sizeof(fixture), &tok) != 0) {
    printf("fixture failed to load, cannot continue\n");
    return 1;
  }

  // --- table lookups --------------------------------------------------------
  check("byte id lookup", btk_byte_id(&tok, 'a') == 'a' &&
        btk_byte_id(&tok, 0) == 0 && btk_byte_id(&tok, 255) == 255, "");
  check("merge rank found, first by key",
        btk_merge_rank(&tok, ' ', 'a') == 2, "");
  check("merge rank found, middle by key",
        btk_merge_rank(&tok, 'a', 'b') == 0, "");
  check("merge rank found, last by key",
        btk_merge_rank(&tok, 256, 'c') == 1, "");
  check("absent pair returns -1", btk_merge_rank(&tok, 'x', 'y') == -1, "");
  check("absent pair below every key", btk_merge_rank(&tok, 0, 1) == -1, "");
  check("absent pair above every key",
        btk_merge_rank(&tok, 0xFFFF, 0xFFFF) == -1, "");

  // --- encoding -------------------------------------------------------------
  {
    const uint16_t want[] = {256};
    check_encode(&tok, "single merge applies", "ab", want, 1);
  }
  {
    // 'a'+'b' is rank 0 so it applies first, then "ab"+'c' is rank 1.
    const uint16_t want[] = {257};
    check_encode(&tok, "merges chain by rank", "abc", want, 1);
  }
  {
    const uint16_t want[] = {'d'};
    check_encode(&tok, "unmerged byte passes through", "d", want, 1);
  }
  {
    const uint16_t want[] = {258};
    check_encode(&tok, "leading space joins its word", " a", want, 1);
  }
  {
    // "a" and " b" are separate pre-tokens, so no pair spans the boundary.
    const uint16_t want[] = {'a', ' ', 'b'};
    check_encode(&tok, "merges do not cross a pre-token boundary",
                 "a b", want, 3);
  }
  {
    const uint16_t want[] = {'i', 't', '\'', 's'};
    check_encode(&tok, "contraction splits as its own piece", "it's", want, 4);
  }
  {
    const uint16_t want[] = {'1', '2', ' ', '3'};
    check_encode(&tok, "digits split from letters and take a space",
                 "12 3", want, 4);
  }
  {
    const uint16_t want[] = {'!', '?'};
    check_encode(&tok, "punctuation runs group together", "!?", want, 2);
  }
  {
    const uint16_t want[] = {0};
    check_encode(&tok, "empty input encodes to nothing", "", want, 0);
  }

  // --- input rejection ------------------------------------------------------
  {
    uint16_t out[BTK_MAX_INPUT_BYTES];
    char high[4] = {'a', (char)0xC3, (char)0xA9, 0};
    check("non-ASCII rejected", encode(&tok, high, out) == BTK_ERR_NOT_ASCII, "");

    char long_input[BTK_MAX_INPUT_BYTES + 8];
    memset(long_input, 'a', sizeof(long_input) - 1);
    long_input[sizeof(long_input) - 1] = 0;
    check("oversized input rejected",
          encode(&tok, long_input, out) == BTK_ERR_TOO_LONG, "");

    char at_limit[BTK_MAX_INPUT_BYTES + 1];
    memset(at_limit, 'a', BTK_MAX_INPUT_BYTES);
    at_limit[BTK_MAX_INPUT_BYTES] = 0;
    check("input exactly at the limit is accepted",
          encode(&tok, at_limit, out) >= 0, "");

    check("output capacity respected",
          bpe_encode_ascii(&tok, "x y z", out, 2) == BTK_ERR_CAPACITY, "");
  }

  printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures,
         failures == 1 ? "" : "s");
  return failures != 0;
}
