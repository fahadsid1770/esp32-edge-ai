// Verifies int8 staging, scale alignment, ranged matvec equivalence, and
// layer/head hook dispatch - the code path the device runs, which verify.c
// (exact int4) does not reach.
//
// Staging unpacks the same nibbles and scales, so results must be bit-identical
// to the unstaged int8 path.
//
//   cc -O3 -DLLM_INT8_ACT=1 -o /tmp/sv runtime/host_verify/staging_verify.c -lm
//   /tmp/sv <model.bin>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../llm.h"

static int failures = 0;

static void check(const char *what, int ok, const char *detail) {
  printf("  %-46s %s%s%s\n", what, ok ? "ok" : "FAIL",
         detail && *detail ? "  " : "", detail ? detail : "");
  if (!ok) failures++;
}

// --- hook dispatch ----------------------------------------------------------
static int layer_hook_calls = 0, head_hook_calls = 0;

static void layer_hook(const QT *t, const float *x, float *y) {
  layer_hook_calls++;
  MATVEC(t, x, y);
}
static void head_hook(const QT *t, const float *x, float *y) {
  head_hook_calls++;
  MATVEC(t, x, y);
}

static double max_abs_diff(const float *a, const float *b, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  return m;
}

/* ---- synthetic untied model -------------------------------------------------
 * The exporter only ever writes TIED_HEAD, so without this the untied branch of
 * llm_load is unexecuted code. Build a tiny valid image with TIED_HEAD clear and
 * output_vocab != input_vocab, and check the loader binds a separate head at the
 * right offset. */
static size_t qt_bytes(int rows, int cols, int group) {
  int ng = (cols + group - 1) / group, rb = (cols + 1) / 2;
  return 4 + (size_t)rows * rb + (size_t)rows * ng * 2;
}

static uint8_t *put_qt(uint8_t *p, int rows, int cols, int group, uint8_t fill) {
  int ng = (cols + group - 1) / group, rb = (cols + 1) / 2;
  int32_t g = group; memcpy(p, &g, 4); p += 4;
  memset(p, fill, (size_t)rows * rb); p += (size_t)rows * rb;
  for (int i = 0; i < rows * ng; i++) { uint16_t h = 0x3C00; memcpy(p, &h, 2); p += 2; } /* 1.0 */
  return p;
}
static uint8_t *put_f(uint8_t *p, int n, float v) {
  for (int i = 0; i < n; i++) { memcpy(p, &v, 4); p += 4; }
  return p;
}

static void test_untied(void) {
  const int V = 16, VOUT = 6, D = 8, L = 1, H = 2, F = 4, P = 4, G = 128;
  size_t need = 56 + qt_bytes(V, D, G) + qt_bytes(L * P, D, G) + (size_t)P * 4
              + qt_bytes(V, L * P, G)
              + (size_t)D * 4 + qt_bytes(3 * D, D, G) + qt_bytes(D, D, G)
              + (size_t)D * 4 + qt_bytes(F, D, G) + qt_bytes(F, D, G)
              + qt_bytes(D, F, G) + qt_bytes(P, D, G) + qt_bytes(D, P, G)
              + (size_t)D * 4
              + (size_t)D * 4 + qt_bytes(VOUT, D, G);
  uint8_t *img = calloc(need, 1), *p = img;
  uint32_t hdr[4] = {0x00454C50u, 1u, 56u, 0u};   /* TIED_HEAD clear */
  memcpy(p, hdr, 16); p += 16;
  uint32_t vio[2] = {(uint32_t)V, (uint32_t)VOUT};
  memcpy(p, vio, 8); p += 8;
  int32_t cfg[7] = {D, L, H, F, P, 32, G};
  memcpy(p, cfg, 28); p += 28;
  float rope = 10000.f; memcpy(p, &rope, 4); p += 4;

  p = put_qt(p, V, D, G, 0x99);              /* tok_emb */
  p = put_qt(p, L * P, D, G, 0x88);          /* ple_model_proj */
  p = put_f(p, P, 1.f);                      /* ple_proj_norm */
  p = put_qt(p, V, L * P, G, 0x88);          /* ple_table */
  p = put_f(p, D, 1.f);                      /* attn_norm */
  p = put_qt(p, 3 * D, D, G, 0x88);
  p = put_qt(p, D, D, G, 0x88);
  p = put_f(p, D, 1.f);                      /* ffn_norm */
  p = put_qt(p, F, D, G, 0x88);
  p = put_qt(p, F, D, G, 0x88);
  p = put_qt(p, D, F, G, 0x88);
  p = put_qt(p, P, D, G, 0x88);
  p = put_qt(p, D, P, G, 0x88);
  p = put_f(p, D, 1.f);                      /* ple_norm */
  p = put_f(p, D, 1.f);                      /* out_norm */
  uint8_t *head_at = p;
  p = put_qt(p, VOUT, D, G, 0x77);           /* out_head, distinct fill */

  /* Reject malformed headers before any tensor is bound. Each case flips one
   * field of an otherwise valid image. */
  printf("\nheader validation rejects malformed images\n");
  {
    /* header field offsets: magic 0, version 4, header_bytes 8, flags 12,
     * input_vocab 16, output_vocab 20, dim 24, n_layers 28, n_heads 32,
     * ffn 36, ple_dim 40, seq_len 44, group 48, rope_theta 52 */
    struct { const char *what; int off; uint32_t val; } bad[] = {
      {"bad magic",                  0,  0x11223344u},
      {"version 0",                  4,  0u},
      {"version from the future",    4,  LLM_FORMAT_VERSION + 1u},
      {"header_bytes too small",     8,  8u},
      {"header_bytes absurd",        8,  1u << 20},
      {"unknown flag bit",          12,  1u << 7},
      {"input_vocab 0",             16,  0u},
      {"output_vocab 0",            20,  0u},
      {"dim 0",                     24,  0u},
      {"dim not divisible by heads",24,  9u},
      {"odd head dim",              24,  10u},   /* 10/2=5, RoPE needs even */
      {"n_layers 0",                28,  0u},
      {"n_layers over the cap",     28,  (uint32_t)(LLM_MAX_LAYERS + 1)},
      {"n_heads 0",                 32,  0u},
      {"ffn 0",                     36,  0u},
      {"ple_dim 0",                 40,  0u},
      {"seq_len 0",                 44,  0u},
      {"group 0",                   48,  0u},
    };
    Model bm;
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
      uint8_t *c = malloc(need);
      memcpy(c, img, need);
      memcpy(c + bad[i].off, &bad[i].val, 4);
      int r = llm_load(c, &bm);
      char d[48]; snprintf(d, sizeof d, "rc=%d", r);
      check(bad[i].what, r != 0, d);
      free(c);
    }
    /* A tied head cannot be longer than the embedding it views. */
    uint8_t *c = malloc(need);
    memcpy(c, img, need);
    uint32_t tied = LLM_FLAG_TIED_HEAD, big = 999u;
    memcpy(c + 12, &tied, 4);
    memcpy(c + 20, &big, 4);          /* output_vocab > input_vocab */
    int r = llm_load(c, &bm);
    char d[48]; snprintf(d, sizeof d, "rc=%d", r);
    check("tied output_vocab > input_vocab", r != 0, d);
    free(c);
  }

  printf("\nuntied format (synthetic)\n");
  Model u;
  int rc = llm_load(img, &u);
  check("llm_load accepts TIED_HEAD clear", rc == 0, "");
  if (rc == 0) {
    char d[96];
    snprintf(d, sizeof d, "in=%d out=%d", u.c.vocab, u.out_vocab);
    check("input_vocab and output_vocab differ",
          u.c.vocab == V && u.out_vocab == VOUT, d);
    check("out_head is a separate tensor, not tok_emb",
          u.out_head.codes != u.tok_emb.codes, "");
    check("out_head bound at the trailing offset",
          u.out_head.codes == head_at + 4, "");
    check("out_head has out_vocab rows", u.out_head.rows == VOUT, "");
    snprintf(d, sizeof d, "%zu vs %zu", u.image_bytes, need);
    check("image_bytes covers the head", u.image_bytes == need, d);
  }
  free(img);
}

// FP32 tensors follow byte-packed quantized ones in the image, so a norm vector
// can begin at any byte. rmsnorm takes a memcpy path when that happens; both
// paths must agree exactly, and a model whose tensors all land aligned would
// never reach the second one.
//
// The equality check alone does not prove the memcpy path is present: hosts
// that tolerate unaligned loads return the same floats either way. What this
// gives is a deterministic unaligned read on every run, so a sanitized build
// reports it whichever model is passed. Run under UBSan to detect a regression.
static void test_unaligned_norm(void) {
  enum { N = 64 };
  printf("\nunaligned norm weights (synthetic)\n");

  float x[N], w[N], aligned_out[N], unaligned_out[N];
  for (int i = 0; i < N; i++) {
    x[i] = (float)(i % 7) - 3.f;
    w[i] = 0.5f + (float)i * 0.01f;
  }

  // Aligned base plus 2, so the address is deterministically 2 mod 4 rather
  // than whatever the array happened to land on.
  _Alignas(float) uint8_t storage[N * sizeof(float) + 2];
  memcpy(storage + 2, w, sizeof w);
  const float *unaligned = (const float *)(void *)(storage + 2);

  check("test vector is really unaligned",
        ((uintptr_t)unaligned & (sizeof(float) - 1)) != 0, "");
  check("reference vector is aligned",
        ((uintptr_t)w & (sizeof(float) - 1)) == 0, "");

  rmsnorm(x, w, N, aligned_out);
  rmsnorm(x, unaligned, N, unaligned_out);
  check("unaligned weights give bit-identical output",
        memcmp(aligned_out, unaligned_out, sizeof aligned_out) == 0, "");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
    return 2;
  }
  const char *path = argv[1];
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *buf = malloc(sz);
  if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 2; }
  fclose(f);

  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 2; }
  printf("model: Vin=%d Vout=%d D=%d L=%d ffn=%d ple=%d  image=%zu bytes\n\n",
         m.c.vocab, m.out_vocab, m.c.dim, m.c.n_layers, m.c.ffn, m.c.ple_dim,
         m.image_bytes);

  printf("image_bytes\n");
  check("equals file size", (long)m.image_bytes == sz, "");

  // --- staging arithmetic ---------------------------------------------------
  printf("\nllm_stage_int8: staged == unstaged\n");
  int D = m.c.dim, P = m.c.ple_dim;

  float *x = malloc(sizeof(float) * 4096);
  for (int i = 0; i < 4096; i++) x[i] = sinf((float)i * 0.37f) * 1.7f;

  // One tensor of each distinct shape the model contains.
  struct { const char *name; QT *t; } cases[] = {
    {"ple_model_proj [L*P, D]", &m.ple_model_proj},
    {"qkv[0]         [3D, D]",  &m.qkv[0]},
    {"attn_proj[0]   [D, D]",   &m.attn_proj[0]},
    {"gate[0]        [F, D]",   &m.gate[0]},
    {"down[0]        [D, F]",   &m.down[0]},
    {"ple_gate[0]    [P, D]",   &m.ple_gate[0]},
    {"ple_proj[0]    [D, P]",   &m.ple_proj[0]},
  };
  int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

  for (int i = 0; i < ncases; i++) {
    QT *t = cases[i].t;
    int rows = t->rows;
    float *ref = malloc(sizeof(float) * rows);
    float *got = malloc(sizeof(float) * rows);

    MATVEC(t, x, ref);                       // unstaged: packed int4 nibbles

    void *sbuf = malloc(llm_stage_int8_bytes(t));
    llm_stage_int8(t, sbuf);
    MATVEC(t, x, got);                       // staged: pre-unpacked int8

    char detail[128];
    double d = max_abs_diff(ref, got, rows);
    snprintf(detail, sizeof detail, "rows=%-5d max|diff|=%.3g", rows, d);
    check(cases[i].name, d == 0.0, detail);

    // scale array must be 4-byte aligned wherever the int8 block ends
    check("  scale8 aligned",
          ((uintptr_t)t->scale8 % sizeof(float)) == 0, "");

    free(ref); free(got);
  }

  printf("\nrow-range split == whole tensor\n");
  {
    QT *t = &m.qkv[0];                       // already staged above
    int rows = t->rows;
    float *whole = malloc(sizeof(float) * rows);
    float *split = malloc(sizeof(float) * rows);
    int8_t *xq = malloc(t->cols);
    float xs;
    quantize_act(x, t->cols, xq, &xs);
    matvec_i8_range(t, xq, xs, whole, 0, rows);
    matvec_i8_range(t, xq, xs, split, 0, rows / 2);
    matvec_i8_range(t, xq, xs, split, rows / 2, rows);
    check("halves == whole",
          max_abs_diff(whole, split, rows) == 0.0, "");
    free(whole); free(split); free(xq);
  }

  // --- hook dispatch --------------------------------------------------------
  printf("\nplatform hook dispatch\n");
  Scratch s;
  memset(&s, 0, sizeof s);
  int L = m.c.n_layers, F = m.c.ffn, S = m.c.seq_len, V = m.out_vocab;
  s.x = calloc(D, 4); s.h = calloc(F > D ? F : D, 4);
  s.qkv = calloc(3 * D, 4); s.att = calloc(D, 4);
  s.g1 = calloc(F, 4); s.g2 = calloc(P > F ? P : F, 4);
  s.ple = calloc(L * P, 4); s.tmpP = calloc(L * P, 4); s.trow = calloc(L * P, 4);
  s.logits = calloc(V, 4); s.scores = calloc(S, 4);
  s.kcache = calloc((size_t)L * S * D, 4); s.vcache = calloc((size_t)L * S * D, 4);

  float *logits_nohook = malloc(sizeof(float) * V);
  llm_forward(&m, 42, 0, &s);
  memcpy(logits_nohook, s.logits, sizeof(float) * V);
  check("no hooks: baseline forward ran", 1, "");

  m.layer_matvec = layer_hook;
  m.head_matvec  = head_hook;
  llm_forward(&m, 42, 0, &s);

  // 1 ple_model_proj + 7 per layer
  int expect_layer = 1 + 7 * L;
  char d2[64];
  snprintf(d2, sizeof d2, "%d calls, expected %d", layer_hook_calls, expect_layer);
  check("layer_matvec called for every per-position matvec",
        layer_hook_calls == expect_layer, d2);
  snprintf(d2, sizeof d2, "%d calls, expected 1", head_hook_calls);
  check("head_matvec called once", head_hook_calls == 1, d2);
  check("hooked forward == unhooked forward",
        max_abs_diff(logits_nohook, s.logits, V) == 0.0, "");

  test_untied();
  test_unaligned_norm();

  printf("\n%s (%d failure%s)\n", failures ? "FAIL" : "PASS",
         failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
