// Barista: a question in, an answer out, one output class at a time. No chat
// history. The model generates the text; it does not select from a catalogue.
//
// Asymmetric vocabulary: it READS 8057 input tokens so it can take varied ASCII
// questions, and WRITES only its 854-class espresso output alphabet.
// That keeps the head a small fraction of per-token compute instead of the bulk
// of it, and it makes words outside the vocabulary literally unsayable. There
// are no digit tokens at all, so the model cannot invent "grind 2 steps finer".
//
// Input is USB serial: type a question, the answer streams back there and,
// when a panel is wired, to the OLED as well.
//
// Configuration: the per-position core is staged to int8 in PSRAM once at boot,
// and eligible per-layer matvecs are split across both LX7 cores.

#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"

#define LLM_INT8_ACT 1
// Set to 1 to print where the time goes, per answer. Costs a few microseconds
// of esp_timer calls per layer; leave off for demos. Measure with USE_DISPLAY 0,
// or the per-piece panel redraw lands in the wall-clock total.
#ifndef BARISTA_PROFILE
#define BARISTA_PROFILE 0
#endif

// Set to 0 to run every matvec on one core, for measuring what the split buys.
#ifndef BARISTA_DUAL_CORE
#define BARISTA_DUAL_CORE 1
#endif
#if BARISTA_PROFILE
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#endif
#include "../../runtime/llm.h"
#include "../../runtime/bpe_tokenizer.h"
#include "generated/tokenizer_encoder.h"
#include "generated/barista_words.h"
#include "generated/barista_out2in.h"

// Bumped when the deployed model changes; shown on the splash screen.
#define BARISTA_VERSION "0.1"

// Set to 0 to run serial-only, with no panel wired. See display.h for wiring.
#ifndef USE_DISPLAY
#define USE_DISPLAY 1
#endif
#if USE_DISPLAY
#include "display.h"
#endif

static Model model;
static Scratch scratch;
static BpeTokenizer tokenizer;
static bool ready = false;

// ---- dual-core per-layer matvec --------------------------------------------
#if BARISTA_DUAL_CORE
static TaskHandle_t worker_h, main_h;
static const QT *job_t; static const int8_t *job_xq; static float job_xs;
static float *job_y; static int job_split;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}
static void layer_matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);
  job_t=t; job_xq=xq; job_xs=xs; job_y=y; job_split=t->rows/2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}
#endif

static int sram_fallbacks = 0;
static void *ps(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM); }
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) { Serial.printf("PSRAM alloc failed: %s (%u)\n", what, (unsigned)n); while (1) delay(1000); }
  return p;
}

/* Internal SRAM, for the hot working set. This is the third tier of the memory
 * split, and which tenant belongs here follows from reads per token:
 *
 *   flash   PLE table + tok_emb   read once per token, one row out of 8057
 *   PSRAM   core weights, KV      read once per position
 *   SRAM    activations, norms    touched many times per token
 *
 * The activations earn their place through a narrower path than the tier table
 * suggests. matvec_q8 quantises the float input once into a static int8 buffer
 * already in internal RAM and the row loop walks that, so the dense inner loop
 * does not re-read these buffers per row. The traffic that does move is the
 * float reads during quantisation, the per-row output writes, RMSNorm and the
 * elementwise ops, the attention scores, and the logits array which the argmax
 * loop scans in full every token.
 *
 * The output head is deliberately NOT here despite fitting: it is read once per
 * token, the same frequency as the core weights beside it in PSRAM, and it would
 * cost internal SRAM for data with no reuse.
 *
 * The KV cache stays in PSRAM: it does not fit, and it is read once per position
 * rather than per matvec. */
// Set to 0 to put the hot working set back in PSRAM. Kept as a switch so the
// three-tier split can be A/B'd on one model: comparing across two models
// confounds it, because answer length changes the forwards-per-piece ratio.
#define BARISTA_SCRATCH_IN_SRAM 1
static void *sram_or_ps(size_t n, const char *what) {
#if !BARISTA_SCRATCH_IN_SRAM
  return ps_or_die(n, what);
#endif
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (p) return p;
  // Falling back is correct but changes what is being measured, so it must be
  // visible: a benchmark labelled "SRAM" could otherwise be partly PSRAM.
  ++sram_fallbacks;
  Serial.printf("SRAM full, %s -> PSRAM (%u B)\n", what, (unsigned)n);
  return ps_or_die(n, what);
}

// Relocate the RMSNorm weight vectors out of mmap'd flash into internal RAM.
// They are separate small vectors rather than one contiguous block, so each is
// copied and repointed individually. Copying also makes them 4-byte aligned,
// which the image does not guarantee: fp32 tensors follow byte-packed quantized
// ones, so rmsnorm reads them through its unaligned path until this runs.
static void copy_norms_to_sram() {
  Cfg *c = &model.c; int D = c->dim, L = c->n_layers, P = c->ple_dim;
  size_t before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const float **vecs[3 * 32 + 2]; int n_vec = 0, sizes[3 * 32 + 2];
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  int moved = 0;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!dst) { ++sram_fallbacks; continue; }  // still correct, but not what
    memcpy(dst, *vecs[i], bytes);              // the label would claim
    *vecs[i] = (const float *)dst;
    ++moved;
  }
  Serial.printf("norms in SRAM: %d/%d vectors, %u B\n", moved, n_vec,
                (unsigned)(before - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
}

static void alloc_scratch() {
  Cfg *c=&model.c; int D=c->dim,L=c->n_layers,P=c->ple_dim,F=c->ffn,S=c->seq_len;
  size_t before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  // hot working set -> internal SRAM
  scratch.x=(float*)sram_or_ps(D*4,"x");            scratch.h=(float*)sram_or_ps((F>D?F:D)*4,"h");
  scratch.qkv=(float*)sram_or_ps(3*D*4,"qkv");      scratch.att=(float*)sram_or_ps(D*4,"att");
  scratch.g1=(float*)sram_or_ps(F*4,"g1");          scratch.g2=(float*)sram_or_ps((P>F?P:F)*4,"g2");
  scratch.ple=(float*)sram_or_ps(L*P*4,"ple");      scratch.tmpP=(float*)sram_or_ps(L*P*4,"tmpP");
  scratch.trow=(float*)sram_or_ps(L*P*4,"trow");
  scratch.logits=(float*)sram_or_ps(model.out_vocab*4,"logits");
  scratch.scores=(float*)sram_or_ps(S*4,"scores");
  // KV cache stays in PSRAM: it is large, and read once per position not per matvec
  scratch.kcache=(float*)ps_or_die((size_t)L*S*D*4,"kcache");
  scratch.vcache=(float*)ps_or_die((size_t)L*S*D*4,"vcache");
  Serial.printf("scratch in SRAM: %u B\n",
                (unsigned)(before - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
#if BARISTA_SCRATCH_IN_SRAM
  copy_norms_to_sram();   // after the scratch total is taken, so the two are separable
#endif
  Serial.printf("sram free %.0f KB\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL)/1024.0);
}

// ---- generation -------------------------------------------------------------
// Room reserved for the answer, so a long question cannot leave the generation
// loop with nowhere to write.
#define BARISTA_ANSWER_ROOM 56
#define BARISTA_MAX_PIECES 48

static void answer(const char *question) {
  uint16_t ids[BTK_MAX_INPUT_BYTES];
  int n = bpe_encode_ascii(&tokenizer, question, ids, (int)(sizeof(ids)/sizeof(ids[0])));
  if (n == BTK_ERR_NOT_ASCII) {
    Serial.println("(ascii only)");
#if USE_DISPLAY
    display_question(question);
    display_notice("(ascii only)");
#endif
    return;
  }
  if (n <= 0 || n > model.c.seq_len - BARISTA_ANSWER_ROOM) {
    Serial.println("(question too long)");
#if USE_DISPLAY
    display_clear();
    display_word("(question too long)", false);
    display_flush();
#endif
    return;
  }

#if USE_DISPLAY
  display_question(question);
#endif

  int64_t t0 = esp_timer_get_time();
#if BARISTA_PROFILE
  llm_profile_reset(&scratch);
#endif
  int pos = 0;
  for (int i = 0; i < n; i++) llm_forward(&model, ids[i], pos++, &scratch);
  llm_forward(&model, BARISTA_OUT2IN[BARISTA_BOS], pos++, &scratch);

  Serial.print("A: ");
  int pieces_out = 0;
  // Each iteration emits one output class. Punctuation is a class, so these are
  // pieces of output rather than readable words.
  for (int step = 0; step < BARISTA_MAX_PIECES && pos < model.c.seq_len; step++) {
    // greedy over output classes
    int best = 0;
    for (int k = 1; k < BARISTA_WORD_COUNT; k++)
      if (scratch.logits[k] > scratch.logits[best]) best = k;
    if (best == BARISTA_EOS) break;
    const char *w = BARISTA_WORDS[best];
    bool punct = (w[1] == '\0' && strchr(".,:;?", w[0]) != NULL);
    if (pieces_out && !punct) Serial.print(' ');
    Serial.print(w);            // stream it as it is produced
    Serial.flush();
#if USE_DISPLAY
    display_word(w, !punct);
    display_flush();
#endif
    pieces_out++;
    // The head emits an output CLASS. Feeding it forward needs the input token
    // id that class corresponds to, which is what out2in holds.
    llm_forward(&model, BARISTA_OUT2IN[best], pos++, &scratch);
  }
  double ms = (esp_timer_get_time() - t0) / 1000.0;
  Serial.printf("\n[%d pieces, %.0f ms, %.1f pieces/s]\n", pieces_out, ms,
                pieces_out * 1000.0 / ms);
#if BARISTA_PROFILE
  {
    // input = embedding row + PLE table row + ple_model_proj + RoPE.
    // Sums over every llm_forward in this answer, prompt positions included,
    // so the forward count exceeds the number of pieces emitted.
    uint64_t in = scratch.profile.input_us, at = scratch.profile.attn_us;
    uint64_t ff = scratch.profile.ffn_us, pl = scratch.profile.ple_us;
    uint64_t hd = scratch.profile.head_us;
    uint64_t tot = in + at + ff + pl + hd;
    if (tot == 0) tot = 1;
    Serial.printf("[profile %u fwd | input %.0fms %.0f%% | attn %.0fms %.0f%% | "
                  "ffn %.0fms %.0f%% | ple %.0fms %.0f%% | head %.0fms %.0f%% | "
                  "accounted %.0f%%]\n",
                  (unsigned)scratch.profile.calls,
                  in/1000.0, 100.0*in/tot, at/1000.0, 100.0*at/tot,
                  ff/1000.0, 100.0*ff/tot, pl/1000.0, 100.0*pl/tot,
                  hd/1000.0, 100.0*hd/tot, 100.0*(tot/1000.0)/ms);
  }
#endif
}

// The line buffer below holds BTK_MAX_INPUT_BYTES, but the hardware CDC receive
// queue defaults to 256, so a longer paste overruns the queue before loop() can
// drain it and bytes are lost mid-line. Size the queue above the longest line
// this sketch will accept, with room for the newline and whatever follows it in
// the same burst.
#define BARISTA_SERIAL_RX_BYTES (2 * BTK_MAX_INPUT_BYTES)

void setup() {
  // Must be requested before begin(), which allocates a 256-byte queue if none
  // exists. A failed allocation returns 0 and begin() then falls back to that
  // default, so the request has to be checked rather than assumed.
  size_t rx_bytes = Serial.setRxBufferSize(BARISTA_SERIAL_RX_BYTES);
  Serial.begin(115200); delay(1500);
  if (rx_bytes != BARISTA_SERIAL_RX_BYTES) {
    Serial.println("serial RX buffer allocation failed");
    return;
  }
  Serial.println("\n=== ESP32 BARISTA ===");
  Serial.println("ask an espresso question; the model writes the answer.");

  if (bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE,
                         &tokenizer)) {
    Serial.println("tokenizer load failed"); return;
  }
#if USE_DISPLAY
  if (!display_begin())
    Serial.printf("no display answering on i2c 0x%02X; serial only\n", OLED_ADDR);
#endif
  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("no model partition"); return; }
  const void *base; esp_partition_mmap_handle_t h;
  if (esp_partition_mmap(part,0,part->size,ESP_PARTITION_MMAP_DATA,&base,&h)!=ESP_OK) {
    Serial.println("mmap failed"); return;
  }
  if (llm_load((const uint8_t*)base, &model)) { Serial.println("bad model"); return; }
  Cfg *c=&model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim);
  // The word tables and the head must describe the same output alphabet. They
  // are generated from the same layout the model was exported against, so a
  // mismatch means one of the two is stale.
  if (model.out_vocab != BARISTA_WORD_COUNT) {
    Serial.printf("word table mismatch: model %d vs table %d\n",
                  model.out_vocab, BARISTA_WORD_COUNT);
    return;
  }
  if ((int)tokenizer.active_vocab > c->vocab) {
    Serial.printf("tokenizer/model mismatch: encoder can emit %u ids, model reads %d\n",
                  (unsigned)tokenizer.active_vocab, c->vocab);
    return;
  }
  // The tokenizer only covers ids a question can produce. Generated answers feed
  // BARISTA_OUT2IN[class] back in, and those reach further. The generator
  // allocates appended ids densely up to the input vocabulary, so the largest
  // mapped id plus one is exactly the width the tables were built for: any other
  // value means the model and the tables came from different layouts.
  {
    uint16_t max_in = 0;
    for (int k = 0; k < BARISTA_WORD_COUNT; k++)
      if (BARISTA_OUT2IN[k] > max_in) max_in = BARISTA_OUT2IN[k];
    if ((int)max_in + 1 != c->vocab) {
      Serial.printf("out2in/model mismatch: mapped ids reach %u, model reads %d\n",
                    (unsigned)max_in, c->vocab);
      return;
    }
  }
  alloc_scratch();

  int staged = llm_stage_core_int8_alloc(&model, ps);
  // The output head is not part of the "core" the staging helper walks, so it
  // would otherwise stay on the slow path: int4, read from flash, single core.
  // Stage it and route it through the same dual-core matvec.
  {
    void *b = ps(llm_stage_int8_bytes(&model.out_head));
    if (b) { llm_stage_int8(&model.out_head, b); ++staged; }
  }
  Serial.printf("int8-staged %d tensors | psram free %.2f MB\n",
                staged, heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1048576.0);
  // Provenance: identify the exact weights and build, not just the shape.
  // FNV-1a over every byte of the model image.
  {
    const uint8_t *img = (const uint8_t *)base;
    const uint32_t *w = (const uint32_t *)base;
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < model.image_bytes; i++) { fp ^= img[i]; fp *= 16777619u; }
    Serial.printf("build: magic=%08x bytes=%u fp=%08x scratch_sram=%d fallbacks=%d\n",
                  (unsigned)w[0], (unsigned)model.image_bytes, (unsigned)fp,
                  BARISTA_SCRATCH_IN_SRAM, sram_fallbacks);
  }

  int dual_core_active = 0;
#if BARISTA_DUAL_CORE
  main_h = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(worker_main,"mv",4096,NULL,2,&worker_h,0)==pdPASS) {
    model.layer_matvec = layer_matvec_par;
    // Only once the worker exists: layer_matvec_par notifies worker_h, so
    // assigning either hook before this point would signal a null handle.
    if (model.out_head.w8) model.head_matvec = layer_matvec_par;
    dual_core_active = 1;
  }
  else Serial.println("dual-core worker failed; running single core");
#endif

  // Which build is actually running. The weights fingerprint above identifies
  // the model; this identifies the switches, so a benchmark can require the
  // configuration it claims to be measuring rather than trusting a label.
#if USE_DISPLAY
  int display_present_now = display_present ? 1 : 0;
#else
  int display_present_now = 0;
#endif
  Serial.printf("config: profile=%d dual_core_requested=%d dual_core_active=%d "
                "display_enabled=%d display_present=%d\n",
                BARISTA_PROFILE, BARISTA_DUAL_CORE, dual_core_active,
                USE_DISPLAY, display_present_now);

  ready = true;
#if USE_DISPLAY
  display_splash();
#endif
  Serial.println("READY>");
}

void loop() {
  static char line[BTK_MAX_INPUT_BYTES];
  static int len = 0;
  static bool overflowed = false;
  if (!ready) { delay(1000); return; }
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      line[len] = '\0';
      // Report the rejection once, at the newline, rather than per byte, and
      // clear both pieces of state so the next question starts fresh.
      if (overflowed) Serial.println("(question too long)");
      else if (len) answer(line);
      len = 0; overflowed = false;
      Serial.println("READY>");
    } else if (len < (int)sizeof(line) - 1) {
      line[len++] = ch;
    } else {
      // Past the buffer. Keep consuming to the newline so the tail of an
      // oversized line cannot be read as the beginning of the next one.
      overflowed = true;
    }
  }
  delay(5);
}
