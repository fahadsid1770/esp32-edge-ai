// Optional on-device screen: the question and the answer appear on a display
// wired to the ESP32 itself, no laptop.
//
// 128x64 I2C mono OLED, 4 wires:
//   GND->GND, VCC->3V3, SCL->GPIO46, SDA->GPIO18
// Set OLED_CONTROLLER to match the panel: 1.3" is usually SH1106, 0.96" usually
// SSD1306.
//
// The panel is probed once at startup. If it does not answer, serial output
// continues and every draw becomes a no-op.
//
// Barista writes whole words, not tokens, and answers are longer than the panel
// holds, so this keeps a line buffer and scrolls. Each update redraws the grid
// into the framebuffer and then transfers the whole framebuffer.
#ifndef DISPLAY_H
#define DISPLAY_H

#include <Adafruit_GFX.h>
#include <Wire.h>

#define OLED_SH1106  1
#define OLED_SSD1306 2
#ifndef OLED_CONTROLLER
#define OLED_CONTROLLER OLED_SH1106
#endif

#define OLED_SDA 18
#define OLED_SCL 46
#ifndef OLED_ADDR
#define OLED_ADDR 0x3C          // some panels are 0x3D
#endif
#define SCR_W 128
#define SCR_H 64
#define CW 6                    // 6x8 base glyph at text size 1
#define CH 8
#define COLS (SCR_W / CW)       // 21
#define ROWS (SCR_H / CH)       // 8

#if OLED_CONTROLLER == OLED_SH1106
#include <Adafruit_SH110X.h>
static Adafruit_SH1106G oled = Adafruit_SH1106G(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SH110X_WHITE
#define OLED_BLACK SH110X_BLACK
#elif OLED_CONTROLLER == OLED_SSD1306
#include <Adafruit_SSD1306.h>
static Adafruit_SSD1306 oled = Adafruit_SSD1306(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SSD1306_WHITE
#define OLED_BLACK SSD1306_BLACK
#else
#error "OLED_CONTROLLER must be OLED_SH1106 or OLED_SSD1306"
#endif

// A row holding this sentinel is drawn as a horizontal rule instead of text, so
// the separator costs the same one row as any other line and scrolls with them.
#define GRID_RULE '\x01'

static char grid[ROWS][COLS + 1];
static int grid_row = 0, grid_col = 0;
// False until the panel both answers on the bus and initialises. Every draw
// checks it, so an absent panel costs one probe rather than a failed write per
// word.
static bool display_present = false;

static void display_clear() {
  for (int r = 0; r < ROWS; r++) grid[r][0] = '\0';
  grid_row = 0;
  grid_col = 0;
}

// Returns false when nothing answers at OLED_ADDR.
static bool display_begin() {
  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(400000);   // 400kHz fast I2C, ~4x quicker frame flush than default
  Wire.beginTransmission(OLED_ADDR);
  display_present = (Wire.endTransmission() == 0);
  if (!display_present) return false;
  // Answering on the bus is not the same as coming up: begin() still allocates
  // the framebuffer and initialises the controller, and can fail at either.
#if OLED_CONTROLLER == OLED_SH1106
  display_present = oled.begin(OLED_ADDR, true);
#else
  display_present = oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
#endif
  if (!display_present) return false;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(OLED_WHITE);
  oled.setTextWrap(false);      // wrapping happens on word boundaries below
  oled.display();
  display_clear();
  return true;
}

// Push the line buffer to the panel. One full framebuffer write.
static void display_flush() {
  if (!display_present) return;
  oled.clearDisplay();
  for (int r = 0; r < ROWS; r++) {
    if (!grid[r][0]) continue;
    if (grid[r][0] == GRID_RULE) {
      oled.drawFastHLine(0, r * CH + CH / 2, SCR_W, OLED_WHITE);
      continue;
    }
    oled.setCursor(0, r * CH);
    oled.print(grid[r]);
  }
  oled.display();
}

// Drop the top line and move everything up, so the newest text stays visible.
static void display_scroll() {
  for (int r = 0; r + 1 < ROWS; r++) memcpy(grid[r], grid[r + 1], COLS + 1);
  grid[ROWS - 1][0] = '\0';
  grid_row = ROWS - 1;
  grid_col = 0;
}

static void display_newline() {
  if (grid_row + 1 < ROWS) { grid_row++; grid_col = 0; }
  else display_scroll();
}

// Separator between the question and the answer.
static void display_rule() {
  if (grid_col > 0) display_newline();
  grid[grid_row][0] = GRID_RULE;
  grid[grid_row][1] = '\0';
  display_newline();
}

// Punctuation must never start a row on its own. When it will not fit, the word
// it belongs to moves down with it.
static void display_carry_last_word() {
  int start = grid_col;
  while (start > 0 && grid[grid_row][start - 1] != ' ') start--;
  if (start == 0) { display_newline(); return; }   // the row is one long word
  char carry[COLS + 1];
  int n = grid_col - start;
  memcpy(carry, &grid[grid_row][start], (size_t)n);
  int end = start;
  while (end > 0 && grid[grid_row][end - 1] == ' ') end--;
  grid[grid_row][end] = '\0';
  grid_col = end;
  display_newline();
  for (int i = 0; i < n; i++) grid[grid_row][grid_col++] = carry[i];
  grid[grid_row][grid_col] = '\0';
}

static void display_putc(char c) {
  if (grid_col >= COLS) display_newline();
  grid[grid_row][grid_col++] = c;
  grid[grid_row][grid_col] = '\0';
}

// Append one word, wrapping whole words. `space_before` is the caller's
// decision, so punctuation can be attached to the word it follows. A word wider
// than the panel is wrapped across rows rather than truncated.
static void display_word_n(const char *w, int len, bool space_before) {
  if (space_before) {
    if (grid_col + 1 + len > COLS && len <= COLS) display_newline();
    else if (grid_col > 0) display_putc(' ');
  } else if (grid_col + len > COLS && len <= COLS) {
    display_carry_last_word();
  }
  for (int i = 0; i < len; i++) display_putc(w[i]);
}

static void display_word(const char *w, bool space_before) {
  display_word_n(w, (int)strlen(w), space_before);
}

// Start a new exchange: the question, then the answer prefix that words append
// to. The question scrolls off on its own once the answer grows past the panel.
static void display_question(const char *q) {
  display_clear();
  display_word("Q:", false);
  const char *p = q;
  while (*p) {
    while (*p == ' ') p++;
    const char *start = p;
    while (*p && *p != ' ') p++;
    if (p == start) break;
    display_word_n(start, (int)(p - start), true);
  }
  display_rule();
  display_word("A:", false);
  display_flush();
}

// Idle screen, shown once after initialisation and replaced by the first
// question. The cup is drawn from primitives rather than stored as a bitmap: it
// is a handful of calls, and the proportions stay easy to adjust.
// Identifies the deployed model release. The sketch defines it; the deploy flow
// recompiles the firmware alongside the model, so the two stay in step.
#ifndef BARISTA_VERSION
#define BARISTA_VERSION "0.0"
#endif
#define BARISTA_TITLE "Barista v" BARISTA_VERSION
#define BARISTA_HINT  "ask about espresso"

static void display_centred(const char *s, int y) {
  int w = (int)strlen(s) * CW;
  oled.setCursor((SCR_W - w) / 2, y);
  oled.print(s);
}

static void display_cup(int cx, int cy) {
  // Steam: two dotted waves rising from the rim.
  for (int i = 0; i < 9; i++) {
    int wave = ((i / 3) % 2) ? 1 : -1;
    oled.drawPixel(cx - 6 + wave, cy - 20 + i, OLED_WHITE);
    oled.drawPixel(cx + 5 - wave, cy - 20 + i, OLED_WHITE);
  }
  oled.drawRoundRect(cx - 13, cy - 9, 26, 18, 3, OLED_WHITE);   // body
  oled.drawLine(cx - 10, cy - 5, cx + 9, cy - 5, OLED_WHITE);   // coffee line
  oled.drawCircle(cx + 17, cy - 1, 5, OLED_WHITE);              // handle
  oled.drawCircle(cx + 17, cy - 1, 4, OLED_WHITE);
  oled.fillRect(cx + 11, cy - 6, 4, 11, OLED_BLACK);            // tidy the join
  oled.drawRoundRect(cx - 19, cy + 11, 38, 4, 2, OLED_WHITE);   // saucer
}

static void display_splash() {
  if (!display_present) return;
  oled.clearDisplay();
  display_centred(BARISTA_TITLE, 0);
  display_cup(SCR_W / 2, 30);
  display_centred(BARISTA_HINT, SCR_H - CH);
  oled.display();
  display_clear();
}

// A short notice in place of an answer, for refused input.
static void display_notice(const char *text) {
  display_newline();
  display_word(text, true);
  display_flush();
}

#endif
