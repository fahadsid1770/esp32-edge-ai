// Host stub: drawing is discarded. The grid is the thing under test, and
// display_flush renders only from the grid.
#pragma once
#define SH110X_WHITE 1
#define SH110X_BLACK 0
struct Adafruit_SH1106G {
  Adafruit_SH1106G(int, int, void *, int) {}
  bool begin(int, bool) { return true; }
  void clearDisplay() {}
  void display() {}
  void setTextSize(int) {}
  void setTextColor(int) {}
  void setTextWrap(bool) {}
  void setCursor(int, int) {}
  void print(const char *) {}
  void drawPixel(int, int, int) {}
  void drawLine(int, int, int, int, int) {}
  void drawCircle(int, int, int, int) {}
  void drawRoundRect(int, int, int, int, int, int) {}
  void fillRect(int, int, int, int, int) {}
  void drawFastHLine(int, int, int, int) {}
};
