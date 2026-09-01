// Host stub: the probe answers, so display_begin proceeds.
#pragma once
struct WireStub {
  void begin(int, int) {}
  void setClock(int) {}
  void beginTransmission(int) {}
  int endTransmission() { return 0; }
};
static WireStub Wire;
