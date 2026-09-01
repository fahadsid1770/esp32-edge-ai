// Verifies the panel's line buffer on the host: word wrap, punctuation spacing,
// scrolling, and words wider than the panel.
//
// Only display_flush touches the controller, and it renders solely from the
// grid, so stubbing the driver leaves the behaviour under test intact.
//
//   c++ -std=c++17 -O2 -Wall -Wextra \
//     -I firmware/esp32_barista/host_verify/stub \
//     -o /tmp/dv firmware/esp32_barista/host_verify/display_verify.cpp
//   /tmp/dv

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>

#define BARISTA_VERSION "test"
#include "../display.h"

static int failures = 0;

static void check(const char *what, bool ok, const std::string &detail = "") {
  printf("  %-50s %s%s%s\n", what, ok ? "ok" : "FAIL",
         detail.empty() ? "" : "  ", detail.c_str());
  if (!ok) failures++;
}

// The grid as one line of text. Rows join with a space because a row break is a
// wrap, not a word boundary that was already spaced.
static std::string visible(bool keep_rule = false) {
  std::string s;
  for (int r = 0; r < ROWS; r++) {
    if (!grid[r][0]) continue;
    if (grid[r][0] == GRID_RULE) { if (keep_rule) s += "|RULE|"; continue; }
    if (!s.empty()) s += " ";
    s += grid[r];
  }
  return s;
}

static std::string squash(std::string in) {
  std::string out;
  bool space = false;
  for (char c : in) {
    if (c == ' ') { if (!space) out += ' '; space = true; }
    else { out += c; space = false; }
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  while (!out.empty() && out.front() == ' ') out.erase(0, 1);
  return out;
}

static bool row_widths_ok() {
  for (int r = 0; r < ROWS; r++)
    if ((int)strlen(grid[r]) > COLS) return false;
  return true;
}

// Writes each word the way the sketch does, and returns the text it intended.
static std::string say(const char *const *words, int n) {
  std::string produced;
  for (int i = 0; i < n; i++) {
    const char *w = words[i];
    bool punct = strlen(w) == 1 && strchr(".,:;?", w[0]) != nullptr;
    display_word(w, !punct);
    if (!produced.empty() && !punct) produced += " ";
    produced += w;
  }
  return produced;
}

int main() {
  display_begin();

  // --- question, rule, answer prefix -----------------------------------------
  display_question("my espresso tastes really bitter, what should i change?");
  std::string laid_out = squash(visible(true));
  check("question text reaches the panel",
        laid_out.find("my espresso tastes really bitter") != std::string::npos);
  check("a rule separates question from answer",
        laid_out.find("|RULE|") != std::string::npos);
  check("the answer prefix follows the rule",
        laid_out.find("|RULE| A:") != std::string::npos);
  check("no row is wider than the panel", row_widths_ok());

  // --- punctuation attaches to the word before it ----------------------------
  display_clear();
  {
    static const char *const w[] = {"go", "finer", ",", "then", "re-pull", "."};
    std::string produced = say(w, 6);
    check("punctuation is not preceded by a space",
          squash(visible()) == squash(produced),
          squash(visible()) == squash(produced) ? "" : "got: " + squash(visible()));
  }

  // --- a long answer scrolls without losing text -----------------------------
  display_question("is my puck too wet?");
  {
    static const char *const w[] = {
        "if", "the", "flow", "is", "quicker", "than", "your", "usual", "pull",
        ",", "the", "setting", "is", "likely", "too", "coarse", ".", "confirm",
        "the", "basket", "and", "that", "the", "grinder", "is", "adjustable",
        ",", "then", "move", "one", "step", "."};
    std::string produced = "A: " + say(w, 32);
    std::string seen = squash(visible());
    std::string want = squash(produced);
    bool suffix = want.size() >= seen.size() &&
                  want.compare(want.size() - seen.size(), seen.size(), seen) == 0;
    check("what is on screen is an exact suffix of what was written", suffix,
          suffix ? "" : "\n    seen: " + seen + "\n    want: " + want);
    check("the panel is full after a long answer", grid[ROWS - 1][0] != '\0');
    check("no row is wider than the panel after scrolling", row_widths_ok());
  }

  // --- words wider than the panel --------------------------------------------
  {
    const char *long_word = "supercalifragilisticexpialidocious";  // 34 > 21
    display_clear();
    display_word(long_word, false);
    std::string seen = squash(visible());
    seen.erase(std::remove(seen.begin(), seen.end(), ' '), seen.end());
    check("a word wider than the panel is wrapped, not truncated",
          seen == long_word, seen == long_word ? "" : "got: " + seen);
    check("no row is wider than the panel for a long word", row_widths_ok());
  }
  {
    // The same word arriving inside a question must survive too: the question
    // path writes spans rather than null-terminated copies.
    const char *q = "why is my supercalifragilisticexpialidocious shot sour";
    display_question(q);
    std::string seen = squash(visible());
    seen.erase(std::remove(seen.begin(), seen.end(), ' '), seen.end());
    check("a long word inside a question is not truncated",
          seen.find("supercalifragilisticexpialidocious") != std::string::npos,
          seen.find("supercalifragilisticexpialidocious") != std::string::npos
              ? "" : "got: " + seen);
  }

  // --- the remaining entry points are exercised, not just compiled ----------
  display_question("is my puck too wet?");
  display_notice("(ascii only)");
  check("a notice appears under the question",
        squash(visible()).find("(ascii only)") != std::string::npos);
  display_splash();
  check("the splash clears the text grid", visible().empty());

  printf("\n%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures,
         failures == 1 ? "" : "s");
  return failures != 0;
}
