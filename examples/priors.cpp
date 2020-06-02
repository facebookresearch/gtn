#include <cstdlib>
#include <sstream>

#include "gtn/gtn.h"

using namespace gtn;

void asgMulti() {
  /*
   * This example shows how to add a different decomposition for a word
   * and incorporate it into the alignment graph. We consider the word "the"
   * and allow the decompositions ["t", "h", "e"] or ["th", "e"].
   */
  std::unordered_map<int, std::string> symbols = {{0, "e"}, {1, "h"}, {2, "t"}};

  // ASG force align graph for "the"
  Graph the =
      load(std::stringstream("0\n"
                             "3\n"
                             "0 0 2\n"
                             "0 1 2\n"
                             "1 1 1\n"
                             "1 2 1\n"
                             "2 2 0\n"
                             "2 3 0\n"));

  // Emissions (recognition) graph
  int N = 3;
  const int T = 4; // graph length (frames)
  Graph emissions;
  emissions.addNode(true);
  for (int t = 1; t <= T; t++) {
    emissions.addNode(false, t == T);
    for (int i = 0; i < N; i++) {
      emissions.addArc(t - 1, t, i);
    }
  }
  draw(the, "asg_the.dot", symbols);
  draw(emissions, "asg_the_emissions.dot", symbols);
  draw(compose(the, emissions), "asg_the_composed.dot", symbols);

  // Add "th" as a token and a possible alignment
  symbols[N++] = "th";
  the.addNode(true);
  the.addArc(4, 4, N - 1);
  the.addArc(4, 3, N - 1);
  for (int t = 1; t <= T; t++) {
    emissions.addArc(t - 1, t, N - 1);
  }

  draw(the, "asg_the_multi.dot", symbols);
  draw(emissions, "asg_the_emissions_multi.dot", symbols);
  draw(compose(the, emissions), "asg_the_composed_multi.dot", symbols);
}

void asgSubLetter() {
  /*
   * This example shows how to add a beginning, middle, and end sub-units for
   * each token.
   */
  std::unordered_map<int, std::string> symbols = {
      {0, "-a"},
      {1, "-a-"},
      {2, "a-"},
      {3, "-c"},
      {4, "-c-"},
      {5, "c-"},
      {6, "-t"},
      {7, "-t-"},
      {8, "t-"},
  };

  // ASG force align graph for "cat"
  Graph cat =
      load(std::stringstream("0\n"
                             "6\n"
                             "0 1 3\n"
                             "1 1 4\n"
                             "1 2 5\n"
                             "2 3 0\n"
                             "3 3 1\n"
                             "3 4 2\n"
                             "4 5 6\n"
                             "5 5 7\n"
                             "5 6 8\n"));

  // Emissions (recognition) graph
  const int N = 9;
  const int T = 6; // graph length (frames)
  Graph emissions;
  emissions.addNode(true);
  for (int t = 1; t <= T; t++) {
    emissions.addNode(false, t == T);
    for (int i = 0; i < N; i++) {
      emissions.addArc(t - 1, t, i);
    }
  }
  draw(cat, "asg_cat_sub.dot", symbols);
  draw(emissions, "asg_cat_sub_emissions.dot", symbols);
}

void ctcForceEps() {
  /*
   * This example shows how to force blanks for a word
   * at various positions. We consider the word "cat".
   */
  std::unordered_map<int, std::string> symbols = {
      {0, "-"}, {1, "a"}, {2, "c"}, {3, "t"}};

  // CTC force align base graph for "cat"
  std::string base =
      "1 1 0\n"
      "1 2 2\n"
      "2 2 2\n"
      "2 3 0\n"
      "3 3 0\n"
      "2 4 1\n"
      "3 4 1\n"
      "4 4 1\n"
      "4 5 0\n"
      "5 5 0\n"
      "5 6 3\n"
      "6 6 3\n"
      "6 7 0\n"
      "7 7 0\n";

  Graph cat = load(std::stringstream("1\n6 7\n" + base + "4 6 3\n"));
  draw(cat, "ctc_cat.dot", symbols);

  cat = load(std::stringstream("0\n7\n0 1 0\n" + base + "4 6 3\n"));
  draw(cat, "ctc_cat_force_start_end.dot", symbols);

  cat = load(std::stringstream("1\n6 7\n" + base));
  draw(cat, "ctc_cat_force_mid.dot", symbols);
}

int main() {
  /*
   * Examples of incorporating optional hand-designed priors into loss
   * functions like ASG and CTC with WFSAs.
   */
  asgMulti();
  asgSubLetter();
  ctcForceEps();
}
