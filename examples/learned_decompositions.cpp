#include <sstream>

#include "gtn/gtn.h"

using namespace gtn;

void asgWithTransducers() {
  // We can simplify ASG to it's core assumptions by using transducers.
  std::unordered_map<int, std::string> symbols = {{0, "e"}, {1, "h"}, {2, "t"}};

  // Each individual token graph encodes the fact that we can consume the same
  // token repeatedly while only emitting a single output.
  Graph e;
  e.addNode(true);
  e.addNode(false, true);
  e.addArc(0, 1, 0);
  e.addArc(1, 1, 0, epsilon);

  Graph h;
  h.addNode(true);
  h.addNode(false, true);
  h.addArc(0, 1, 1);
  h.addArc(1, 1, 1, epsilon);

  Graph t;
  t.addNode(true);
  t.addNode(false, true);
  t.addArc(0, 1, 2);
  t.addArc(1, 1, 2, epsilon);

  draw(e, "asg_e_graph.dot", symbols, symbols);
  draw(h, "asg_h_graph.dot", symbols, symbols);
  draw(t, "asg_t_graph.dot", symbols, symbols);

  // The closure of the union of the individual token graphs encodes the
  // fact that we can emit any sequence of zero or more tokens.
  auto tokens = closure(union_({e, h, t}));
  draw(tokens, "asg_tokens.dot", symbols, symbols);

  // The "the" graph can be simplified to just accepting the token sequence:
  //   ['t', 'h', 'e']
  auto the =
      loadTxt(std::stringstream("0\n"
                             "3\n"
                             "0 1 2\n"
                             "1 2 1\n"
                             "2 3 0\n"));
  draw(the, "asg_simple_the.dot", symbols);

  // This gives the standard force align graph for "the"
  auto asg_the = compose(tokens, the);
  draw(asg_the, "asg_eps_the.dot", symbols, symbols);

  // Clean-up / remove unneeded epsilons to speed things up.
  asg_the = remove(asg_the, epsilon);
  draw(asg_the, "asg_fst_the.dot", symbols, symbols);

  // Get an FSA (using the FST would be fine, but this way we don't have to
  // worry about the order in future composes with other FSAs).
  asg_the = projectInput(asg_the);
  draw(asg_the, "asg_fsa_the.dot", symbols);

  // At this point, we can use the standard emissions and transitions graph to
  // complete the ASG loss (see e.g. examples/asg.cpp).
}

void ctcWithTransducers() {
  // Just as in ASG, we can simplify CTC to it's core assumptions by using
  // transducers. We use "<B>" to denote the "blank" token to distinguish
  // it from an actual epsilon.
  std::unordered_map<int, std::string> symbols = {
      {0, "e"}, {1, "h"}, {2, "t"}, {3, "<B>"}};

  // The only difference from ASG and CTC is that we add a special blank (<B>)
  // token.
  Graph e;
  e.addNode(true);
  e.addNode(false, true);
  e.addArc(0, 1, 0);
  e.addArc(1, 1, 0, epsilon);

  Graph h;
  h.addNode(true);
  h.addNode(false, true);
  h.addArc(0, 1, 1);
  h.addArc(1, 1, 1, epsilon);

  Graph t;
  t.addNode(true);
  t.addNode(false, true);
  t.addArc(0, 1, 2);
  t.addArc(1, 1, 2, epsilon);

  // The <B> token is optional
  Graph blank;
  blank.addNode(true, true);
  blank.addArc(0, 0, 3, epsilon);
  draw(blank, "ctc_blank.dot", symbols, symbols);

  // Everything else is the same as in ASG!
  auto tokens = closure(union_({e, h, t, blank}));
  draw(tokens, "ctc_tokens.dot", symbols, symbols);

  // The "the" graph can be simplified to just accepting the token sequence:
  //   ['t', 'h', 'e']
  Graph the =
      loadTxt(std::stringstream("0\n"
                             "3\n"
                             "0 1 2\n"
                             "1 2 1\n"
                             "2 3 0\n"));

  // This gives the standard CTC force align graph for "the"
  auto ctc_the = remove(compose(tokens, the), epsilon);
  draw(ctc_the, "ctc_eps_the.dot", symbols, symbols);

  // At this point, we can use the standard emissions graph to
  // complete the CTC loss (see e.g. examples/ctc.cpp).
}

void wordDecompositions() {
  // Now that we know how to simplify ASG and CTC with transducers, we can do
  // things like adding many decompositions for a given word easily and in a
  // generic way. For example maybe we want to allow all possible
  // decompositions for the word "the":
  //   [t, h, e], [th, e], [t, he], [the]
  // given the token set [e, h, t, he, th, the].
  std::unordered_map<int, std::string> symbols = {
      {0, "e"}, {1, "h"}, {2, "t"}, {3, "th"}, {4, "he"}, {5, "the"}};

  std::vector<Graph> tokens_vec;
  for (auto& kv : symbols) {
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, kv.first);
    g.addArc(1, 1, kv.first, epsilon);
    tokens_vec.push_back(g);
  }
  auto tokens = closure(union_(tokens_vec));

  // The graph for "the" encodes the fact that multiple decompositions are
  // allowed and is the only change needed (other than augmenting the token
  // set).
  auto the =
      loadTxt(std::stringstream("0\n"
                             "3\n"
                             "0 1 2 -1\n"
                             "0 2 3 -1\n"
                             "0 3 5 5\n"
                             "1 2 1 -1\n"
                             "1 3 4 5\n"
                             "2 3 0 5\n"));
  draw(the, "word_decomps_the.dot", symbols, symbols);

  auto asg_the = remove(compose(tokens, the), epsilon);
  draw(asg_the, "asg_word_decomps_the.dot", symbols, symbols);

  // At this point, we can use the standard emissions and transitions graph to
  // complete the ASG loss (see e.g. examples/asg.cpp).

  // CTC with word decompositions is exactly the same except with the addition
  // of the special blank token.

  // Add blank to the symbol table
  symbols[6] = "<B>";

  // Add blank graph to the tokens graph
  Graph blank;
  blank.addNode(true, true);
  blank.addArc(0, 0, 6, epsilon);
  tokens_vec.push_back(blank);

  tokens = closure(union_(tokens_vec));
  auto ctc_the = remove(compose(tokens, the), epsilon);
  draw(ctc_the, "ctc_word_decomps_the.dot", symbols, symbols);
}

int main() {
  asgWithTransducers();
  ctcWithTransducers();
  wordDecompositions();
}
