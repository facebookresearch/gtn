#pragma once
#include <functional>
#include <utility>
#include <vector>

#include "graph.h"

namespace gtn {
namespace detail {

class ArcMatcher {
 public:
  virtual void match(int lnode, int rnode, bool matchIn = false) = 0;
  virtual bool hasNext() = 0;
  virtual std::pair<int, int> next() = 0;
};

class UnsortedMatcher : public ArcMatcher {
 public:
  UnsortedMatcher(const Graph& lhs, const Graph& rhs) :
    lhs_(lhs),
    rhs_(rhs) {};

  /* Match the arcs on the left node `lnode` and the right node `rnode`. If
   * `matchIn = false` (default) then arcs will be matched by `olabel`
   * otherwise they will be matched by `ilabel`.
   */
  void match(int lnode, int rnode, bool matchIn /* = false*/) override;
  bool hasNext() override;
  std::pair<int, int> next() override;

 private:
  const Graph& lhs_;
  const Graph& rhs_;
  std::vector<int>::const_iterator lIt_;
  std::vector<int>::const_iterator lItEnd_;
  std::vector<int>::const_iterator rItBegin_;
  std::vector<int>::const_iterator rIt_;
  std::vector<int>::const_iterator rItEnd_;
};

class SinglySortedMatcher : public ArcMatcher {
 public:
  SinglySortedMatcher(
      const Graph& lhs,
      const Graph& rhs,
      bool searchLhs = false);

  void match(int lnode, int rnode, bool matchIn /* = false */) override;

  bool hasNext() override;

  std::pair<int, int> next() override;

 private:
  const Graph& lhs_;
  const Graph& rhs_;
  bool searchLhs_;
  std::vector<int>::const_iterator searchIt_;
  std::vector<int>::const_iterator searchItBegin_;
  std::vector<int>::const_iterator searchItEnd_;
  std::vector<int>::const_iterator queryIt_;
  std::vector<int>::const_iterator queryItEnd_;
  std::function<bool(int, int)> comparisonFn_;
};


class DoublySortedMatcher : public ArcMatcher {
 public:
  DoublySortedMatcher(
      const Graph& lhs,
      const Graph& rhs) :
    lhs_(lhs),
    rhs_(rhs) {};

  void match(int lnode, int rnode, bool matchIn /* = false */) override;

  bool hasNext() override;

  std::pair<int, int> next() override;

 private:
  const Graph& lhs_;
  const Graph& rhs_;
  bool searchLhs_;
  std::vector<int>::const_iterator searchIt_;
  std::vector<int>::const_iterator searchItBegin_;
  std::vector<int>::const_iterator searchItEnd_;
  std::vector<int>::const_iterator queryIt_;
  std::vector<int>::const_iterator queryItEnd_;
  std::function<bool(int, int)> comparisonFn_;
};


/* Composes two transducers. */
Graph compose(const Graph& lhs, const Graph& rhs, std::shared_ptr<ArcMatcher> matcher);

} // namespace detail
} // namespace gtn
