#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <random>

// #include <unordered_map>
// #include <unordered_set>

// #include "neighbor.hpp"
// #include "planresult.hpp"

namespace libMultiRobotPlanning {

/*!
  \example a_star.cpp Simple example using a 2D grid world and
  up/down/left/right
  actions
*/

/*! \brief A* Algorithm to find the shortest path

This class implements the A* algorithm. A* is an informed search algorithm
that finds the shortest path for a given map. It can use a heuristic that
needsto be admissible.

This class can either use a fibonacci heap, or a d-ary heap. The latter is the
default. Define "USE_FIBONACCI_HEAP" to use the fibonacci heap instead.

\tparam State Custom state for the search. Needs to be copy'able
\tparam Action Custom action for the search. Needs to be copy'able
\tparam Cost Custom Cost type (integer or floating point types)
\tparam Environment This class needs to provide the custom A* logic. In
    particular, it needs to support the following functions:
  - `Cost admissibleHeuristic(const State& s)`\n
    This function can return 0 if no suitable heuristic is available.

  - `bool isSolution(const State& s)`\n
    Return true if the given state is a goal state.

  - `void getNeighbors(const State& s, std::vector<Neighbor<State, Action,
   int> >& neighbors)`\n
    Fill the list of neighboring state for the given state s.

  - `void onExpandNode(const State& s, int fScore, int gScore)`\n
    This function is called on every expansion and can be used for statistical
purposes.

  - `void onDiscover(const State& s, int fScore, int gScore)`\n
    This function is called on every node discovery and can be used for
   statistical purposes.

    \tparam StateHasher A class to convert a state to a hash value. Default:
   std::hash<State>
*/
template <typename State, typename Action, typename Reward, typename Environment, typename URNG = std::default_random_engine>
class MonteCarloTreeSearch {
 public:
  MonteCarloTreeSearch(
    Environment& environment,
    URNG& generator,
    size_t num_nodes,
    float Cp)
    : m_env(environment)
    , m_generator(generator)
    , m_num_nodes(num_nodes)
    , m_Cp(Cp)
    {}

  bool search(const State& startState, Action& result) {
    // we pre-allocate all memory to ensure that pointers stay valid
    m_nodes.clear();
    m_nodes.reserve(m_num_nodes+1);

    m_nodes.resize(1);
    auto& root = m_nodes[0];
    root.state = startState;

    while (root.number_of_visits < m_num_nodes) {
      // selection + expansion
      Node* node = treePolicy(root);
      assert(node != nullptr);
//      if (node == nullptr) {
//        return false;
//      }
      Reward reward = m_env.rollout(node->state);
      backpropagation(node, reward);
    }
    // std::cout << "R " << root.reward.first << " " << root.reward.second << std::endl;
    const Node* node = bestChild(&root, 0);
    if (node != nullptr) {
      result = node->action_to_node;
      return true;
    }
    return false;
  }

  const Reward& rootNodeReward() const {
    assert(m_nodes.size() >= 1);
    return m_nodes[0].reward;
  }

  size_t rootNodeNumVisits() const {
    assert(m_nodes.size() >= 1);
    return m_nodes[0].number_of_visits;
  }

  std::vector<std::pair<Action, float>> valuePerAction() const {
    assert(m_nodes.size() >= 1);
    std::vector<std::pair<Action, float>> result;

    for (Node* c : m_nodes[0].children) {
      // float value = m_env.rewardToFloat(m_nodes[0].state, c->reward) / (float)c->number_of_visits;
      float value = c->number_of_visits / (float)m_nodes[0].number_of_visits;
      result.emplace_back(std::make_pair(c->action_to_node, value));
    }
    return result;
  }

  void exportToDot(std::ostream& stream) const
  {
    stream << "digraph MCTS {\n";
    stream << "\tnode[label=\"\"];\n";
    stream << "\tnode[shape=point];\n";

    // compute depth -> {idx} lookup table
    std::map<size_t, std::vector<size_t>> depthToIdx;
    for (size_t i = 0; i < m_nodes.size(); ++i) {
      size_t depth = m_nodes[i].computeDepth();
      depthToIdx[depth].push_back(i);
    }

    // output nodes with the same rank (depth)
    for (const auto& iter : depthToIdx) {
      stream << "\t# rank " << iter.first << " with " << iter.second.size() << " nodes\n";
      stream << "\t{rank=same";
      for (size_t i : iter.second) {
        stream << ";n" << i;
      }
      stream << "}\n";
    }

    // output node value
    for (size_t i = 1; i < m_nodes.size(); ++i) {
      float value = m_env.rewardToFloat(m_nodes[i].parent->state, m_nodes[i].reward) / m_nodes[i].number_of_visits;
      stream << "\tn" << i << " [width=" << value << "]\n";
    }

    // output nodes
    for (size_t i = 0; i < m_nodes.size(); ++i) {
      if (m_nodes[i].parent) {
        size_t j = m_nodes[i].parent - &m_nodes[0];
        stream << "\tn" << j << " -> n" << i << "\n";
      }
    }
    stream << "}\n";
  }


 private:
  struct Node {
    Node()
        : state()
        , parent(nullptr)
        , action_to_node()
        , number_of_visits(0)
        , reward()
        , children()
        , gotActions(false)
        , untried_actions()
    {
    }

    State state;
    Node* parent;
    Action action_to_node;

    size_t number_of_visits;
    Reward reward;

    std::vector<Node*> children;
    bool gotActions;
    std::vector<Action> untried_actions;

    size_t computeDepth() const {
      size_t depth = 0;
      const Node* ptr = parent;
      while(ptr) {
        ptr = ptr->parent;
        ++depth;
      }
      return depth;
    }

  };

  Node* treePolicy(Node& node)
  {
    Node* nodePtr = &node;
    while (nodePtr && !m_env.isTerminal(nodePtr->state)) {
      // try to expand this node
      Node* child = expand(nodePtr);
      if (child != nullptr) {
        return child;
      } else {
        child = bestChild(nodePtr, m_Cp);
        if (child == nullptr) {
          return nodePtr;
        }
        nodePtr = child;
      }
    }
    return nodePtr;
  }

  Node* expand(Node* nodePtr)
  {
    // if this node was never attempted to be expanded, query potential actions first
    if (!nodePtr->gotActions) {
      nodePtr->untried_actions = m_env.computeValidActions(nodePtr->state);
      nodePtr->gotActions = true;
    }

    // try to expand until we either found a valid action or have no more valid actions
    m_nodes.resize(m_nodes.size() + 1);
    auto& newNode = m_nodes[m_nodes.size()-1];
    while (nodePtr->untried_actions.size() > 0) {
      // shuffle on demand
      std::uniform_int_distribution<int> dist(0, nodePtr->untried_actions.size() - 1);
      int idx = dist(m_generator);
      std::swap(nodePtr->untried_actions.back(), nodePtr->untried_actions.begin()[idx]);

      const auto& action = nodePtr->untried_actions.back();
      bool success = m_env.step(nodePtr->state, action, newNode.state);
      if (success) {
        newNode.parent = nodePtr;
        newNode.action_to_node = action;
        nodePtr->untried_actions.pop_back();
        nodePtr->children.push_back(&newNode);
        return &newNode;
      }
      nodePtr->untried_actions.pop_back();
    }
    // there was no valid expansion
    m_nodes.pop_back();
    return nullptr;
  }

  Node* bestChild(const Node* nodePtr, float Cp)
  {
#if 0
    // static to avoid dynamic allocation every call
    static std::vector<std::pair<float, Node*>> weightedChildren;
    static std::vector<Node*> bestChildren;

    if (nodePtr->children.size() > 0) {
      float bestValue = 0;
      weightedChildren.clear();
      weightedChildren.reserve(nodePtr->children.size());
      // compute the weights and record best value
      for (Node* c : nodePtr->children) {
        float value = m_env.rewardToFloat(nodePtr->state, c->reward) / c->number_of_visits + Cp * sqrtf(2 * logf(nodePtr->number_of_visits) / c->number_of_visits);
        assert(value >= 0);
        weightedChildren.emplace_back(std::make_pair(value, c));
        if (value > bestValue) {
          bestValue = value;
        }
      }
      // find all candidates with the best value
      bestChildren.clear();
      for (const auto& pair : weightedChildren) {
        if (pair.first == bestValue) {
          bestChildren.push_back(pair.second);
        }
      }
      // break ties randomly
      if (bestChildren.size() > 1) {
        std::uniform_int_distribution<int> dist(0, bestChildren.size() - 1);
        int idx = dist(m_generator);
        return bestChildren[idx];
      }
      return bestChildren[0];
    }
    return nullptr;
#else
    Node* result = nullptr;
    float bestValue = -1;
    for (Node* c : nodePtr->children) {
      float value = m_env.rewardToFloat(nodePtr->state, c->reward) / c->number_of_visits + Cp * sqrtf(2 * logf(nodePtr->number_of_visits) / c->number_of_visits);
      assert(value >= 0);
      if (value > bestValue) {
        bestValue = value;
        result = c;
      }
    }
    return result;
#endif
  }

  void backpropagation(Node* nodePtr, const Reward& reward)
  {
    do {
      nodePtr->number_of_visits += 1;
      nodePtr->reward += reward;
      nodePtr = nodePtr->parent;
    } while (nodePtr != nullptr);
  }



 private:
  Environment& m_env;
  URNG& m_generator;
  std::vector<Node> m_nodes;
  size_t m_num_nodes;
  float m_Cp;
};

}  // namespace libMultiRobotPlanning
