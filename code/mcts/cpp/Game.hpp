#pragma once

#include <random>

#include "RobotState.hpp"
#include "RobotType.hpp"
#include "GameState.hpp"

typedef std::pair<float, float> Reward;

Reward& operator+=(Reward& r1, const Reward& r2) {
  r1.first += r2.first;
  r1.second += r2.second;
  return r1;
}

template <class T>
std::vector<std::vector<T>> cart_product(const std::vector<std::vector<T>>& v)
{
  std::vector<std::vector<T>> s = {{}};
  for (const auto& u : v) {
    std::vector<std::vector<T>> r;
    for (const auto& x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}


template <std::size_t NumAttackers, std::size_t NumDefenders>
class Game {
 public:
  typedef GameState<NumAttackers, NumDefenders> GameStateT;
  typedef std::array<RobotAction, NumAttackers + NumDefenders> GameActionT;

  Game(
    const std::array<RobotType, NumAttackers>& attackerTypes,
    const std::array<RobotType, NumDefenders>& defenderTypes,
    float dt,
    const Eigen::Vector2f& goal,
    size_t maxDepth,
    std::default_random_engine& generator)
    : m_attackerTypes(attackerTypes)
    , m_defenderTypes(defenderTypes)
    , m_dt(dt)
    , m_goal(goal)
    , m_maxDepth(maxDepth)
    , m_generator(generator)
  {
  }

  bool step(const GameStateT& state, const GameActionT& action, GameStateT& nextState)
  {
    if (state.depth >= m_maxDepth) {
      return false;
    }

    // copy current state
    nextState = state;
    // update active robots
    if (state.turn == GameStateT::Turn::Attackers) {
      for (size_t i = 0; i < NumAttackers; ++i) {
        if (nextState.attackers[i].status == RobotState::Status::Active) {
          m_attackerTypes[i].step(nextState.attackers[i], action[i], m_dt, nextState.attackers[i]);
          if (!m_attackerTypes[i].isStateValid(nextState.attackers[i])) {
            return false;
          }
        }
      }
      nextState.turn = GameStateT::Turn::Defenders;
    } else {
      for (size_t i = 0; i < NumDefenders; ++i) {
        if (nextState.defenders[i].status == RobotState::Status::Active) {
          m_defenderTypes[i].step(nextState.defenders[i], action[NumAttackers + i], m_dt, nextState.defenders[i]);
          if (!m_defenderTypes[i].isStateValid(nextState.defenders[i])) {
            return false;
          }
        }
      }
      nextState.turn = GameStateT::Turn::Attackers;
    }
    // Update status
    for (size_t i = 0; i < NumAttackers; ++i) {
      if (nextState.attackers[i].status == RobotState::Status::Active) {
        float distToGoalSquared = (nextState.attackers[i].position - m_goal).squaredNorm();
        if (distToGoalSquared <= m_attackerTypes[i].tag_radiusSquared) {
          // std::cout << "d2g " << distToGoalSquared << std::endl;
          nextState.attackers[i].status = RobotState::Status::ReachedGoal;
          nextState.activeMask.reset(i);
        }

        for (size_t j = 0; j < NumDefenders; ++j) {
          if (nextState.defenders[j].status == RobotState::Status::Active) {
            float distToDefenderSquared = (nextState.attackers[i].position - nextState.defenders[j].position).squaredNorm();
            if (distToDefenderSquared <= m_defenderTypes[j].tag_radiusSquared) {
              nextState.attackers[i].status = RobotState::Status::Captured;
              nextState.activeMask.reset(i);
            }
          }
        }
      }
    }

    // update accumulated reward
    float r = computeReward(nextState);
    nextState.attackersReward += r;
    nextState.defendersReward += (1 - r);

    nextState.depth += 1;

    return true;
  }

  bool isTerminal(const GameStateT& state)
  {
    return state.activeMask.none();
    // for (const auto& attacker : state.attackers) {
    //   if (attacker.status == RobotState::Status::Active) {
    //     return false;
    //   }
    // }
    // return true;
  }

  float rewardToFloat(const GameStateT& state, const Reward& reward)
  {
    if (state.turn == GameStateT::Turn::Attackers) {
      return reward.first;
    }
    return reward.second;
  }

  void getPossibleActions(const GameStateT& state, std::vector<GameActionT>& actions)
  {
    // We could filter here the "valid" actions, but this is also checked in the "step" function
    actions.clear();

    if (state.turn == GameStateT::Turn::Attackers) {
      const auto& cache = m_possibleActionsAttackersMap.find(state.activeMask);
      if (cache == m_possibleActionsAttackersMap.end()) {
        // cache miss -> compute new action set
        actions = computeActions(state);
        // store result in cache
        m_possibleActionsAttackersMap[state.activeMask] = actions;
      }
      else {
        // cache hit -> copy to output
        actions = cache->second;
      }
    } else {
      if (m_possibleActionsDefender.size() == 0) {
        m_possibleActionsDefender = computeActions(state);
      }
      actions = m_possibleActionsDefender;
    }
  }

  Reward rollout(const GameStateT& state)
  {
    // float reward = computeReward(state);
    GameStateT s = state;
    while (true) {
      std::vector<GameActionT> actions;
      getPossibleActions(s, actions);

      while (actions.size() > 0) {
        // shuffle on demand
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        int idx = dist(m_generator);
        std::swap(actions.back(), actions.begin()[idx]);

        const auto& action = actions.back();
        GameStateT nextState;
        bool valid = step(s, action, nextState);
        if (valid) {
          s = nextState;
          break;
        }
        actions.pop_back();
      }

      if (actions.size() == 0) {
        break;
      }
    }
    float attackersReward = s.attackersReward;
    float defendersReward = s.defendersReward;

    // propagate reward for remaining timesteps
    assert(s.depth <= m_maxDepth);
    size_t remainingTimesteps = m_maxDepth - s.depth;
    if (remainingTimesteps > 0 && isTerminal(s)) {
      float r = computeReward(s);
      attackersReward += remainingTimesteps * r;
      defendersReward += remainingTimesteps * (1-r);
    }

    return Reward(attackersReward / m_maxDepth, defendersReward / m_maxDepth);
  }

// private:
  float computeReward(const GameStateT& state)
  {
    float cumulativeReward = 0.0;
    for (const auto& attacker : state.attackers) {
      switch(attacker.status) {
        case RobotState::Status::Active:
          cumulativeReward += 0.5;
          break;
        case RobotState::Status::ReachedGoal:
          cumulativeReward += 1.0;
          break;
        // 0 in the remaining case (robot has been tagged)
      }
    }

    return cumulativeReward / NumAttackers;
  }

private:

  std::vector<GameActionT> computeActions(const GameStateT& state)
  {
    std::vector<GameActionT> actions;
    std::vector<std::vector<RobotAction>> allActions;

    // generate actions for active robots only
    if (state.turn == GameStateT::Turn::Attackers) {
      for (size_t i = 0; i < NumAttackers; ++i) {
        if (state.attackers[i].status == RobotState::Status::Active) {
          allActions.push_back(m_attackerTypes[i].possibleActions);
        } else {
          allActions.push_back({m_attackerTypes[i].invalidAction});
        }
      }
      for (size_t i = 0; i < NumDefenders; ++i) {
        allActions.push_back({m_defenderTypes[i].invalidAction});
      }
    } else {
      for (size_t i = 0; i < NumAttackers; ++i) {
        allActions.push_back({m_attackerTypes[i].invalidAction});
      }
      for (size_t i = 0; i < NumDefenders; ++i) {
        allActions.push_back(m_defenderTypes[i].possibleActions);
      }
    }

    // compute cartesian product
    const auto& cartActions = cart_product(allActions);

    // convert to std::vector<GameActionT>
    actions.resize(cartActions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
      assert(cartActions[i].size() == NumAttackers + NumDefenders);
      for (size_t j = 0; j < NumAttackers + NumDefenders; ++j) {
        actions[i][j] = cartActions[i][j];
      }
    }

    return actions;
  }



private:
  const std::array<RobotType, NumAttackers>& m_attackerTypes;
  const std::array<RobotType, NumDefenders>& m_defenderTypes;
  float m_dt;
  Eigen::Vector2f m_goal;
  size_t m_maxDepth;
  std::default_random_engine& m_generator;

  // Maps activeRobots -> possible actions
  std::unordered_map<std::bitset<NumAttackers>, std::vector<GameActionT>> m_possibleActionsAttackersMap;
  std::vector<GameActionT> m_possibleActionsDefender;
};