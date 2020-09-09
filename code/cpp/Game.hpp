#pragma once

#include <random>

// #include "robots/RobotState.hpp"
// #include "robots/RobotType.hpp"
#include "GameState.hpp"

#include "GLAS.hpp"

typedef std::pair<float, float> Reward;

Reward& operator+=(Reward& r1, const Reward& r2) {
  r1.first += r2.first;
  r1.second += r2.second;
  return r1;
}

Reward operator/(const Reward r, size_t scale) {
  Reward result;
  result.first = r.first / scale;
  result.second = r.second / scale;
  return result;
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


template<class Robot>
class Game {
 public:
  typedef typename Robot::Type RobotTypeT;
  typedef typename Robot::Action RobotActionT;
  typedef typename Robot::State RobotStateT;

  typedef GameState<Robot> GameStateT;
  typedef std::vector<RobotActionT> GameActionT;

  Game(
    const std::vector<RobotTypeT>& attackerTypes,
    const std::vector<RobotTypeT>& defenderTypes,
    float dt,
    const Eigen::Matrix<float, Robot::StateDim, 1>& goal,
    size_t maxDepth,
    std::default_random_engine& generator)
    : m_attackerTypes(attackerTypes)
    , m_defenderTypes(defenderTypes)
    , m_dt(dt)
    , m_goal(goal)
    , m_maxDepth(maxDepth)
    , m_generator(generator)
    , m_glas_a(generator)
    , m_glas_b(generator)
    , m_rollout_beta(0.0)
  {
  }

  bool step(const GameStateT& state, const GameActionT& action, GameStateT& nextState)
  {
    assert(state.attackers.size() == m_attackerTypes.size());
    assert(state.defenders.size() == m_defenderTypes.size());

    if (state.depth >= m_maxDepth) {
      return false;
    }

    size_t NumAttackers = state.attackers.size();
    size_t NumDefenders = state.defenders.size();

    // copy current state
    nextState = state;
    // update active robots
    if (state.turn == GameStateT::Turn::Attackers) {
      for (size_t i = 0; i < NumAttackers; ++i) {
        if (nextState.attackers[i].status == RobotStateT::Status::Active) {
          m_attackerTypes[i].step(nextState.attackers[i], action[i], m_dt, nextState.attackers[i]);
          if (!m_attackerTypes[i].isStateValid(nextState.attackers[i])) {
            return false;
          }
        }
      }
      nextState.turn = GameStateT::Turn::Defenders;
    } else {
      for (size_t i = 0; i < NumDefenders; ++i) {
        if (nextState.defenders[i].status == RobotStateT::Status::Active) {
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
      if (nextState.attackers[i].status == RobotStateT::Status::Active) {
        float distToGoalSquared = (nextState.attackers[i].position() - m_goal.template head<2>()).squaredNorm();
        if (distToGoalSquared <= m_attackerTypes[i].tag_radiusSquared) {
          // std::cout << "d2g " << distToGoalSquared << std::endl;
          nextState.attackers[i].status = RobotStateT::Status::ReachedGoal;
          nextState.activeMask &= ~(1 << i); // reset bit i
        }

        for (size_t j = 0; j < NumDefenders; ++j) {
          if (nextState.defenders[j].status == RobotStateT::Status::Active) {
            float distToDefenderSquared = (nextState.attackers[i].position() - nextState.defenders[j].position()).squaredNorm();
            if (distToDefenderSquared <= m_defenderTypes[j].tag_radiusSquared) {
              nextState.attackers[i].status = RobotStateT::Status::Captured;
              nextState.activeMask &= ~(1 << i); // reset bit i
            }
          }
        }
      }
    }

    // Collision check
    // This only affects active robots within their respective teams.
    // If robots from different teams get too close to each other, this is considered tagging.
    for (size_t i = 0; i < NumAttackers; ++i) {
      if (nextState.attackers[i].status == RobotStateT::Status::Active) {
        for (size_t j = i+1; j < NumAttackers; ++j) {
          if (nextState.attackers[j].status == RobotStateT::Status::Active) {
            float dist = (nextState.attackers[i].position() - nextState.attackers[j].position()).norm();
            float maxDist = m_attackerTypes[i].radius + m_attackerTypes[j].radius;
            if (dist < maxDist) {
              return false;
            }
          }
        }
      }
    }

    for (size_t i = 0; i < NumDefenders; ++i) {
      for (size_t j = i+1; j < NumDefenders; ++j) {
        float dist = (nextState.defenders[i].position() - nextState.defenders[j].position()).norm();
        float maxDist = m_defenderTypes[i].radius + m_defenderTypes[j].radius;
        if (dist < maxDist) {
          return false;
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

  bool isValid(const GameStateT& state)
  {
    assert(state.attackers.size() == m_attackerTypes.size());
    assert(state.defenders.size() == m_defenderTypes.size());

    size_t NumAttackers = state.attackers.size();
    size_t NumDefenders = state.defenders.size();

    for (size_t i = 0; i < NumAttackers; ++i) {
      if (state.attackers[i].status == RobotStateT::Status::Active) {
        if (!m_attackerTypes[i].isStateValid(state.attackers[i])) {
          return false;
        }
        for (size_t j = i+1; j < NumAttackers; ++j) {
          if (state.attackers[j].status == RobotStateT::Status::Active) {
            float dist = (state.attackers[i].position() - state.attackers[j].position()).norm();
            float maxDist = m_attackerTypes[i].radius + m_attackerTypes[j].radius;
            if (dist < maxDist) {
              return false;
            }
          }
        }
      }
    }

    for (size_t i = 0; i < NumDefenders; ++i) {
      if (!m_defenderTypes[i].isStateValid(state.defenders[i])) {
        return false;
      }
      for (size_t j = i+1; j < NumDefenders; ++j) {
        float dist = (state.defenders[i].position() - state.defenders[j].position()).norm();
        float maxDist = m_defenderTypes[i].radius + m_defenderTypes[j].radius;
        if (dist < maxDist) {
          return false;
        }
      }
    }
    return true;
  }

  bool isTerminal(const GameStateT& state)
  {
    return state.activeMask == 0;
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
    actions = getPossibleActions(state);
  }

  Reward rollout(const GameStateT& state)
  {
    GameStateT s = state;
    GameStateT nextState;

    std::uniform_real_distribution<float> dist(0.0,1.0);

    while (true) {
      bool valid = true;
      if (m_rollout_beta > 0 && dist(m_generator) < m_rollout_beta) {
        // Use NN if rollout_beta is > 0 probabilistically
        assert(m_glas_a.valid() && m_glas_b.valid());

        const auto action = computeActionsWithGLAS(m_glas_a, m_glas_b, s, m_goal, m_attackerTypes, m_defenderTypes, m_generator, false);

        // step twice (once for each player)
        valid &= step(s, action, nextState);
        valid &= step(nextState, action, nextState);
      } else {
        // use regular random rollout

        // compute and step for player1
        const auto& actions1 = getPossibleActions(s);

        std::uniform_int_distribution<int> dist1(0, actions1.size() - 1);
        int idx1 = dist1(m_generator);
        const auto& action1 = actions1[idx1];

        valid &= step(s, action1, nextState);

        // compute and step for player2
        const auto& actions2 = getPossibleActions(nextState);

        std::uniform_int_distribution<int> dist2(0, actions2.size() - 1);
        int idx2 = dist2(m_generator);
        const auto& action2 = actions2[idx2];

        valid &= step(nextState, action2, nextState);
      }

      if (valid) {
        s = nextState;
        if (isTerminal(s)) {
          break;
        }
      } else {
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
        case RobotStateT::Status::Active:
          cumulativeReward += 0.5;
          break;
        case RobotStateT::Status::ReachedGoal:
          cumulativeReward += 1.0;
          break;
        // 0 in the remaining case (robot has been tagged)
      }
    }

    return cumulativeReward / state.attackers.size();
  }

  float rolloutBeta() const {
    return m_rollout_beta;
  }

  void setRolloutBeta(float rollout_beta) {
    m_rollout_beta = rollout_beta;
  }

  GLAS<Robot::StateDim>& glasA()
  {
    return m_glas_a;
  }

  GLAS<Robot::StateDim>& glasB()
  {
    return m_glas_b;
  }

  const auto& attackerTypes() const
  {
    return m_attackerTypes;
  }

  const auto& defenderTypes() const
  {
    return m_defenderTypes;
  }

  const auto& goal() const
  {
    return m_goal;
  }

private:

  const std::vector<GameActionT>& getPossibleActions(const GameStateT& state)
  {
    // We could filter here the "valid" actions, but this is also checked in the "step" function
    if (state.turn == GameStateT::Turn::Attackers) {
      const auto& cache = m_possibleActionsAttackersMap.find(state.activeMask);
      if (cache == m_possibleActionsAttackersMap.end()) {
        // cache miss -> compute new action set
        m_possibleActionsAttackersMap[state.activeMask] = computeActions(state);
        return m_possibleActionsAttackersMap[state.activeMask];
      }
      else {
        // cache hit
        return cache->second;
      }
    } else {
      if (m_possibleActionsDefender.size() == 0) {
        m_possibleActionsDefender = computeActions(state);
      }
      return m_possibleActionsDefender;
    }
  }


  std::vector<GameActionT> computeActions(const GameStateT& state)
  {
    std::vector<GameActionT> actions;
    std::vector<std::vector<RobotActionT>> allActions;

    size_t NumAttackers = state.attackers.size();
    size_t NumDefenders = state.defenders.size();

    // generate actions for active robots only
    if (state.turn == GameStateT::Turn::Attackers) {
      for (size_t i = 0; i < NumAttackers; ++i) {
        if (state.attackers[i].status == RobotStateT::Status::Active) {
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
      actions[i].resize(cartActions[i].size());
      for (size_t j = 0; j < NumAttackers + NumDefenders; ++j) {
        actions[i][j] = cartActions[i][j];
      }
    }

    return actions;
  }



private:
  std::vector<RobotTypeT> m_attackerTypes;
  std::vector<RobotTypeT> m_defenderTypes;
  float m_dt;
  Eigen::Matrix<float, Robot::StateDim, 1> m_goal;
  size_t m_maxDepth;
  std::default_random_engine& m_generator;
  GLAS<Robot::StateDim> m_glas_a;
  GLAS<Robot::StateDim> m_glas_b;
  float m_rollout_beta;

  // Maps activeRobots -> possible actions
  std::unordered_map<uint32_t, std::vector<GameActionT>> m_possibleActionsAttackersMap;
  std::vector<GameActionT> m_possibleActionsDefender;
};