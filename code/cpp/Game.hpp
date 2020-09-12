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

    // Note: The current step function never returns false, but instead marks
    //       robots as "invalid" if they attempt to execute an invalid action.

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
            nextState.attackers[i].status = RobotStateT::Status::Invalid;
          }
        }
      }
      nextState.turn = GameStateT::Turn::Defenders;
    } else {
      for (size_t i = 0; i < NumDefenders; ++i) {
        if (nextState.defenders[i].status == RobotStateT::Status::Active) {
          m_defenderTypes[i].step(nextState.defenders[i], action[NumAttackers + i], m_dt, nextState.defenders[i]);
          if (!m_defenderTypes[i].isStateValid(nextState.defenders[i])) {
            nextState.defenders[i].status = RobotStateT::Status::Invalid;
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
        }

        for (size_t j = 0; j < NumDefenders; ++j) {
          if (nextState.defenders[j].status == RobotStateT::Status::Active) {
            float distToDefenderSquared = (nextState.attackers[i].position() - nextState.defenders[j].position()).squaredNorm();
            if (distToDefenderSquared <= m_defenderTypes[j].tag_radiusSquared) {
              nextState.attackers[i].status = RobotStateT::Status::Captured;
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
              // mark both involved robot as collided
              nextState.attackers[i].status = RobotStateT::Status::Invalid;
              nextState.attackers[j].status = RobotStateT::Status::Invalid;
            }
          }
        }
      }
    }

    for (size_t i = 0; i < NumDefenders; ++i) {
      if (nextState.defenders[i].status == RobotStateT::Status::Active) {
        for (size_t j = i+1; j < NumDefenders; ++j) {
          if (nextState.defenders[j].status == RobotStateT::Status::Active) {
            float dist = (nextState.defenders[i].position() - nextState.defenders[j].position()).norm();
            float maxDist = m_defenderTypes[i].radius + m_defenderTypes[j].radius;
            if (dist < maxDist) {
              // mark both involved robot as collided
              nextState.defenders[i].status = RobotStateT::Status::Invalid;
              nextState.defenders[j].status = RobotStateT::Status::Invalid;
            }
          }
        }
      }
    }

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
    // Done if:
    // i) one robot reached the goal
    bool anyActive = false;
    for (const auto& attacker : state.attackers) {
      if (attacker.status == RobotState::Status::ReachedGoal) {
        return true;
      }
      if (attacker.status == RobotState::Status::Active) {
        anyActive = true;
      }
    }
    // ii) no attacker is active anymore
    if (!anyActive) {
      return true;
    }
    // iii) maximum time horizon reached
    if (state.depth >= m_maxDepth) {
      return true;
    }

    return false;
  }

  float rewardToFloat(const GameStateT& state, const Reward& reward)
  {
    if (state.turn == GameStateT::Turn::Attackers) {
      return reward.first;
    }
    return reward.second;
  }

  std::vector<GameActionT> computeValidActions(const GameStateT& state)
  {
    std::vector<std::vector<RobotActionT>> allActions;
    computeValidActionsPerRobot(state, allActions);

    // compute cartesian product
    const auto cartActions = cart_product(allActions);
    return cartActions;
  }

  Reward rollout(const GameStateT& state)
  {
    GameStateT s = state;
    GameStateT nextState;

    std::uniform_real_distribution<float> dist(0.0,1.0);
    std::vector<std::vector<RobotActionT>> actions;

    while (true) {
      bool valid = true;
      if (m_rollout_beta > 0 && dist(m_generator) < m_rollout_beta) {
        // Use NN if rollout_beta is > 0 probabilistically
        assert(m_glas_a.valid() && m_glas_b.valid());

        const auto action = computeActionsWithGLAS(m_glas_a, m_glas_b, s, m_goal, m_attackerTypes, m_defenderTypes, m_dt, false);
        // TODO: need some logic here to only allow valid actions...

        // step twice (once for each player)
        valid &= step(s, action, nextState);
        valid &= step(nextState, action, nextState);
      } else {
        // use regular random rollout

        // compute and step for player1
        computeValidActionsPerRobot(s, actions);

        std::uniform_int_distribution<int> dist1(0, cartProductSize(actions) - 1);
        int idx1 = dist1(m_generator);
        const auto& action1 = getCartProduct(actions, idx1);

        valid &= step(s, action1, nextState);

        // compute and step for player2
        computeValidActionsPerRobot(nextState, actions);

        std::uniform_int_distribution<int> dist2(0, cartProductSize(actions) - 1);
        int idx2 = dist2(m_generator);
        const auto& action2 = getCartProduct(actions, idx2);

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
    float r = computeReward(s);
    return Reward(r, 1 - r);
  }

// private:
  float computeReward(const GameStateT& state)
  {
    // // our reward is the closest distance to the goal
    // float minDistToGoal = std::numeric_limits<float>::infinity();
    // for (const auto& attacker : state.attackers) {
    //   float distToGoal = (attacker.position() - m_goal.template head<2>()).norm();
    //   minDistToGoal = std::min(minDistToGoal, distToGoal);
    // }
    // return expf(10.0 * -minDistToGoal);
    int numActive = 0;
    for (const auto& attacker : state.attackers) {
      if (attacker.status == RobotStateT::Status::ReachedGoal) {
        return 1;
      }
      if (attacker.status == RobotStateT::Status::Active) {
        ++numActive;
      }
    }

    if (numActive == 0) {
      return 0;
    }

    // Tie otherwise (essentially timeout)
    return 0.5;
  }

  float rolloutBeta() const {
    return m_rollout_beta;
  }

  void setRolloutBeta(float rollout_beta) {
    m_rollout_beta = rollout_beta;
  }

  GLAS<Robot>& glasA()
  {
    return m_glas_a;
  }

  GLAS<Robot>& glasB()
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

  float dt() const
  {
    return m_dt;
  }

private:

  void computeValidActions(const RobotStateT& robotState, const RobotTypeT& robotType, std::vector<RobotActionT>& result) const
  {
    result.clear();
    RobotStateT nextRobotState;

    if (robotState.status == RobotStateT::Status::Active) {
      for (const auto& action : robotType.possibleActions) {
        robotType.step(robotState, action, m_dt, nextRobotState);
        if (robotType.isStateValid(nextRobotState)) {
          result.push_back(action);
        }
      }
    }
  }

  void computeValidActionsPerRobot(const GameStateT& state, std::vector<std::vector<RobotActionT>>& result)
  {
    size_t NumAttackers = state.attackers.size();
    size_t NumDefenders = state.defenders.size();

    result.resize(NumAttackers+NumDefenders);

    // generate valid actions
    if (state.turn == GameStateT::Turn::Attackers) {
      for (size_t i = 0; i < NumAttackers; ++i) {
        computeValidActions(state.attackers[i], m_attackerTypes[i], result[i]);
        if (result[i].empty()) {
          result[i].push_back(m_attackerTypes[i].invalidAction);
        }
      }
      for (size_t i = 0; i < NumDefenders; ++i) {
        result[NumAttackers + i] = {m_defenderTypes[i].invalidAction};
      }
    } else {
      for (size_t i = 0; i < NumAttackers; ++i) {
        result[i].push_back(m_attackerTypes[i].invalidAction);
      }
      for (size_t i = 0; i < NumDefenders; ++i) {
        computeValidActions(state.defenders[i], m_defenderTypes[i], result[NumAttackers + i]);
        if (result[NumAttackers + i].empty()) {
          result[NumAttackers + i].push_back(m_defenderTypes[i].invalidAction);
        }
      }
    }
  }

  size_t cartProductSize(const std::vector<std::vector<RobotActionT>>& allActions)
  {
    size_t result = 1;
    for (const auto& v : allActions) {
      result *= v.size();
    }
    return result;
  }

  GameActionT getCartProduct(const std::vector<std::vector<RobotActionT>>& allActions, size_t idx)
  {
    size_t NumRobots = allActions.size();
    GameActionT action(NumRobots);

    // Example:
    // num robots = 3. sizes: [[a,b,c], [i,ii], [1]] => num actions = 3*2*1 = 6
    // [a,i,1], [b,i,1], [c,i,1], [a,ii,1], [b,ii,1], [c,ii,1]
    for (size_t i = 0; i < NumRobots; ++i) {
      action[i] = allActions[i][idx % allActions[i].size()];
      idx /= allActions[i].size();
    }
    return action;
  }

  std::vector<GameActionT> cart_product(const std::vector<std::vector<RobotActionT>>& v)
  {
    std::vector<GameActionT> s = {{}};
    for (const auto& u : v) {
      std::vector<GameActionT> r;
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

private:
  std::vector<RobotTypeT> m_attackerTypes;
  std::vector<RobotTypeT> m_defenderTypes;
  float m_dt;
  Eigen::Matrix<float, Robot::StateDim, 1> m_goal;
  size_t m_maxDepth;
  std::default_random_engine& m_generator;
  GLAS<Robot> m_glas_a;
  GLAS<Robot> m_glas_b;
  float m_rollout_beta;
};