#pragma once

#include <random>

// #include "robots/RobotState.hpp"
// #include "robots/RobotType.hpp"
#include "GameState.hpp"

#include "Policy.hpp"

#define REWARD_MODEL_BASIC_TERMINAL         1
#define REWARD_MODEL_TIME_EXPANDED_TERMINAL 2
#define REWARD_MODEL_CUMULATIVE             3

#define REWARD_MODEL REWARD_MODEL_BASIC_TERMINAL

#define ROLLOUT_MODE_POLICY          1
#define ROLLOUT_MODE_RANDOM          2
#define ROLLOUT_MODE_VALUE_RANDOM    3
#define ROLLOUT_MODE_VALUE_POLICY    4

#define ROLLOUT_MODE ROLLOUT_MODE_VALUE_POLICY

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
  typedef Policy<Robot> PolicyT;

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
            nextState.attackers[i].state.fill(nanf(""));
          }
        }
      }

      nextState.turn = GameStateT::Turn::Attackers;
      for (size_t i = 0; i < NumDefenders; ++i) {
        if (state.defenders[i].status == RobotStateT::Status::Active) {
          nextState.turn = GameStateT::Turn::Defenders;
          break;
        }
      }

    } else {
      for (size_t i = 0; i < NumDefenders; ++i) {
        if (nextState.defenders[i].status == RobotStateT::Status::Active) {
          m_defenderTypes[i].step(nextState.defenders[i], action[NumAttackers + i], m_dt, nextState.defenders[i]);
          if (!m_defenderTypes[i].isStateValid(nextState.defenders[i])) {
            nextState.defenders[i].status = RobotStateT::Status::Invalid;
            nextState.defenders[i].state.fill(nanf(""));
          }
        }
      }

      nextState.turn = GameStateT::Turn::Defenders;
      for (size_t i = 0; i < NumAttackers; ++i) {
        if (state.attackers[i].status == RobotStateT::Status::Active) {
          nextState.turn = GameStateT::Turn::Attackers;
          break;
        }
      }
    }

    // Update status
    for (size_t i = 0; i < NumAttackers; ++i) {
      if (nextState.attackers[i].status == RobotStateT::Status::Active) {
        float distToGoalSquared = (nextState.attackers[i].position() - m_goal.template head<2>()).squaredNorm();
        if (distToGoalSquared <= m_attackerTypes[i].tag_radiusSquared) {
          // std::cout << "d2g " << distToGoalSquared << std::endl;
          nextState.attackers[i].status = RobotStateT::Status::ReachedGoal;
          // mark the robot as not active anymore
          nextState.attackers[i].state.fill(nanf(""));
        }

        for (size_t j = 0; j < NumDefenders; ++j) {
          if (nextState.defenders[j].status == RobotStateT::Status::Active) {
            float distToDefenderSquared = (nextState.attackers[i].position() - nextState.defenders[j].position()).squaredNorm();
            if (distToDefenderSquared <= m_defenderTypes[j].tag_radiusSquared) {
              nextState.attackers[i].status = RobotStateT::Status::Captured;
              // mark the robot as not active anymore
              nextState.attackers[i].state.fill(nanf(""));
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
              nextState.attackers[i].state.fill(nanf(""));
              nextState.attackers[j].status = RobotStateT::Status::Invalid;
              nextState.attackers[j].state.fill(nanf(""));
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
              nextState.defenders[i].state.fill(nanf(""));
              nextState.defenders[j].status = RobotStateT::Status::Invalid;
              nextState.defenders[j].state.fill(nanf(""));
            }
          }
        }
      }
    }

    nextState.depth += 1;

    nextState.cumulativeReward += computeReward(nextState);

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
    // bool anyActive = false;
    // for (const auto& attacker : state.attackers) {
    //   if (attacker.status == RobotState::Status::ReachedGoal) {
    //     return true;
    //   }
    //   if (attacker.status == RobotState::Status::Active) {
    //     anyActive = true;
    //   }
    // }

    // i) all robot reached the goal
    bool allAtGoal = true;
    bool anyActive = false;
    for (const auto& attacker : state.attackers) {
      if (attacker.status != RobotState::Status::ReachedGoal) {
        allAtGoal = false;
      }

      if (attacker.status == RobotState::Status::Active) {
        anyActive = true;
      }
    }
    if (allAtGoal) {
      return true; 
    }

    // ii) no attacker is active anymore
    if (state.attackers.size() > 0 && !anyActive) {
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

  GameActionT sampleAction(
    const GameStateT& state,
    const PolicyT& policyAttacker,
    const PolicyT& policyDefender,
    bool deterministic,
    float beta2 = -1)
  {
    size_t NumAttackers = state.attackers.size();
    size_t NumDefenders = state.defenders.size();

    GameActionT result(NumAttackers + NumDefenders);

    if (state.turn == GameStateT::Turn::Attackers) {
      for (size_t i = 0; i < NumAttackers; ++i) {
        result[i] = policyAttacker.sampleAction(state.attackers[i], m_attackerTypes[i], true, i, state, m_goal, deterministic, beta2);
      }
      for (size_t i = 0; i < NumDefenders; ++i) {
        result[NumAttackers + i] = m_defenderTypes[i].invalidAction;
      }
    } else {
      for (size_t i = 0; i < NumAttackers; ++i) {
        result[i] = m_attackerTypes[i].invalidAction;
      }
      for (size_t i = 0; i < NumDefenders; ++i) {
        result[NumAttackers + i] = policyDefender.sampleAction(state.defenders[i], m_defenderTypes[i], false, i, state, m_goal, deterministic, beta2);
      }
    }
    return result;
  }

  Reward rollout(
    const GameStateT& state,
    const PolicyT& policyAttacker,
    const PolicyT& policyDefender,
    bool deterministic,
    float beta3)
  {
#if ROLLOUT_MODE == ROLLOUT_MODE_VALUE_POLICY || ROLLOUT_MODE == ROLLOUT_MODE_VALUE_RANDOM
    std::uniform_real_distribution<float> dist(0.0,1.0);
    if (   policyAttacker.glasConst().valid()
        && policyDefender.glasConst().valid()
        && !isTerminal(state)
        && dist(m_generator) < beta3) {
      float reward =  estimateValue(state, policyAttacker, policyDefender);
      return Reward(reward, 1 - reward);
    }
#endif

    GameStateT s = state;
    GameStateT nextState;

    while (true) {
#if ROLLOUT_MODE == ROLLOUT_MODE_POLICY || ROLLOUT_MODE == ROLLOUT_MODE_VALUE_POLICY
      const auto action = sampleAction(s, policyAttacker, policyDefender, deterministic);
#elif ROLLOUT_MODE == ROLLOUT_MODE_RANDOM || ROLLOUT_MODE == ROLLOUT_MODE_VALUE_RANDOM
      const auto action = sampleAction(s, policyAttacker, policyDefender, deterministic, 0.0);
#endif

      bool valid = step(s, action, nextState);

      if (valid) {
        s = nextState;
        if (isTerminal(s)) {
          break;
        }
      } else {
        break;
      }
    }

#if REWARD_MODEL == REWARD_MODEL_CUMULATIVE
     // propagate reward for remaining timesteps
    float reward = s.cumulativeReward;
    assert(s.depth <= m_maxDepth + 1);
    size_t remainingTimesteps = m_maxDepth + 1 - s.depth;
    if (remainingTimesteps > 0) {
      float r = computeReward(s);
      reward += remainingTimesteps * r;
    }
    reward /= m_maxDepth;
#endif // REWARD_MODEL_CUMULATIVE

#if REWARD_MODEL == REWARD_MODEL_TIME_EXPANDED_TERMINAL
    // Option 2: accumulate terminal reward
    assert(s.depth <= m_maxDepth + 1);
    size_t remainingTimesteps = m_maxDepth + 1 - s.depth;
    float reward = computeReward(s) * (remainingTimesteps + 1);
    reward /= m_maxDepth;
#endif // REWARD_MODEL == REWARD_MODEL_TIME_EXPANDED_TERMINAL

#if REWARD_MODEL == REWARD_MODEL_BASIC_TERMINAL
    float reward = computeReward(s);
#endif // REWARD_MODEL == REWARD_MODEL_BASIC_TERMINAL

    return Reward(reward, 1 - reward);
  }

// private:
  float computeReward(const GameStateT& state)
  {
    // std::cout << "compRew " << state << std::endl;
    // our reward is the closest distance to the goal
    // float minDistToGoal = std::numeric_limits<float>::infinity();
    // float minDistToGoal = (1.41 * 0.25);
    // for (const auto& attacker : state.attackers) {
    //   float distToGoal = (attacker.position() - m_goal.template head<2>()).norm();
    //   minDistToGoal = std::min(minDistToGoal, distToGoal);
    // }
    // return expf(10.0 * -minDistToGoal);
    // return 1.0 - minDistToGoal /(1.41 * 0.25);

    int numAttackerActive = 0;
    float reachedGoal = 0.0;
    for (const auto& attacker : state.attackers) {
      if (   attacker.status == RobotStateT::Status::Active
          || attacker.status == RobotStateT::Status::ReachedGoal) {
        ++numAttackerActive;
      }
      if (attacker.status == RobotStateT::Status::ReachedGoal) {
        // minDistToGoal = 0.0;
        // reachedGoal = 1;
        reachedGoal = reachedGoal + 1.0 / (float)state.attackers.size();
      }
    }

    int numDefendersActive = 0;
    for (const auto& defender : state.defenders) {
      if (   defender.status == RobotStateT::Status::Active) {
        ++numDefendersActive;
      }
    }

    float r1 = 0.0;
    if (state.attackers.size() > 0){
      r1 = numAttackerActive / (float)state.attackers.size();
    }

    float r2 = 0.0;
    if (state.defenders.size() > 0){
      r2 = (1.0f - numDefendersActive / (float)state.defenders.size());
    }

    return ( r1 + r2 + reachedGoal ) / 3.0f;    

    // return (   numAttackerActive / (float)state.attackers.size()
    //          + (1.0f - numDefendersActive / (float)state.defenders.size())
    //          + reachedGoal) / 3.0f;
          // + (1.0 - minDistToGoal /(1.41 * 0.25))) / 3;


    // int numActive = 0;
    // int numCaptured = 0;
    // for (const auto& attacker : state.attackers) {
    //   if (attacker.status == RobotStateT::Status::ReachedGoal) {
    //     return 1;
    //   }
    //   if (attacker.status == RobotStateT::Status::Active) {
    //     ++numActive;
    //   }
    //   if (attacker.status == RobotStateT::Status::Captured) {
    //     ++numCaptured;
    //   }
    // }

    // // 'old' reward: defender wins if no attacker is active

    // // if (numActive == 0) {
    // //   return 0;
    // // }

    // // 'new' reward: defender wins if all attackers are tagged

    // if (numCaptured == state.attackers.size()) {
    //   return 0;
    // }

    // // Tie otherwise (essentially timeout)
    // return 0.5;
  }

  float estimateValue(
    const GameStateT& state,
    const PolicyT& policyAttacker,
    const PolicyT& policyDefender)
  {
    if (state.turn == GameStateT::Turn::Defenders)
    {
      if (policyAttacker.glasConst().valid()) {
        // we check the turn on the child node, so in this case compute the reward
        // from the perspective of attackers
        float value_sum = 0;
        for (size_t j = 0; j < state.attackers.size(); ++j) {
          auto result_a = policyAttacker.glasConst().eval(state, m_goal, m_attackerTypes[j], true, j, true);
          float value = std::get<0>(result_a);
          value_sum += value;
        }
        return value_sum / state.attackers.size();
      }
    } else {
      if (policyDefender.glasConst().valid()) {
        float value_sum = 0;
        for (size_t j = 0; j < state.defenders.size(); ++j) {
          auto result_b = policyDefender.glasConst().eval(state, m_goal, m_defenderTypes[j], false, j, true);
          float value = std::get<0>(result_b);
          value_sum += value;
        }
        return 1.0 - value_sum / state.defenders.size();
      }
    }
    return std::nanf("");
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
  std::vector<RobotTypeT> m_attackerTypes;
  std::vector<RobotTypeT> m_defenderTypes;
  float m_dt;
  Eigen::Matrix<float, Robot::StateDim, 1> m_goal;
  size_t m_maxDepth;
  std::default_random_engine& m_generator;
};