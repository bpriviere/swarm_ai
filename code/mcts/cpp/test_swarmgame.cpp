#include <iostream>
#include <array>
#include <Eigen/Dense>
#include <fstream>

#include "monte_carlo_tree_search.hpp"
#include "eigen_helper.hpp"

typedef Eigen::Vector2f RobotAction;

///

// template <class T, std::size_t Dim>
// std::vector<std::array<T,Dim>> cart_product(const std::array<std::vector<T>, Dim>& v)
// {
//   std::vector<std::array<T,Dim>> s = {{}};
//   // for (const auto& u : v) {
//   for (size_t d = 0; d < Dim; ++d) {
//     std::vector<std::array<T,Dim>> r;
//     for (const auto& x : s) {
//       for (const auto y : v[d]) {
//         r.push_back(x);
//         r.back().push_back(y);
//       }
//     }
//     s = std::move(r);
//   }
//   return s;
// }


struct RobotState
{
public:
  enum class Status
  {
    Active      = 0,
    Captured    = 1,
    ReachedGoal = 2,
  };

  Eigen::Vector2f position; // m
  Eigen::Vector2f velocity; // m/s
  Status status;
};

std::ostream& operator<<(std::ostream& out, const RobotState& s)
{
  Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

  out << "RobotState(p=" << s.position.format(fmt) << ",v=" << s.velocity.format(fmt) << ",";
  switch(s.status) {
    case RobotState::Status::Active:
      out << "Active";
      break;
    case RobotState::Status::Captured:
      out << "Captured";
      break;
    case RobotState::Status::ReachedGoal:
      out << "ReachedGoal";
      break;
  }
  out << ")";
  return out;
}

class RobotType
{
public:
  Eigen::Vector2f p_min;
  Eigen::Vector2f p_max;
  float velocity_limit;
  float acceleration_limit;
  std::vector<RobotAction> possibleActions;
  // RobotAction invalidAction;

  void step(const RobotState& state, const RobotAction& action, float dt, RobotState& result) const
  {
    result.position = state.position + state.velocity * dt;
    Eigen::Vector2f velocity = state.velocity + action * dt;

    float alpha = velocity.norm() / velocity_limit;
    result.velocity = velocity / std::max(alpha, 1.0f);
    // result.velocity = clip(velocity, -velocity_limit, velocity_limit);
    // result.velocity = velocity.cwiseMin(velocity_limit).cwiseMax(-velocity_limit);
  }

  bool isStateValid(const RobotState& state) const
  {
    return (state.position.array() >= p_min.array()).all() && (state.position.array() <= p_max.array()).all();
  }

  void init()
  {
    possibleActions.resize(9);
    possibleActions[0] << -acceleration_limit, -acceleration_limit;
    possibleActions[1] << -acceleration_limit, 0;
    possibleActions[2] << -acceleration_limit, acceleration_limit;
    possibleActions[3] << 0, -acceleration_limit;
    possibleActions[4] << 0, 0;
    possibleActions[5] << 0, acceleration_limit;
    possibleActions[6] << acceleration_limit, -acceleration_limit;
    possibleActions[7] << acceleration_limit, 0;
    possibleActions[8] << acceleration_limit, acceleration_limit;

    // invalidAction << nanf("") , nanf("");
  }
};

template <std::size_t NumAttackers, std::size_t NumDefenders>
class GameState
{
public:
  enum class Turn
  {
    Attackers = 0,
    Defenders = 1,
  };

  Turn turn;
  std::array<RobotState, NumAttackers> attackers;
  float attackersReward;
  std::array<RobotState, NumDefenders> defenders;
  float defendersReward;

  size_t depth;
};

template <std::size_t NumAttackers, std::size_t NumDefenders>
std::ostream& operator<<(std::ostream& out, const GameState<NumAttackers, NumDefenders>& s)
{
  Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

  out << "GameState(turn=";
  switch(s.turn) {
    case GameState<NumAttackers, NumDefenders>::Turn::Attackers:
      out << "Attackers";
      break;
    case GameState<NumAttackers, NumDefenders>::Turn::Defenders:
      out << "Defenders";
      break;
  }
  out << ",attackers=";
  for(const auto& attacker : s.attackers) {
    out << attacker << ",";
  }
  out << "defenders=";
  for(const auto& defender : s.defenders) {
    out << defender << ",";
  }
  out << ")";
  return out;
}

typedef std::pair<float, float> Reward;

Reward operator+=(Reward& r1, const Reward& r2) {
  r1.first += r2.first;
  r1.second += r2.second;
  return r1;
}

template <std::size_t NumAttackers, std::size_t NumDefenders>
class Environment {
 public:
  typedef GameState<NumAttackers, NumDefenders> GameStateT;
  typedef std::array<RobotAction, NumAttackers + NumDefenders> GameActionT;

  Environment(
    const std::array<RobotType, NumAttackers>& attackerTypes,
    const std::array<RobotType, NumDefenders>& defenderTypes,
    float dt,
    const Eigen::Vector2f& goal,
    float goalRadius,
    float tagRadius,
    size_t maxDepth,
    std::default_random_engine& generator)
    : m_attackerTypes(attackerTypes)
    , m_defenderTypes(defenderTypes)
    , m_dt(dt)
    , m_goal(goal)
    , m_goalRadiusSquared(goalRadius * goalRadius)
    , m_tagRadiusSquared(tagRadius * tagRadius)
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
        if (distToGoalSquared <= m_goalRadiusSquared) {
          // std::cout << "d2g " << distToGoalSquared << std::endl;
          nextState.attackers[i].status = RobotState::Status::ReachedGoal;
        }

        for (size_t j = 0; j < NumDefenders; ++j) {
          if (nextState.defenders[j].status == RobotState::Status::Active) {
            float distToDefenderSquared = (nextState.attackers[i].position - nextState.defenders[j].position).squaredNorm();
            if (distToDefenderSquared <= m_tagRadiusSquared) {
              nextState.attackers[i].status = RobotState::Status::Captured;
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
    for (const auto& attacker : state.attackers) {
      if (attacker.status == RobotState::Status::Active) {
        return false;
      }
    }
    return true;
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

    // std::vector<std::vector<RobotAction>> allActions;

    // generate actions for active robots only
    if (state.turn == GameStateT::Turn::Attackers) {
      for (size_t i = 0; i < NumAttackers; ++i) {
        // if (nextState.attackers[i].status == RobotState::Status::Active) {
        //   allActions.push_back(m_attackerTypes[i].possibleActions);
        // } else {
        //   allActions.push_back.push_back(m_attackerTypes[i].invalidAction);
        // }

        // TODO: This logic only works for a single robot
        if (state.attackers[i].status == RobotState::Status::Active) {
          for (const auto& a : m_attackerTypes[i].possibleActions) {
            actions.resize(actions.size() + 1);
            actions.back()[i] = a;
          }
        }

      }
    } else {
      for (size_t i = 0; i < NumDefenders; ++i) {
        if (state.defenders[i].status == RobotState::Status::Active) {
          // if (nextState.defenders[i].status == RobotState::Status::Active) {
          //   allActions.push_back(m_defenderTypes[i].possibleActions);
          // } else {
          //   allActions.push_back.push_back(m_defenderTypes[i].invalidAction);
          // }

          // TODO: This logic only works for a single robot
          if (state.defenders[i].status == RobotState::Status::Active) {
            for (const auto& a : m_defenderTypes[i].possibleActions) {
              actions.resize(actions.size() + 1);
              actions.back()[NumAttackers + i] = a;
            }
          }
        }
      }
    }

    // auto cartActions = cart_product(allActions);
    // for (const auto& a : cartActions) {
    //   actions.push
    // }


  }

  Reward rollout(const GameStateT& state)
  {
    // float reward = computeReward(state);
    GameStateT s = state;
    while (true) {
      std::vector<GameActionT> actions;
      getPossibleActions(s, actions);
      std::shuffle(actions.begin(), actions.end(), m_generator);

      while (actions.size() > 0) {
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
  const std::array<RobotType, NumAttackers>& m_attackerTypes;
  const std::array<RobotType, NumDefenders>& m_defenderTypes;
  float m_dt;
  Eigen::Vector2f m_goal;
  float m_goalRadiusSquared;
  float m_tagRadiusSquared;
  size_t m_maxDepth;
  std::default_random_engine& m_generator;
};


int main(int argc, char* argv[]) {

  size_t num_nodes = 1000;

  std::random_device r;
  std::default_random_engine generator(r());
  // std::default_random_engine generator(0);

  GameState<1,1> state;
  state.turn = GameState<1,1>::Turn::Attackers;
  state.attackers[0].status = RobotState::Status::Active;
  state.attackers[0].position << 0.05,0.2;
  state.attackers[0].velocity << 0,0;
  // state.attackers[1].status = RobotState::Status::Active;
  // state.attackers[1].position << -1,0;
  // state.attackers[1].velocity << 0,0;

  state.defenders[0].status = RobotState::Status::Active;
  state.defenders[0].position << 0.45,0.2;
  state.defenders[0].velocity << 0,0;

  std::cout << state << std::endl;

  RobotType robotTypeAttacker;
  robotTypeAttacker.p_min << 0, 0;
  robotTypeAttacker.p_max << 0.5, 0.5;
  robotTypeAttacker.velocity_limit = 0.125 / sqrtf(2.0);
  robotTypeAttacker.acceleration_limit = 0.25 / sqrtf(2.0);
  robotTypeAttacker.init();

  RobotType robotTypeDefender;
  robotTypeDefender.p_min << 0, 0;
  robotTypeDefender.p_max << 0.5, 0.5;
  robotTypeDefender.velocity_limit = 0.125 / sqrtf(2.0);
  robotTypeDefender.acceleration_limit = 0.125 / sqrtf(2.0);
  robotTypeDefender.init();

  std::array<RobotType, 1> attackerTypes;
  attackerTypes[0] = robotTypeAttacker;
  std::array<RobotType, 1> defenderTypes;
  defenderTypes[0] = robotTypeDefender;

  float dt = 0.25;
  Eigen::Vector2f goal;
  goal << 0.45,0.375;
  float goalRadius = 0.025;
  float tagRadius = 0.025;

  size_t max_depth = 1000;

  Environment<1,1> env(attackerTypes, defenderTypes, dt, goal, goalRadius, tagRadius, max_depth, generator);

  libMultiRobotPlanning::MonteCarloTreeSearch<GameState<1,1>, std::array<RobotAction,2>, Reward, Environment<1,1>> mcts(env, generator, num_nodes, 1.4);

  std::ofstream out("output.csv");

  for(int i = 0; ; ++i) {
    state.attackersReward = 0;
    state.defendersReward = 0;
    state.depth = 0;
    if (i % 2 == 0) {
      out << state.attackers[0].position(0) << "," << state.attackers[0].position(1) << ","
          << state.attackers[0].velocity(0) << "," << state.attackers[0].velocity(1) << ","
          << state.defenders[0].position(0) << "," << state.defenders[0].position(1) << ","
          << state.defenders[0].velocity(0) << "," << state.defenders[0].velocity(1) << std::endl;
    }

    std::array<RobotAction,2> action;
    bool success = mcts.search(state, action);
    if (!success) {
      break;
    }
    env.step(state, action, state);
    float f = env.computeReward(state);
    std::cout << state << "reward: " << f << std::endl;
    // Reward r;
    // assert(r.first == 0);
    // assert(r.second == 0);
    // int i;
    // for (i = 0; i < 100000; ++i) {
    //   r += env.rollout(state);
    // }
    // std::cout << i << " " << r.first / i << " " << r.second / i << std::endl;
  }

  return 0;
}
