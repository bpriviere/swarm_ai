#include <iostream>
#include <array>
#include <Eigen/Dense>
#include <fstream>
#include <bitset>
#include <unordered_map>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

typedef std::pair<float, float> Reward;

Reward& operator+=(Reward& r1, const Reward& r2) {
  r1.first += r2.first;
  r1.second += r2.second;
  return r1;
}

#include "monte_carlo_tree_search.hpp"

typedef Eigen::Vector2f RobotAction;

///

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
  float tag_radiusSquared;
  std::vector<RobotAction> possibleActions;
  RobotAction invalidAction;

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
  std::bitset<NumAttackers> activeMask;
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

template <std::size_t NumAttackers, std::size_t NumDefenders>
void runMCTS(const YAML::Node& config, const std::string& outputFile)
{
  using EnvironmentT = Environment<NumAttackers, NumDefenders>;
  using GameStateT = typename EnvironmentT::GameStateT;
  using GameActionT = typename EnvironmentT::GameActionT;

  size_t num_nodes = config["tree_size"].as<int>();

  size_t seed;
  if (config["seed"]) {
    seed = config["seed"].as<size_t>();
  } else {
    std::random_device r;
    seed = r();
  }
  std::cout << "Using seed " << seed << std::endl;
  std::default_random_engine generator(seed);

  GameStateT state;

  state.turn = GameStateT::Turn::Attackers;
  state.activeMask.set();
  std::uniform_real_distribution<float> xPosDist(config["reset_xlim_A"][0].as<float>(),config["reset_xlim_A"][1].as<float>());
  std::uniform_real_distribution<float> yPosDist(config["reset_ylim_A"][0].as<float>(),config["reset_ylim_A"][1].as<float>());
  // std::uniform_real_distribution<float> velDist(-config["speed_limit_a"].as<float>() / sqrtf(2.0), config["speed_limit_a"].as<float>() / sqrtf(2.0));
  for (size_t i = 0; i < NumAttackers; ++i) {
    state.attackers[i].status = RobotState::Status::Active;
    state.attackers[i].position << xPosDist(generator),yPosDist(generator);
    // state.attackers[i].velocity << velDist(generator),velDist(generator);
    state.attackers[i].velocity << 0,0;
  }
  xPosDist = std::uniform_real_distribution<float>(config["reset_xlim_B"][0].as<float>(),config["reset_xlim_B"][1].as<float>());
  yPosDist = std::uniform_real_distribution<float>(config["reset_ylim_B"][0].as<float>(),config["reset_ylim_B"][1].as<float>());
  // velDist = std::uniform_real_distribution<float>(-config["speed_limit_b"].as<float>() / sqrtf(2.0), config["speed_limit_b"].as<float>() / sqrtf(2.0));
  for (size_t i = 0; i < NumDefenders; ++i) {
    state.defenders[i].status = RobotState::Status::Active;
    state.defenders[i].position << xPosDist(generator),yPosDist(generator);
    // state.defenders[i].velocity << velDist(generator),velDist(generator);
    state.defenders[i].velocity << 0,0;
  }

  std::cout << state << std::endl;

  std::array<RobotType, NumAttackers> attackerTypes;
  for (size_t i = 0; i < NumAttackers; ++i) {
    const auto& node = config["robots"][i];
    attackerTypes[i].p_min << config["env_xlim"][0].as<float>(), config["env_ylim"][0].as<float>();
    attackerTypes[i].p_max << config["env_xlim"][1].as<float>(), config["env_ylim"][1].as<float>();
    attackerTypes[i].velocity_limit = node["speed_limit"].as<float>(); // / sqrtf(2.0);
    attackerTypes[i].acceleration_limit = node["acceleration_limit"].as<float>() / sqrtf(2.0);
    attackerTypes[i].tag_radiusSquared = powf(node["tag_radius"].as<float>(), 2);
    attackerTypes[i].init();
  }
  std::array<RobotType, NumDefenders> defenderTypes;
  for (size_t i = 0; i < NumDefenders; ++i) {
    const auto& node = config["robots"][i+NumAttackers];
    defenderTypes[i].p_min << config["env_xlim"][0].as<float>(), config["env_ylim"][0].as<float>();
    defenderTypes[i].p_max << config["env_xlim"][1].as<float>(), config["env_ylim"][1].as<float>();
    defenderTypes[i].velocity_limit = node["speed_limit"].as<float>() / sqrtf(2.0);
    defenderTypes[i].acceleration_limit = node["acceleration_limit"].as<float>() / sqrtf(2.0);
    defenderTypes[i].tag_radiusSquared = powf(node["tag_radius"].as<float>(), 2);
    defenderTypes[i].init();
  }

  float dt = config["sim_dt"].as<float>();
  Eigen::Vector2f goal;
  goal << config["goal"][0].as<float>(),config["goal"][1].as<float>();

  size_t max_depth = config["rollout_horizon"].as<int>();

  EnvironmentT env(attackerTypes, defenderTypes, dt, goal, max_depth, generator);

  libMultiRobotPlanning::MonteCarloTreeSearch<GameStateT, GameActionT, Reward, EnvironmentT> mcts(env, generator, num_nodes, 1.4);

  std::ofstream out(outputFile);
  // write file header
  for (size_t j = 0; j < NumAttackers+NumDefenders; ++j) {
    out << "x,y,vx,vy,ax,ay,";
  }
  out << "rewardAttacker,rewardDefender" << std::endl;

  GameActionT action;
  GameActionT lastAction;
  GameStateT lastState = state;
  float rewardAttacker;
  float rewardDefender;
  for(int i = 0; ; ++i) {
    state.attackersReward = 0;
    state.defendersReward = 0;
    state.depth = 0;

    lastAction = action;
    bool success = mcts.search(state, action);
    if (state.turn == GameStateT::Turn::Attackers) {
      rewardAttacker = env.rewardToFloat(state, mcts.rootNodeReward()) / mcts.rootNodeNumVisits();
    } else {
      rewardDefender = env.rewardToFloat(state, mcts.rootNodeReward()) / mcts.rootNodeNumVisits();
    }

    if ((i > 0 && i % 2 == 0) || !success) {
      // if we are done, print out current state, otherwise print last state and the action we took
      if (!success) {
        lastState = state;
      }

      for (size_t j = 0; j < NumAttackers; ++j) {
        out << lastState.attackers[j].position(0) << "," << lastState.attackers[j].position(1) << ","
            << lastState.attackers[j].velocity(0) << "," << lastState.attackers[j].velocity(1) << ","
            << action[j](0) << "," << action[j](1) << ",";
      }
      for (size_t j = 0; j < NumDefenders; ++j) {
        out << lastState.defenders[j].position(0) << "," << lastState.defenders[j].position(1) << ","
            << lastState.defenders[j].velocity(0) << "," << lastState.defenders[j].velocity(1) << ","
            << lastAction[j+NumAttackers](0) << "," << lastAction[j+NumAttackers](1) << ",";
      }
      out << rewardAttacker << "," << rewardDefender << std::endl;
      lastState = state;
    }

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
}


int main(int argc, char* argv[]) {

  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(),"input file (YAML)")
    ("output,o", po::value<std::string>(&outputFile)->required(),"output file (YAML)");

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error& e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  YAML::Node config = YAML::LoadFile(inputFile);

  int numAttackers = config["num_nodes_A"].as<int>();
  int numDefenders = config["num_nodes_B"].as<int>();

  if (numAttackers == 1 && numDefenders == 1) {
    runMCTS<1,1>(config, outputFile);
  }
  else if (numAttackers == 2 && numDefenders == 1) {
    runMCTS<2,1>(config, outputFile);
  }
  else if (numAttackers == 1 && numDefenders == 2) {
    runMCTS<1,2>(config, outputFile);
  }
  else if (numAttackers == 2 && numDefenders == 2) {
    runMCTS<2,2>(config, outputFile);
  } else {
    std::cerr << "Need to recompile for " << numAttackers << "," << numDefenders << std::endl;
    return 1;
  }

  return 0;
}
