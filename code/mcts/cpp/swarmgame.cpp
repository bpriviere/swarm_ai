#include <iostream>
#include <array>
#include <Eigen/Dense>
#include <fstream>
#include <bitset>
#include <unordered_map>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

#include "Game.hpp"

#include "monte_carlo_tree_search.hpp"

#include "GLAS.hpp"

template <std::size_t NumAttackers, std::size_t NumDefenders>
void runMCTS(
  const YAML::Node& config,
  const std::string& outputFile,
  std::default_random_engine& generator,
  GLAS* glas_a,
  GLAS* glas_b)
{
  using EnvironmentT = Game<NumAttackers, NumDefenders>;
  using GameStateT = typename EnvironmentT::GameStateT;
  using GameActionT = typename EnvironmentT::GameActionT;

  size_t num_nodes = config["tree_size"].as<int>();

  GameStateT state;

  state.turn = GameStateT::Turn::Attackers;
  state.activeMask.set();
  
  std::array<RobotType, NumAttackers> attackerTypes;
  for (size_t i = 0; i < NumAttackers; ++i) {
    const auto& node = config["robots"][i];
    attackerTypes[i].p_min << config["env_xlim"][0].as<float>(), config["env_ylim"][0].as<float>();
    attackerTypes[i].p_max << config["env_xlim"][1].as<float>(), config["env_ylim"][1].as<float>();
    attackerTypes[i].velocity_limit = node["speed_limit"].as<float>(); // / sqrtf(2.0);
    attackerTypes[i].acceleration_limit = node["acceleration_limit"].as<float>() / sqrtf(2.0);
    attackerTypes[i].tag_radiusSquared = powf(node["tag_radius"].as<float>(), 2);
    attackerTypes[i].r_senseSquared = powf(node["r_sense"].as<float>(), 2);
    attackerTypes[i].init();

    state.attackers[i].status = RobotState::Status::Active;
    state.attackers[i].position << node["x0"][0].as<float>(),node["x0"][1].as<float>();
    state.attackers[i].velocity << node["x0"][2].as<float>(),node["x0"][3].as<float>();

  }
  std::array<RobotType, NumDefenders> defenderTypes;
  for (size_t i = 0; i < NumDefenders; ++i) {
    const auto& node = config["robots"][i+NumAttackers];
    defenderTypes[i].p_min << config["env_xlim"][0].as<float>(), config["env_ylim"][0].as<float>();
    defenderTypes[i].p_max << config["env_xlim"][1].as<float>(), config["env_ylim"][1].as<float>();
    defenderTypes[i].velocity_limit = node["speed_limit"].as<float>() / sqrtf(2.0);
    defenderTypes[i].acceleration_limit = node["acceleration_limit"].as<float>() / sqrtf(2.0);
    defenderTypes[i].tag_radiusSquared = powf(node["tag_radius"].as<float>(), 2);
    defenderTypes[i].r_senseSquared = powf(node["r_sense"].as<float>(), 2);
    defenderTypes[i].init();

    state.defenders[i].status = RobotState::Status::Active;
    state.defenders[i].position << node["x0"][0].as<float>(),node["x0"][1].as<float>();
    state.defenders[i].velocity << node["x0"][2].as<float>(),node["x0"][3].as<float>();
  }
  std::cout << state << std::endl;

  float dt = config["sim_dt"].as<float>();
  Eigen::Vector2f goal;
  goal << config["goal"][0].as<float>(),config["goal"][1].as<float>();

  size_t max_depth = config["rollout_horizon"].as<int>();

  EnvironmentT env(attackerTypes, defenderTypes, dt, goal, max_depth, generator, glas_a, glas_b);

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
  std::string inputFileNN;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(),"input file (YAML)")
    ("inputNN,n", po::value<std::string>(&inputFileNN),"input config file NN (YAML)")
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

  size_t seed;
  if (config["seed"]) {
    seed = config["seed"].as<size_t>();
  } else {
    std::random_device r;
    seed = r();
  }
  std::cout << "Using seed " << seed << std::endl;
  std::default_random_engine generator(seed);

  GLAS* glas_a = nullptr;
  GLAS* glas_b = nullptr;
  if (!inputFileNN.empty()) {
    YAML::Node cfg_nn = YAML::LoadFile(inputFileNN);

    glas_a = new GLAS(cfg_nn["team_a"], generator);
    glas_b = new GLAS(cfg_nn["team_b"], generator);
  }

  int numAttackers = config["num_nodes_A"].as<int>();
  int numDefenders = config["num_nodes_B"].as<int>();

  if (numAttackers == 1 && numDefenders == 1) {
    runMCTS<1,1>(config, outputFile, generator, glas_a, glas_b);
  }
  else if (numAttackers == 2 && numDefenders == 1) {
    runMCTS<2,1>(config, outputFile, generator, glas_a, glas_b);
  }
  else if (numAttackers == 1 && numDefenders == 2) {
    runMCTS<1,2>(config, outputFile, generator, glas_a, glas_b);
  }
  else if (numAttackers == 2 && numDefenders == 2) {
    runMCTS<2,2>(config, outputFile, generator, glas_a, glas_b);
  } else {
    std::cerr << "Need to recompile for " << numAttackers << "," << numDefenders << std::endl;
    return 1;
  }

  return 0;
}
