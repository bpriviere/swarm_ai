#include <iostream>
#include <array>
#include <Eigen/Dense>
#include <fstream>
#include <bitset>
#include <unordered_map>
#include <random>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

#include "Game.hpp"
#include "GLAS.hpp"

template <std::size_t NumAttackers, std::size_t NumDefenders>
void runGame(
  const YAML::Node& config,
  const YAML::Node& cfg_nn,
  const std::string& outputFile)
{
  using EnvironmentT = Game<NumAttackers, NumDefenders>;
  using GameStateT = typename EnvironmentT::GameStateT;
  using GameActionT = typename EnvironmentT::GameActionT;

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
  state.depth = 0;
  state.attackersReward = 0;
  state.defendersReward = 0;
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

  EnvironmentT env(attackerTypes, defenderTypes, dt, goal, 1e6, generator);

  // load GLAS
  GLAS glas_a(cfg_nn["team_a"], generator);
  GLAS glas_b(cfg_nn["team_b"], generator);

  // std::vector<Eigen::Vector4f> input_a(1);
  // input_a[0] << 0,0,0,0;

  // std::vector<Eigen::Vector4f> input_b;

  // Eigen::Vector4f goal;
  // goal << 0,0,0,0;

  // auto action = glas_b.computeAction(input_a, input_b, goal, /*deterministic*/false);
  // std::cout << action << std::endl;

  //


  std::ofstream out(outputFile);
  // write file header
  for (size_t j = 0; j < NumAttackers+NumDefenders; ++j) {
    out << "x,y,vx,vy,ax,ay,";
  }
  out << "rewardAttacker,rewardDefender" << std::endl;

  GameActionT action;
  // GameActionT lastAction;
  // GameStateT lastState = state;
  // float rewardAttacker;
  // float rewardDefender;
  std::vector<Eigen::Vector4f> input_a;
  std::vector<Eigen::Vector4f> input_b;
  Eigen::Vector4f relGoal;
  for(int i = 0; ; ++i) {

    action = computeActionsWithGLAS(glas_a, glas_b, state, goal, attackerTypes, defenderTypes, generator);

    // output state & action

    for (size_t j = 0; j < NumAttackers; ++j) {
      out << state.attackers[j].position(0) << "," << state.attackers[j].position(1) << ","
          << state.attackers[j].velocity(0) << "," << state.attackers[j].velocity(1) << ","
          << action[j](0) << "," << action[j](1) << ",";
    }
    for (size_t j = 0; j < NumDefenders; ++j) {
      out << state.defenders[j].position(0) << "," << state.defenders[j].position(1) << ","
          << state.defenders[j].velocity(0) << "," << state.defenders[j].velocity(1) << ","
          << action[j+NumAttackers](0) << "," << action[j+NumAttackers](1) << ",";
    }
    out << state.attackersReward / state.depth << ","
        << state.defendersReward / state.depth << std::endl;

    // step forward (twice: once for each player)

    bool success = env.step(state, action, state);
    std::cout << state << " s: " << success << std::endl;
    success &= env.step(state, action, state);
    std::cout << state << " s: " << success << std::endl;
    if (!success || env.isTerminal(state)) {
      break;
    }

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
    ("input,i", po::value<std::string>(&inputFile)->required(),"input config file (YAML)")
    ("inputNN,n", po::value<std::string>(&inputFileNN)->required(),"input config file NN (YAML)")
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
  YAML::Node cfg_nn = YAML::LoadFile(inputFileNN);

  int numAttackers = config["num_nodes_A"].as<int>();
  int numDefenders = config["num_nodes_B"].as<int>();

  if (numAttackers == 1 && numDefenders == 1) {
    runGame<1,1>(config, cfg_nn, outputFile);
  }
  else if (numAttackers == 2 && numDefenders == 1) {
    runGame<2,1>(config, cfg_nn, outputFile);
  }
  else if (numAttackers == 1 && numDefenders == 2) {
    runGame<1,2>(config, cfg_nn, outputFile);
  }
  else if (numAttackers == 2 && numDefenders == 2) {
    runGame<2,2>(config, cfg_nn, outputFile);
  } else {
    std::cerr << "Need to recompile for " << numAttackers << "," << numDefenders << std::endl;
    return 1;
  }

  return 0;
}
