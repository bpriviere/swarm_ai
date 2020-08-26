#include <iostream>
#include <array>
#include <Eigen/Dense>
#include <fstream>
#include <bitset>
#include <unordered_map>
#include <random>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

#include "robots/DoubleIntegrator2D.hpp"

#include "Game.hpp"
#include "GLAS.hpp"

typedef DoubleIntegrator2D RobotT;
// typedef RobotT::State RobotStateT;
typedef RobotT::Type RobotTypeT;
// typedef GameState<RobotT> GameStateT;
typedef Game<RobotT> GameT;

void runGame(
  const YAML::Node& config,
  const YAML::Node& cfg_nn,
  const std::string& outputFile)
{
  using EnvironmentT = GameT;
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

  int NumAttackers = config["num_nodes_A"].as<int>();
  int NumDefenders = config["num_nodes_B"].as<int>();

  GameStateT state;
  state.attackers.resize(NumAttackers);
  state.defenders.resize(NumDefenders);

  state.turn = GameStateT::Turn::Attackers;
  state.activeMask = 0;
  state.depth = 0;
  state.attackersReward = 0;
  state.defendersReward = 0;

  std::vector<RobotTypeT> attackerTypes(NumAttackers);
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
    assert(i<32);
    state.activeMask |= (1<<i);
    state.attackers[i].state << node["x0"][0].as<float>(),node["x0"][1].as<float>()
                              , node["x0"][2].as<float>(),node["x0"][3].as<float>();

  }
  std::vector<RobotTypeT> defenderTypes(NumDefenders);
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
    state.defenders[i].state << node["x0"][0].as<float>(),node["x0"][1].as<float>()
                              , node["x0"][2].as<float>(),node["x0"][3].as<float>();
  }
  std::cout << state << std::endl;

  float dt = config["sim_dt"].as<float>();
  Eigen::Vector4f goal;
  goal << config["goal"][0].as<float>(),config["goal"][1].as<float>(), 0, 0;

  // load GLAS
  GLAS<RobotT::StateDim> glas_a(generator);
  glas_a.load(cfg_nn["team_a"]);
  GLAS<RobotT::StateDim> glas_b(generator);
  glas_b.load(cfg_nn["team_b"]);

  EnvironmentT env(attackerTypes, defenderTypes, dt, goal, 1e6, generator, glas_a, glas_b, 0.0);


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

    action = computeActionsWithGLAS(glas_a, glas_b, state, goal, attackerTypes, defenderTypes, generator, true);

    // output state & action

    for (size_t j = 0; j < NumAttackers; ++j) {
      out << state.attackers[j].state(0) << "," << state.attackers[j].state(1) << ","
          << state.attackers[j].state(2) << "," << state.attackers[j].state(3) << ","
          << action[j](0) << "," << action[j](1) << ",";
    }
    for (size_t j = 0; j < NumDefenders; ++j) {
      out << state.defenders[j].state(0) << "," << state.defenders[j].state(1) << ","
          << state.defenders[j].state(2) << "," << state.defenders[j].state(3) << ","
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

  runGame(config, cfg_nn, outputFile);

  return 0;
}
