#include <sstream>
#include <random>
#include <fstream>

#include <Eigen/Geometry>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "robots/Dubins2D.hpp"

// #include "robots/RobotState.hpp"
#include "GameState.hpp"
// #include "robots/RobotType.hpp"

#include <yaml-cpp/yaml.h>
#include "GLAS.hpp"

#include "Game.hpp"
#include "monte_carlo_tree_search.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

typedef Dubins2D RobotT;
typedef RobotT::State RobotStateT;
typedef RobotT::Type RobotTypeT;
typedef GameState<RobotT> GameStateT;
typedef Game<RobotT> GameT;
typedef DeepSetNN<RobotT::StateDim> DeepSetNNT;
// typedef DiscreteEmptyNet<RobotT::StateDim> DiscreteEmptyNetT;
typedef GLAS<RobotT> GLAST;

// global variables
std::random_device g_r;
std::default_random_engine g_generator(g_r());

template<class T>
std::string toString(const T& x) 
{
  std::stringstream sstr;
  sstr << x;
  return sstr.str();
}

void seed(size_t seed)
{
  g_generator = std::default_random_engine(seed);
}

class MCTSResult
{
public:
  bool success;
  GameT::GameActionT bestAction;
  Reward expectedReward;
  std::vector<std::pair<GameT::GameActionT, float>> valuePerAction;
};

MCTSResult search(
  GameT& game,
  const GameT::GameStateT& startState,
  size_t num_nodes,
  float rollout_beta,
  float Cp,
  float pw_C,
  float pw_alpha,
  float vf_beta,
  const char* export_dot = nullptr)
{
  game.setRolloutBeta(rollout_beta);
  MCTSResult result;
  libMultiRobotPlanning::MonteCarloTreeSearch<GameT::GameStateT, GameT::GameActionT, Reward, GameT> mcts(
    game, g_generator, num_nodes, Cp, pw_C, pw_alpha, vf_beta);
  result.success = mcts.search(startState, result.bestAction);
  if (result.success) {
    result.expectedReward = mcts.rootNodeReward() / mcts.rootNodeNumVisits();
    result.valuePerAction = mcts.valuePerAction();
  }
  if (export_dot) {
    std::ofstream stream(export_dot);
    mcts.exportToDot(stream);
  }
  return result;
}

GameT::GameActionT eval(
  GameT& game,
  const GameT::GameStateT& startState,
  bool deterministic)
{
  return computeActionsWithGLAS(
    game.glasA(), game.glasB(),
    startState, game.goal(),
    game.attackerTypes(), game.defenderTypes(),
	deterministic);
}


PYBIND11_MODULE(mctscppdubins2D, m) {


  // helper functions
  m.def("seed", &seed);
  m.def("search", &search,
    "game"_a,
    "start_state"_a,
    "num_nodes"_a,
    "rollout_beta"_a,
    "Cp"_a,
    "pw_C"_a,
    "pw_alpha"_a,
    "vf_beta"_a,
    "export_dot"_a = nullptr);
  m.def("eval", &eval);

  // helper classes
  py::class_<std::default_random_engine> (m, "default_random_engine");

  py::class_<MCTSResult> (m, "MCTSResult")
    .def_readonly("success", &MCTSResult::success)
    .def_readonly("bestAction", &MCTSResult::bestAction)
    .def_readonly("expectedReward", &MCTSResult::expectedReward)
    .def_readonly("valuePerAction", &MCTSResult::valuePerAction);

  // RobotState
  py::class_<RobotT::State> robotState(m, "RobotState");

  py::enum_<RobotState::Status>(robotState, "Status")
    .value("Active", RobotState::Status::Active)
    .value("Captured", RobotState::Status::Captured)
    .value("ReachedGoal", RobotState::Status::ReachedGoal)
    .value("Invalid", RobotState::Status::Invalid);

  robotState.def(py::init())
    .def(py::init<const Eigen::Vector4f&>())                          // This might need to change
    // .def_property_readonly("position", &RobotStateT::position)
    // .def_property_readonly("velocity", &RobotStateT::velocity)
    .def_readwrite("state", &RobotStateT::state)
    .def_readwrite("status", &RobotStateT::status)
    .def("__repr__", &toString<RobotStateT>);

  // GameState
  py::class_<GameStateT> gameState(m, "GameState");

  py::enum_<GameStateT::Turn>(gameState, "Turn")
    .value("Attackers", GameStateT::Turn::Attackers)
    .value("Defenders", GameStateT::Turn::Defenders);

  gameState.def(py::init())
    .def(py::init<
      GameStateT::Turn&,
      const std::vector<RobotStateT>&,
      const std::vector<RobotStateT>&>())
    .def_readwrite("turn", &GameStateT::turn)
    .def_readwrite("attackers", &GameStateT::attackers)
    .def_readwrite("defenders", &GameStateT::defenders)
    .def_readwrite("depth", &GameStateT::depth)
    .def("__repr__", &toString<GameStateT>);

  // RobotType
  py::class_<RobotTypeT> (m, "RobotType")
    .def(py::init<
      const Eigen::Vector2f&,
      const Eigen::Vector2f&,
      float, float, float, float, float>())                            // Changes between SI (4f) and DI (5f)
    .def_readwrite("p_min", &RobotTypeT::p_min)
    .def_readwrite("p_max", &RobotTypeT::p_max)
    .def_readwrite("velocity_limit", &RobotTypeT::velocity_limit)
    .def_readonly("acceleration_limit", &RobotTypeT::acceleration_limit) // Commented in SI
    .def_readwrite("tag_radiusSquared", &RobotTypeT::tag_radiusSquared)
    .def_readwrite("r_senseSquared", &RobotTypeT::r_senseSquared)
    .def_readwrite("radius", &RobotTypeT::radius)
    .def("__repr__", &toString<RobotTypeT>);

  // FeedForwardNN
  py::class_<FeedForwardNN> (m, "FeedForwardNN")
    .def("addLayer", &FeedForwardNN::addLayer)
    .def("eval", &FeedForwardNN::eval)
    .def_property_readonly("sizeIn", &FeedForwardNN::sizeIn)
    .def_property_readonly("sizeOut", &FeedForwardNN::sizeOut);

  // DeepSetNN
  py::class_<DeepSetNNT> (m, "DeepSetNN")
    .def("eval", &DeepSetNNT::eval)
    .def_property_readonly("sizeOut", &DeepSetNNT::sizeOut)
    .def_property_readonly("phi", &DeepSetNNT::phi)
    .def_property_readonly("rho", &DeepSetNNT::rho);

  // // DiscreteEmptyNet
  // py::class_<DiscreteEmptyNetT> (m, "DiscreteEmptyNet")
  //   .def("eval", &DiscreteEmptyNetT::eval)
  //   .def_property_readonly("deepSetA", &DiscreteEmptyNetT::deepSetA)
  //   .def_property_readonly("deepSetB", &DiscreteEmptyNetT::deepSetB)
  //   .def_property_readonly("psi", &DiscreteEmptyNetT::psi);


  // GLAS
  py::class_<GLAST> (m, "GLAS")
    // .def(py::init<std::default_random_engine&>(), "generator"_a = g_generator)
    // .def("computeAction", &GLAST::computeAction)
    .def_property_readonly("deepSetA", &GLAST::deepSetA)
    .def_property_readonly("deepSetB", &GLAST::deepSetB)
    .def_property_readonly("psi", &GLAST::psi)
    .def_property_readonly("encoder", &GLAST::encoder)
    .def_property_readonly("decoder", &GLAST::decoder);

  // Game
  py::class_<GameT> (m, "Game")
    .def(py::init<
      const std::vector<RobotTypeT>&,
      const std::vector<RobotTypeT>&,
      float,
      const Eigen::Vector4f&,                        // This might need to change
      size_t,
      std::default_random_engine&>(),
      "attackerTypes"_a,
      "defenderTypes"_a,
      "dt"_a,
      "goal"_a,
      "maxDepth"_a,
      "generator"_a = g_generator)
    .def("step", &GameT::step)
    .def("isTerminal", &GameT::isTerminal)
    .def("isValid", &GameT::isValid)
    .def("computeReward", &GameT::computeReward)
    .def_property_readonly("glasA", &GameT::glasA)
    .def_property_readonly("glasB", &GameT::glasB)
    .def_property_readonly("attackerTypes", &GameT::attackerTypes)
    .def_property_readonly("defenderTypes", &GameT::defenderTypes);
}