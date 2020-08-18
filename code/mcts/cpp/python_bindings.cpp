#include <sstream>
#include <random>

#include <Eigen/Geometry>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "RobotState.hpp"
#include "GameState.hpp"
#include "RobotType.hpp"

#include <yaml-cpp/yaml.h>
#include "GLAS.hpp"

#include "Game.hpp"
#include "monte_carlo_tree_search.hpp"

namespace py = pybind11;

template<class T>
std::string toString(const T& x) 
{
  std::stringstream sstr;
  sstr << x;
  return sstr.str();
}

std::default_random_engine createRandomGenerator(size_t seed)
{
  return std::default_random_engine(seed);
}

class MCTSResult
{
public:
  bool success;
  Game::GameActionT bestAction;
  Reward expectedReward;
  std::vector<std::pair<Game::GameActionT, float>> valuePerAction;
};

MCTSResult search(
  Game& game,
  const GameState& startState,
  std::default_random_engine& generator,
  size_t num_nodes)
{
  MCTSResult result;
  libMultiRobotPlanning::MonteCarloTreeSearch<GameState, Game::GameActionT, Reward, Game> mcts(game, generator, num_nodes, 1.4);
  result.success = mcts.search(startState, result.bestAction);
  if (result.success) {
    result.expectedReward = mcts.rootNodeReward() / mcts.rootNodeNumVisits();
    result.valuePerAction = mcts.valuePerAction();
  }
  return result;
}


PYBIND11_MODULE(mctscpp, m) {


  // helper functions
  m.def("createRandomGenerator", &createRandomGenerator);
  m.def("search", &search);
  m.def("computeActionsWithGLAS", &computeActionsWithGLAS);

  // helper classes
  py::class_<std::default_random_engine> (m, "default_random_engine");

  py::class_<MCTSResult> (m, "MCTSResult")
    .def_readonly("success", &MCTSResult::success)
    .def_readonly("bestAction", &MCTSResult::bestAction)
    .def_readonly("expectedReward", &MCTSResult::expectedReward)
    .def_readonly("valuePerAction", &MCTSResult::valuePerAction);

  // RobotState
  py::class_<RobotState> robotState(m, "RobotState");

  py::enum_<RobotState::Status>(robotState, "Status")
    .value("Active", RobotState::Status::Active)
    .value("Captured", RobotState::Status::Captured)
    .value("ReachedGoal", RobotState::Status::ReachedGoal);

  robotState.def(py::init())
    .def(py::init<const Eigen::Vector2f&, const Eigen::Vector2f&>())
    .def_readwrite("position", &RobotState::position)
    .def_readwrite("velocity", &RobotState::velocity)
    .def_readwrite("status", &RobotState::status)
    .def("__repr__", &toString<RobotState>);

  // GameState
  py::class_<GameState> gameState(m, "GameState");

  py::enum_<GameState::Turn>(gameState, "Turn")
    .value("Attackers", GameState::Turn::Attackers)
    .value("Defenders", GameState::Turn::Defenders);

  gameState.def(py::init())
    .def(py::init<
      GameState::Turn&,
      const std::vector<RobotState>&,
      const std::vector<RobotState>&>())
    .def_readwrite("turn", &GameState::turn)
    .def_readwrite("attackers", &GameState::attackers)
    .def_readwrite("attackersReward", &GameState::attackersReward)
    .def_readwrite("defenders", &GameState::defenders)
    .def_readwrite("defendersReward", &GameState::defendersReward)
    .def_readwrite("depth", &GameState::depth)
    .def("__repr__", &toString<GameState>);

  // RobotType
  py::class_<RobotType> (m, "RobotType")
    .def(py::init<
      const Eigen::Vector2f&,
      const Eigen::Vector2f&,
      float, float, float, float>())
    .def_readwrite("p_min", &RobotType::p_min)
    .def_readwrite("p_max", &RobotType::p_max)
    .def_readwrite("velocity_limit", &RobotType::velocity_limit)
    .def_readonly("acceleration_limit", &RobotType::acceleration_limit)
    .def_readwrite("tag_radiusSquared", &RobotType::tag_radiusSquared)
    .def_readwrite("r_senseSquared", &RobotType::r_senseSquared);

  // FeedForwardNN
  py::class_<FeedForwardNN> (m, "FeedForwardNN")
    .def("addLayer", &FeedForwardNN::addLayer)
    .def("eval", &FeedForwardNN::eval)
    .def_property_readonly("sizeIn", &FeedForwardNN::sizeIn)
    .def_property_readonly("sizeOut", &FeedForwardNN::sizeOut);

  // DeepSetNN
  py::class_<DeepSetNN> (m, "DeepSetNN")
    .def("eval", &DeepSetNN::eval)
    .def_property_readonly("sizeOut", &DeepSetNN::sizeOut)
    .def_property_readonly("phi", &DeepSetNN::phi)
    .def_property_readonly("rho", &DeepSetNN::rho);

  // DiscreteEmptyNet
  py::class_<DiscreteEmptyNet> (m, "DiscreteEmptyNet")
    .def("eval", &DiscreteEmptyNet::eval)
    .def_property_readonly("deepSetA", &DiscreteEmptyNet::deepSetA)
    .def_property_readonly("deepSetB", &DiscreteEmptyNet::deepSetB)
    .def_property_readonly("psi", &DiscreteEmptyNet::psi);


  // GLAS
  py::class_<GLAS> (m, "GLAS")
    .def(py::init<std::default_random_engine&>())
    .def("computeAction", &GLAS::computeAction)
    .def_property_readonly("discreteEmptyNet", &GLAS::discreteEmptyNet);

  // Game
  py::class_<Game> (m, "Game")
    .def(py::init<
      const std::vector<RobotType>&,
      const std::vector<RobotType>&,
      float,
      const Eigen::Vector2f&,
      size_t,
      std::default_random_engine&,
      const GLAS*,
      const GLAS*,
      float>())
    .def("step", &Game::step)
    .def("isTerminal", &Game::isTerminal);
}

// PYBIND11_MODULE(mctscpp, m) {
//     py::class_<QuadrotorCppEnv>(m, "QuadrotorCppEnv")
//         .def(py::init())
//         .def("step", &QuadrotorCppEnv::step)
//         .def("get_state", &QuadrotorCppEnv::get_state, py::return_value_policy::reference_internal)
//         .def("set_state", &QuadrotorCppEnv::set_state);
// }