#include <sstream>
#include <random>

#include <Eigen/Geometry>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "robots/DoubleIntegrator2D.hpp"

// #include "robots/RobotState.hpp"
#include "GameState.hpp"
// #include "robots/RobotType.hpp"

#include <yaml-cpp/yaml.h>
#include "GLAS.hpp"

#include "Game.hpp"
#include "monte_carlo_tree_search.hpp"

namespace py = pybind11;

typedef DoubleIntegrator2D RobotT;
typedef RobotT::State RobotStateT;
typedef RobotT::Type RobotTypeT;
typedef GameState<RobotT> GameStateT;
typedef Game<RobotT> GameT;

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
  GameT::GameActionT bestAction;
  Reward expectedReward;
  std::vector<std::pair<GameT::GameActionT, float>> valuePerAction;
};

MCTSResult search(
  GameT& game,
  const GameT::GameStateT& startState,
  std::default_random_engine& generator,
  size_t num_nodes)
{
  MCTSResult result;
  libMultiRobotPlanning::MonteCarloTreeSearch<GameT::GameStateT, GameT::GameActionT, Reward, GameT> mcts(game, generator, num_nodes, 1.4);
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
  m.def("computeActionsWithGLAS", &computeActionsWithGLAS<RobotT>);

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
    .value("ReachedGoal", RobotState::Status::ReachedGoal);

  robotState.def(py::init())
    .def(py::init<const Eigen::Vector4f&>())
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
    .def_readwrite("attackersReward", &GameStateT::attackersReward)
    .def_readwrite("defenders", &GameStateT::defenders)
    .def_readwrite("defendersReward", &GameStateT::defendersReward)
    .def_readwrite("depth", &GameStateT::depth)
    .def("__repr__", &toString<GameStateT>);

  // RobotType
  py::class_<RobotTypeT> (m, "RobotType")
    .def(py::init<
      const Eigen::Vector2f&,
      const Eigen::Vector2f&,
      float, float, float, float>())
    .def_readwrite("p_min", &RobotTypeT::p_min)
    .def_readwrite("p_max", &RobotTypeT::p_max)
    .def_readwrite("velocity_limit", &RobotTypeT::velocity_limit)
    .def_readonly("acceleration_limit", &RobotTypeT::acceleration_limit)
    .def_readwrite("tag_radiusSquared", &RobotTypeT::tag_radiusSquared)
    .def_readwrite("r_senseSquared", &RobotTypeT::r_senseSquared)
    .def("__repr__", &toString<RobotTypeT>);

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
  py::class_<GameT> (m, "Game")
    .def(py::init<
      const std::vector<RobotTypeT>&,
      const std::vector<RobotTypeT>&,
      float,
      const Eigen::Vector4f&,
      size_t,
      std::default_random_engine&,
      const GLAS&,
      const GLAS&,
      float>())
    .def("step", &GameT::step)
    .def("isTerminal", &GameT::isTerminal);
}

// PYBIND11_MODULE(mctscpp, m) {
//     py::class_<QuadrotorCppEnv>(m, "QuadrotorCppEnv")
//         .def(py::init())
//         .def("step", &QuadrotorCppEnv::step)
//         .def("get_state", &QuadrotorCppEnv::get_state, py::return_value_policy::reference_internal)
//         .def("set_state", &QuadrotorCppEnv::set_state);
// }