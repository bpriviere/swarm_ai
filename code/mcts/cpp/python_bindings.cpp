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

std::tuple<GLAS, GLAS> createGLAS(
  const std::string& inputFileNN,
  std::default_random_engine& generator)
{
  YAML::Node cfg_nn = YAML::LoadFile(inputFileNN);

  GLAS glas_a(cfg_nn["team_a"], generator);
  GLAS glas_b(cfg_nn["team_b"], generator);

  return std::make_tuple<>(glas_a, glas_b);
}

Game::GameActionT search(
  Game& game,
  const GameState& startState,
  std::default_random_engine& generator,
  size_t num_nodes)
{
  libMultiRobotPlanning::MonteCarloTreeSearch<GameState, Game::GameActionT, Reward, Game> mcts(game, generator, num_nodes, 1.4);
  Game::GameActionT result;
  bool success = mcts.search(startState, result);
  if (!success) {
    result.clear();
  }
  return result;
}


PYBIND11_MODULE(mctscpp, m) {

  // helper functions
  m.def("createRandomGenerator", &createRandomGenerator);
  m.def("createGLAS", &createGLAS);
  m.def("search", &search);

  // helper classes
  py::class_<std::default_random_engine> (m, "default_random_engine");

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

  // GLAS
  py::class_<GLAS> (m, "GLAS");

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
      const GLAS*>())
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