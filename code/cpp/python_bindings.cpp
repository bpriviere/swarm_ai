#include <sstream>
#include <random>
#include <fstream>

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
#include "Policy.hpp"

#include "Game.hpp"
#include "monte_carlo_tree_search.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

typedef DoubleIntegrator2D RobotT;
typedef RobotT::State RobotStateT;
typedef RobotT::Type RobotTypeT;
typedef GameState<RobotT> GameStateT;
typedef Game<RobotT> GameT;
typedef DeepSetNN<RobotT::StateDim> DeepSetNNT;
// typedef DiscreteEmptyNet<RobotT::StateDim> DiscreteEmptyNetT;
typedef GLAS<RobotT> GLAST;
typedef Policy<RobotT> PolicyT;
typedef ValuePredictor<RobotT> ValuePredictorT;

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
  Eigen::MatrixXf tree;
};

class MCTSSettings
{
public:
  MCTSSettings()
    : num_nodes(10000)
    , Cp(1.4)
    , pw_C(1.0)
    , pw_alpha(0.25)
    , beta1(0)
    , beta3(0)
    , export_dot(nullptr)
    , export_tree(false)
  {
  }

  size_t num_nodes;
  float Cp;
  float pw_C;
  float pw_alpha;
  float beta1; // gain for best-child
  float beta3; // gain for rollout
  const char* export_dot;
  bool export_tree;
};

MCTSResult search(
  GameT& game,
  const GameT::GameStateT& startState,
  const PolicyT& myPolicy,
  const std::vector<PolicyT>& opponentPolicies,
  const ValuePredictorT& valuePredictor,
  const MCTSSettings& settings)
{
  MCTSResult result;
  libMultiRobotPlanning::MonteCarloTreeSearch<GameT::GameStateT, GameT::GameActionT, Reward, GameT, PolicyT, ValuePredictorT> mcts(
    game, g_generator,
    settings.num_nodes,
    settings.Cp,
    settings.pw_C,
    settings.pw_alpha,
    settings.beta1,
    settings.beta3);
  result.success = mcts.search(startState, myPolicy, opponentPolicies, valuePredictor, result.bestAction);
  if (result.success) {
    result.expectedReward = mcts.rootNodeReward() / mcts.rootNodeNumVisits();
    result.valuePerAction = mcts.valuePerAction();
    if (settings.export_tree) {
      result.tree = mcts.exportToMatrix();
    }
  }
  if (settings.export_dot) {
    std::ofstream stream(settings.export_dot);
    mcts.exportToDot(stream);
  }
  return result;
}

GameT::GameActionT eval(
  GameT& game,
  const GameT::GameStateT& startState,
  const PolicyT& policyAttacker,
  const PolicyT& policyDefender,
  bool deterministic)
{
  auto result = game.sampleAction(startState, policyAttacker, policyDefender, deterministic);
  return result;
}


PYBIND11_MODULE(mctscpp, m) {


  // helper functions
  m.def("seed", &seed);
  m.def("search", &search,
    "game"_a,
    "start_state"_a,
    "my_policy"_a,
    "opponent_policies"_a,
    "value_predictor"_a,
    "settings"_a);
  m.def("eval", &eval);

  // helper classes
  py::class_<std::default_random_engine> (m, "default_random_engine");

  py::class_<MCTSResult> (m, "MCTSResult")
    .def_readonly("success", &MCTSResult::success)
    .def_readonly("bestAction", &MCTSResult::bestAction)
    .def_readonly("expectedReward", &MCTSResult::expectedReward)
    .def_readonly("valuePerAction", &MCTSResult::valuePerAction)
    .def_readonly("tree", &MCTSResult::tree);

  py::class_<MCTSSettings> (m, "MCTSSettings")
    .def(py::init())
    .def_readwrite("num_nodes", &MCTSSettings::num_nodes)
    .def_readwrite("Cp", &MCTSSettings::Cp)
    .def_readwrite("pw_C", &MCTSSettings::pw_C)
    .def_readwrite("pw_alpha", &MCTSSettings::pw_alpha)
    .def_readwrite("beta1", &MCTSSettings::beta1)
    .def_readwrite("beta3", &MCTSSettings::beta3)
    .def_readwrite("export_dot", &MCTSSettings::export_dot)
    .def_readwrite("export_tree", &MCTSSettings::export_tree);

  // RobotState
  py::class_<RobotT::State> robotState(m, "RobotState");

  py::enum_<RobotState::Status>(robotState, "Status")
    .value("Active", RobotState::Status::Active)
    .value("Captured", RobotState::Status::Captured)
    .value("ReachedGoal", RobotState::Status::ReachedGoal)
    .value("Invalid", RobotState::Status::Invalid);

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
    .def_readwrite("defenders", &GameStateT::defenders)
    .def_readwrite("depth", &GameStateT::depth)
    .def("__repr__", &toString<GameStateT>);

  // RobotType
  py::class_<RobotTypeT> (m, "RobotType")
    .def(py::init<
      const Eigen::Vector2f&,
      const Eigen::Vector2f&,
      float, float, float, float, float>())
    .def_readwrite("p_min", &RobotTypeT::p_min)
    .def_readwrite("p_max", &RobotTypeT::p_max)
    .def_readwrite("velocity_limit", &RobotTypeT::velocity_limit)
    .def_readonly("acceleration_limit", &RobotTypeT::acceleration_limit)
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
    // .def_property_readonly("psi", &GLAST::psi)
    // .def_property_readonly("encoder", &GLAST::encoder)
    // .def_property_readonly("decoder", &GLAST::decoder)
    // .def_property_readonly("value", &GLAST::value)
    .def_property_readonly("policy", &GLAST::policy);

  // Policy
  py::class_<PolicyT> (m, "Policy")
    .def(py::init<
      const std::string&,
      std::default_random_engine&>(),
      "name"_a,
      "generator"_a = g_generator)
    .def("__repr__", &toString<PolicyT>)
    .def_property_readonly("glas", &PolicyT::glas)
    .def_property("name", &PolicyT::name, &PolicyT::setName)
    .def_property("weight", &PolicyT::weight, &PolicyT::setWeight)
    .def_property("beta2", &PolicyT::beta2, &PolicyT::setBeta2);

  // ValuePredictor
  py::class_<ValuePredictorT> (m, "ValuePredictor")
    .def(py::init<
      const std::string&>(),
      "name"_a)
    .def("__repr__", &toString<ValuePredictorT>)
    .def("estimate", &ValuePredictorT::estimate)
    .def_property("name", &ValuePredictorT::name, &ValuePredictorT::setName)
    .def_property_readonly("deepSetA", &ValuePredictorT::deepSetA)
    .def_property_readonly("deepSetB", &ValuePredictorT::deepSetB)
    .def_property_readonly("value", &ValuePredictorT::value);

  // Game
  py::class_<GameT> (m, "Game")
    .def(py::init<
      const std::vector<RobotTypeT>&,
      const std::vector<RobotTypeT>&,
      float,
      const Eigen::Vector4f&,
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
    .def_property_readonly("attackerTypes", &GameT::attackerTypes)
    .def_property_readonly("defenderTypes", &GameT::defenderTypes);
}
