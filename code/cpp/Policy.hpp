#pragma once

#include <random>

#include "GLAS.hpp"

template<class Robot>
class Policy
{
public:
  typedef typename Robot::Type RobotTypeT;
  typedef typename Robot::Action RobotActionT;
  typedef typename Robot::State RobotStateT;

  typedef GameState<Robot> GameStateT;
  // typedef std::vector<RobotActionT> GameActionT;

  Policy(
    const std::string& name,
    std::default_random_engine& generator)
    : m_name(name)
    , m_generator(generator)
    , m_rollout_beta(0.0)
    , m_weight(1.0)
    , m_glas(generator)
  {
  }

  RobotActionT sampleAction(
    const RobotStateT& state,
    const RobotTypeT& robotType,
    bool teamAttacker,
    size_t idx,
    const GameStateT& gameState,
    const Eigen::Matrix<float, Robot::StateDim, 1>& goal,
    bool deterministic,
    float rollout_beta = -1) const
  {
    if (state.status != RobotState::Status::Active) {
      return robotType.invalidAction;
    }
    if (rollout_beta == -1) {
      rollout_beta = m_rollout_beta;
    }

    std::uniform_real_distribution<float> dist(0.0,1.0);

    if (rollout_beta >= 1 || (rollout_beta > 0 && dist(m_generator) < rollout_beta)) {
      // Use NN if rollout_beta is > 0 probabilistically
      assert(m_glas.valid());
      auto result = m_glas.eval(gameState, goal, robotType, teamAttacker, idx, deterministic);
      return std::get<1>(result);
    } else {
      // use uniform random sample (no deterministic option)
      std::uniform_real_distribution<float> distTheta(0.0, 2*M_PI);
      std::uniform_real_distribution<float> distMag(0.0, 1.0);
      float theta = distTheta(m_generator);
      float mag = sqrtf(distMag(m_generator)) * robotType.actionLimit();
      return RobotActionT(cosf(theta) * mag, sinf(theta) * mag);
    }
  }

  const std::string& name() const {
    return m_name;
  }

  void setName(const std::string& name) {
    m_name = name;
  }

  float rolloutBeta() const {
    return m_rollout_beta;
  }

  void setRolloutBeta(float rollout_beta) {
    m_rollout_beta = rollout_beta;
  }

  float weight() const {
    return m_weight;
  }

  void setWeight(float weight) {
    m_weight = weight;
  }

  GLAS<Robot>& glas()
  {
    return m_glas;
  }

  const GLAS<Robot>& glasConst() const
  {
    return m_glas;
  }

  bool valid() const
  {
    return m_glas.valid() || m_rollout_beta <= 0;
  }

  friend std::ostream& operator<<(std::ostream& out, const Policy& p)
  {
    out <<"Policy(" << p.m_name;
    out <<",rolloutBeta=" << p.m_rollout_beta;
    out <<",weight=" << p.m_weight;
    out << ")";
    return out;
  }

private:
  std::string m_name;
  std::default_random_engine& m_generator;
  float m_rollout_beta;
  float m_weight;
  GLAS<Robot> m_glas;
};
