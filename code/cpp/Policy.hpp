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
    std::default_random_engine& generator)
    : m_generator(generator)
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
    bool deterministic) const
  {
    if (state.status != RobotState::Status::Active) {
      return robotType.invalidAction;
    }

    std::uniform_real_distribution<float> dist(0.0,1.0);

    if (m_rollout_beta > 0 && dist(m_generator) < m_rollout_beta) {
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

private:
  std::default_random_engine& m_generator;
  float m_rollout_beta;
  float m_weight;
  GLAS<Robot> m_glas;
};
