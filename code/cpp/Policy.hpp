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
    , m_beta2(0.0)
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
    float beta2 = -1) const
  {
    if (state.status != RobotState::Status::Active) {
      return robotType.invalidAction;
    }
    if (beta2 == -1) {
      beta2 = m_beta2;
    }

    std::uniform_real_distribution<float> dist(0.0,1.0);

    if (m_glas.valid() && (beta2 >= 1 || (beta2 > 0 && dist(m_generator) < beta2))) {
      // Use NN if beta2 is > 0 probabilistically
      auto result = m_glas.eval(gameState, goal, robotType, teamAttacker, idx, deterministic);
      return result;
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

  float beta2() const {
    return m_beta2;
  }

  void setBeta2(float beta2) {
    m_beta2 = beta2;
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
    return m_glas.valid() || m_beta2 <= 0;
  }

  friend std::ostream& operator<<(std::ostream& out, const Policy& p)
  {
    out <<"Policy(" << p.m_name;
    out <<",beta2=" << p.m_beta2;
    out <<",weight=" << p.m_weight;
    out << ")";
    return out;
  }

private:
  std::string m_name;
  std::default_random_engine& m_generator;
  float m_beta2;
  float m_weight;
  GLAS<Robot> m_glas;
};
