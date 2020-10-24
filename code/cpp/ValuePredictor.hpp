#pragma once

#include <random>
#include <algorithm>

#include "GLAS.hpp"

template<class Robot>
class ValuePredictor
{
public:
  typedef typename Robot::Type RobotTypeT;
  typedef typename Robot::Action RobotActionT;
  typedef typename Robot::State RobotStateT;
  typedef Eigen::Matrix<float, Robot::StateDim, 1> StateVector;

  typedef GameState<Robot> GameStateT;
  // typedef std::vector<RobotActionT> GameActionT;

  ValuePredictor(
    const std::string& name,
    std::default_random_engine& generator)
    : m_name(name)
    , m_generator(generator)
    , m_ds_a()
    , m_ds_b()
    , m_value()
  {
  }

  float estimate(
    const GameStateT& state,
    const Eigen::Matrix<float, Robot::StateDim, 1>& goal,
    bool deterministic) const
  {
    const size_t NumAttackers = state.attackers.size();
    const size_t NumDefenders = state.defenders.size();

    std::vector<StateVector> input_a;
    std::vector<StateVector> input_b;

    // compute input_a
    int num_reached_goal = 0;
    for (size_t i = 0; i < NumAttackers; ++i) {
      if (state.attackers[i].status == Robot::State::Status::Active) {
        auto relState = state.attackers[i].state - goal;
        input_a.push_back(relState);
      }
      if (state.attackers[i].status == Robot::State::Status::ReachedGoal) {
        ++num_reached_goal;
      }
    }

    // compute input_b
    for (size_t i = 0; i < NumDefenders; ++i) {
      if (state.defenders[i].status == Robot::State::Status::Active) {
        auto relState = state.defenders[i].state - goal;
        input_b.push_back(relState);
      }
    }

    Eigen::VectorXf value_input(m_value.sizeIn());
    value_input.segment(0, m_ds_a.sizeOut()) = m_ds_a.eval(input_a);
    value_input.segment(m_ds_a.sizeOut(), m_ds_b.sizeOut()) = m_ds_b.eval(input_b);
    value_input(m_ds_a.sizeOut()+m_ds_b.sizeOut()+0) = NumAttackers;
    value_input(m_ds_a.sizeOut()+m_ds_b.sizeOut()+1) = NumDefenders;
    value_input(m_ds_a.sizeOut()+m_ds_b.sizeOut()+2) = num_reached_goal;

    auto val = m_value.eval(value_input);

    auto mu = val.segment<1>(0);
    float value;
    if (deterministic) {
      value = mu(0);
    } else {
      auto logvar = val.segment<1>(1);
      auto sd = logvar.array().exp().sqrt();
      std::normal_distribution<float> dist(mu(0),sd(0));
      value = dist(m_generator);
    }
    // value = std::clamp(value, 0, 1);
    value = std::min(std::max(value, 0.0f), 1.0f);
    return value;
  }

  const std::string& name() const {
    return m_name;
  }

  void setName(const std::string& name) {
    m_name = name;
  }

  auto& deepSetA()
  {
    return m_ds_a;
  }

  auto& deepSetB()
  {
    return m_ds_b;
  }

  auto& value()
  {
    return m_value;
  }

  bool valid() const
  {
    return m_value.valid();
  }

  friend std::ostream& operator<<(std::ostream& out, const ValuePredictor& p)
  {
    out <<"ValuePredictor(" << p.m_name;
    out << ")";
    return out;
  }

private:
  std::string m_name;
  std::default_random_engine& m_generator;

  DeepSetNN<Robot::StateDim> m_ds_a;
  DeepSetNN<Robot::StateDim> m_ds_b;
  FeedForwardNN m_value;
};
