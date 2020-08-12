#pragma once
#include <vector>

#include <Eigen/Dense>

#include "RobotState.hpp"

typedef Eigen::Vector2f RobotAction;

class RobotType
{
public:
  Eigen::Vector2f p_min;
  Eigen::Vector2f p_max;
  float velocity_limit;
  float acceleration_limit;
  float tag_radiusSquared;
  float r_senseSquared;
  std::vector<RobotAction> possibleActions;
  RobotAction invalidAction;

  void step(const RobotState& state, const RobotAction& action, float dt, RobotState& result) const
  {
    result.position = state.position + state.velocity * dt;
    Eigen::Vector2f velocity = state.velocity + action * dt;

    float alpha = velocity.norm() / velocity_limit;
    result.velocity = velocity / std::max(alpha, 1.0f);
    // result.velocity = clip(velocity, -velocity_limit, velocity_limit);
    // result.velocity = velocity.cwiseMin(velocity_limit).cwiseMax(-velocity_limit);
  }

  bool isStateValid(const RobotState& state) const
  {
    return (state.position.array() >= p_min.array()).all() && (state.position.array() <= p_max.array()).all();
  }

  void init()
  {
    possibleActions.resize(9);
    possibleActions[0] << -acceleration_limit, -acceleration_limit;
    possibleActions[1] << -acceleration_limit, 0;
    possibleActions[2] << -acceleration_limit, acceleration_limit;
    possibleActions[3] << 0, -acceleration_limit;
    possibleActions[4] << 0, 0;
    possibleActions[5] << 0, acceleration_limit;
    possibleActions[6] << acceleration_limit, -acceleration_limit;
    possibleActions[7] << acceleration_limit, 0;
    possibleActions[8] << acceleration_limit, acceleration_limit;

    // invalidAction << nanf("") , nanf("");
  }
};