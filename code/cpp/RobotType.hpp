#pragma once
#include <vector>

#include <Eigen/Dense>

#include "RobotState.hpp"

typedef Eigen::Vector2f RobotAction;

class RobotType
{
public:

  RobotType() = default;

  RobotType(
    const Eigen::Vector2f& p_min,
    const Eigen::Vector2f& p_max,
    float v_max,
    float a_max,
    float tag_radius,
    float r_sense)
    : p_min(p_min)
    , p_max(p_max)
    , velocity_limit(v_max)
    , acceleration_limit(a_max)
    , tag_radiusSquared(tag_radius*tag_radius)
    , r_senseSquared(r_sense*r_sense)
  {
    init();
  }


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

    invalidAction << nanf("") , nanf("");
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotType& rt)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotType(p_min=" << rt.p_min.format(fmt)
        << ",p_max=" << rt.p_max.format(fmt)
        << ",v_max=" << rt.velocity_limit
        << ",a_max=" << rt.acceleration_limit
        << ",tag_radius=" << sqrt(rt.tag_radiusSquared)
        << ",r_sense=" << sqrt(rt.r_senseSquared)
        << ")";
    return out;
  }

};