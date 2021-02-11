#pragma once
#include <vector>

#include <Eigen/Dense>

#include "RobotState.hpp"

template<size_t EnvDim>
class RobotType
{
public:
  typedef Eigen::Matrix<float, EnvDim, 1> VectorEnv;

  RobotType() = default;

  RobotType(
    const VectorEnv& p_min,
    const VectorEnv& p_max,
    float tag_radius,
    float goal_radius,
    float r_sense,
    float radius)
    : p_min(p_min)
    , p_max(p_max)
    , tag_radiusSquared(tag_radius*tag_radius)
    , goal_radiusSquared(goal_radius*goal_radius)
    , r_senseSquared(r_sense*r_sense)
    , radius(radius)
  {
  }

  VectorEnv p_min;
  VectorEnv p_max;
  float tag_radiusSquared;
  float goal_radiusSquared;
  float r_senseSquared;
  float radius;
};