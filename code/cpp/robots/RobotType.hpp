#pragma once
#include <vector>

#include <Eigen/Dense>

#include "RobotState.hpp"

class RobotType
{
public:

  RobotType() = default;

  RobotType(
    const Eigen::Vector2f& p_min,
    const Eigen::Vector2f& p_max,
    float tag_radius,
    float r_sense)
    : p_min(p_min)
    , p_max(p_max)
    , tag_radiusSquared(tag_radius*tag_radius)
    , r_senseSquared(r_sense*r_sense)
  {
  }

  Eigen::Vector2f p_min;
  Eigen::Vector2f p_max;
  float tag_radiusSquared;
  float r_senseSquared;
};