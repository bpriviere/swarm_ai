#pragma once
#include "RobotState.hpp"
#include "RobotType.hpp"

// Uncomment the following line to clip the environment, rather than executing a validity check
// #define CLIP_ENVIRONMENT
// Uncomment the following line to scale the velocity
#define SCALE_VELOCITY

struct RobotStateDoubleIntegrator2D
  : public RobotState
{
public:
  RobotStateDoubleIntegrator2D() = default;

  RobotStateDoubleIntegrator2D(
    const Eigen::Vector4f& s)
    : RobotState()
    , state(s)
  {
    status = Status::Active;
  }

  // x, y, vx, vy
  Eigen::Vector4f state;

  const auto position() const {
    return state.segment<2>(0);
  }

  auto position() {
    return state.segment<2>(0);
  }

  const auto velocity() const {
    return state.segment<2>(2);
  }

  auto velocity() {
    return state.segment<2>(2);
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotStateDoubleIntegrator2D& s)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotState(p=" << s.position().format(fmt)
        << ",v=" << s.velocity().format(fmt)
        << "," << s.status << ")";
    return out;
  }
};

typedef Eigen::Vector2f RobotActionDoubleIntegrator2D; // m/s^2

class RobotTypeDoubleIntegrator2D
  : public RobotType
{
public:

  RobotTypeDoubleIntegrator2D() = default;

  RobotTypeDoubleIntegrator2D(
    const Eigen::Vector2f& p_min,
    const Eigen::Vector2f& p_max,
    float v_max,
    float a_max,
    float tag_radius,
    float r_sense,
    float radius)
    : RobotType(p_min, p_max, tag_radius, r_sense, radius)
    , velocity_limit(v_max)
    , acceleration_limit(a_max)
  {
    init();
  }

  float velocity_limit;
  float acceleration_limit;
  std::vector<RobotActionDoubleIntegrator2D> possibleActions;
  RobotActionDoubleIntegrator2D invalidAction;

  void step(const RobotStateDoubleIntegrator2D& state,
    const RobotActionDoubleIntegrator2D& action,
    float dt,
    RobotStateDoubleIntegrator2D& result) const
  {
#ifdef CLIP_ENVIRONMENT
    auto position = state.position() + state.velocity() * dt;
    result.position() = position.cwiseMin(p_max).cwiseMax(p_min);
#else
    result.position() = state.position() + state.velocity() * dt;
#endif

#ifdef SCALE_VELOCITY
    auto velocity = state.velocity() + action * dt;
    float alpha = velocity.norm() / velocity_limit;
    result.velocity() = velocity / std::max(alpha, 1.0f);
#else
    result.velocity() = state.velocity() + action * dt;
#endif
  }

  bool isStateValid(const RobotStateDoubleIntegrator2D& state) const
  {
    bool positionValid;
    bool velocityValid;
#ifdef CLIP_ENVIRONMENT
    positionValid = true;
#else
    positionValid = (state.position().array() >= p_min.array()).all() && (state.position().array() <= p_max.array()).all();
    // positionValid = true;
#endif

#ifdef SCALE_VELOCITY
    velocityValid = true;
#else
    velocityValid = state.velocity().norm() < velocity_limit;
#endif

    return positionValid && velocityValid;
  }

  void init()
  {
    possibleActions.resize(9);
    possibleActions[0] << -acceleration_limit / sqrtf(2), -acceleration_limit / sqrtf(2);
    possibleActions[1] << -acceleration_limit, 0;
    possibleActions[2] << -acceleration_limit / sqrtf(2), acceleration_limit / sqrtf(2);
    possibleActions[3] << 0, -acceleration_limit;
    possibleActions[4] << 0, 0;
    possibleActions[5] << 0, acceleration_limit;
    possibleActions[6] << acceleration_limit / sqrtf(2), -acceleration_limit / sqrtf(2);
    possibleActions[7] << acceleration_limit, 0;
    possibleActions[8] << acceleration_limit / sqrtf(2), acceleration_limit / sqrtf(2);

    invalidAction << nanf("") , nanf("");
  }

  float actionLimit() const {
    return acceleration_limit;
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotTypeDoubleIntegrator2D& rt)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotTypeDI2D(p_min=" << rt.p_min.format(fmt)
        << ",p_max=" << rt.p_max.format(fmt)
        << ",v_max=" << rt.velocity_limit
        << ",a_max=" << rt.acceleration_limit
        << ",tag_radius=" << sqrt(rt.tag_radiusSquared)
        << ",r_sense=" << sqrt(rt.r_senseSquared)
        << ")";
    return out;
  }

};

class DoubleIntegrator2D
{
public:
  typedef RobotStateDoubleIntegrator2D State;
  typedef RobotActionDoubleIntegrator2D Action;
  typedef RobotTypeDoubleIntegrator2D Type;

  static constexpr int StateDim = 4;
  static constexpr int ActionDim = 2;
};
