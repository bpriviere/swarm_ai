#pragma once
#include "RobotState.hpp"
#include "RobotType.hpp"

// Uncomment the following line to clip the environment, rather than executing a validity check
// #define CLIP_ENVIRONMENT

struct RobotStateSingleIntegrator2D
  : public RobotState
{
public:
  RobotStateSingleIntegrator2D() = default;

  RobotStateSingleIntegrator2D(
    const Eigen::Vector2f& s)
    : RobotState()
    , state(s)
  {
    status = Status::Active;
  }

  // x, y
  Eigen::Vector2f state;

  const auto position() const {
    return state.segment<2>(0);
  }

  auto position() {
    return state.segment<2>(0);
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotStateSingleIntegrator2D& s)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotState(p=" << s.position().format(fmt)
        << "," << s.status << ")";
    return out;
  }
};

typedef Eigen::Vector2f RobotActionSingleIntegrator2D; // m/s

class RobotTypeSingleIntegrator2D
  : public RobotType
{
public:

  RobotTypeSingleIntegrator2D() = default;

  RobotTypeSingleIntegrator2D(
    const Eigen::Vector2f& p_min,
    const Eigen::Vector2f& p_max,
    float v_max,
    float tag_radius,
    float r_sense,
    float radius)
    : RobotType(p_min, p_max, tag_radius, r_sense, radius)
    , velocity_limit(v_max)
  {
    init();
  }

  float velocity_limit;
  std::vector<RobotActionSingleIntegrator2D> possibleActions;
  RobotActionSingleIntegrator2D invalidAction;

  void step(const RobotStateSingleIntegrator2D& state,
    const RobotActionSingleIntegrator2D& action,
    float dt,
    RobotStateSingleIntegrator2D& result) const
  {
#ifdef CLIP_ENVIRONMENT
    auto position = state.state + action * dt;
    result.state = position.cwiseMin(p_max).cwiseMax(p_min);
#else
    result.state = state.state + action * dt;
#endif
  }

  bool isStateValid(const RobotStateSingleIntegrator2D& state) const
  {
#ifdef CLIP_ENVIRONMENT
    return true;
#else
    return (state.position().array() >= p_min.array()).all() && (state.position().array() <= p_max.array()).all();
#endif
  }

  void init()
  {
    possibleActions.resize(9);
    possibleActions[0] << -velocity_limit / sqrtf(2), -velocity_limit / sqrtf(2);
    possibleActions[1] << -velocity_limit, 0;
    possibleActions[2] << -velocity_limit / sqrtf(2), velocity_limit / sqrtf(2);
    possibleActions[3] << 0, -velocity_limit;
    possibleActions[4] << 0, 0;
    possibleActions[5] << 0, velocity_limit;
    possibleActions[6] << velocity_limit / sqrtf(2), -velocity_limit / sqrtf(2);
    possibleActions[7] << velocity_limit, 0;
    possibleActions[8] << velocity_limit / sqrtf(2), velocity_limit / sqrtf(2);

    invalidAction << nanf("") , nanf("");
  }

  float actionLimit() const {
    return velocity_limit;
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotTypeSingleIntegrator2D& rt)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotTypeSI2D(p_min=" << rt.p_min.format(fmt)
        << ",p_max=" << rt.p_max.format(fmt)
        << ",v_max=" << rt.velocity_limit
        << ",tag_radius=" << sqrt(rt.tag_radiusSquared)
        << ",r_sense=" << sqrt(rt.r_senseSquared)
        << ")";
    return out;
  }

};

class SingleIntegrator2D
{
public:
  typedef RobotStateSingleIntegrator2D State;
  typedef RobotActionSingleIntegrator2D Action;
  typedef RobotTypeSingleIntegrator2D Type;

  static const size_t StateDim = 2;
  static const size_t ActionDim = 2;
};
