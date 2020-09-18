#pragma once
#include "RobotState.hpp"
#include "RobotType.hpp"

//#include <math.h>

// Uncomment the following line to clip the environment, rather than executing a validity check
// P.S: They don't work at the moment...
// #define CLIP_ENVIRONMENT
// Uncomment the following line to scale the velocity
// #define SCALE_VELOCITY

struct RobotStateDubins2D
  : public RobotState
{
public:
  RobotStateDubins2D() = default;

  RobotStateDubins2D(
    const Eigen::Vector4f& s)
    : RobotState()
    , state(s)
  {
    status = Status::Active;
  }

  // x, y, theta, v_dot
  Eigen::Vector4f state;

  const auto position() const {
    return state.segment<2>(0);
  }

  auto position() {
    return state.segment<2>(0);
  }

  const auto position_X() const {
    return state.segment<1>(0);
  }

  auto position_X() {
    return state.segment<1>(0);
  }
  
  const auto position_Y() const {
    return state.segment<1>(1);
  }

  auto position_Y() {
    return state.segment<1>(1);
  }
  
  const auto theta() const {
    return state.segment<1>(2);
  }

  auto theta() {
    return state.segment<1>(2);
  }

  const auto velocity() const {
    return state.segment<1>(3);
  }

  auto velocity() {
    return state.segment<1>(3);
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotStateDubins2D& s)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotState( ("<< s.position_X().format(fmt)
        << ","<< s.position_Y().format(fmt)
        << "),theta=" << s.theta().format(fmt)
        << ",v=" << s.velocity().format(fmt)
        << "," << s.status << ")";
    return out;
  }
};

typedef Eigen::Vector2f RobotActionDubins2D; // theta_dot, V_dot [ rad/s, m/s^2 ]

class RobotTypeDubins2D
  : public RobotType
{
public:

  RobotTypeDubins2D() = default;

  RobotTypeDubins2D( const Eigen::Vector2f& p_min, const Eigen::Vector2f& p_max, float v_max, float a_max, float tag_radius, float r_sense, float radius)
    : RobotType(p_min, p_max, tag_radius, r_sense, radius)
    , velocity_limit(v_max)
    , acceleration_limit(a_max)
  {
    init();
  }

  float velocity_limit;
  float acceleration_limit;
  std::vector<RobotActionDubins2D> possibleActions;
  RobotActionDubins2D invalidAction;

  // Iterate a timestep
  void step(const RobotStateDubins2D& state, const RobotActionDubins2D& action, float dt, RobotStateDubins2D& result) const
  {
    //result.position_X() = state.position_X() + state.velocity()*state.theta().cos() * dt;
    //result.position_Y() = state.position_Y() + state.velocity()*state.theta().sin() * dt;

    result.position_X() = state.position_X() + dt * state.velocity()*state.theta() * dt;  // Missing the cos() and sin() parts, but at least it compiles...
    result.position_Y() = state.position_Y() + dt * state.velocity()*state.theta() * dt;

    result.theta()      = state.theta()    + dt * action.segment<1>(0) / std::max(state.velocity().norm(), 0.1f);
    result.velocity()   = state.velocity() + dt * action.segment<1>(1);

  }

  // Check if current state is valid
  bool isStateValid(const RobotStateDubins2D& state) const
  {
    bool positionValid;
    bool velocityValid;
    
    // Check if the position is valid
    positionValid = true;
    positionValid = (state.position().array() >= p_min.array()).all() && (state.position().array() <= p_max.array()).all();
    
    // Check if velocity is valid
    velocityValid = state.velocity().norm() < velocity_limit;

    // Return 
    return positionValid && velocityValid;
  }

  // Intialise (currently the action space)
  void init()
  {
    // Possible actions (theta_dot, V_dot)
    //    - no change: 0 for both
    //    - turning: +- accel_lim / min(0.1,V) -> The velocity limiting is applied in step() 
    //    - accel: +- accel_limit 
    possibleActions.resize(9);

    possibleActions[0] << -acceleration_limit*0.1f, -acceleration_limit;
    possibleActions[1] << -acceleration_limit*0.1f,                 0.0;
    possibleActions[2] << -acceleration_limit*0.1f,  acceleration_limit;
    
    possibleActions[3] << 0.0, -acceleration_limit;
    possibleActions[4] << 0.0,                 0.0;
    possibleActions[5] << 0.0,  acceleration_limit;
    
    possibleActions[6] << acceleration_limit*0.1f, -acceleration_limit;
    possibleActions[7] << acceleration_limit*0.1f,                 0.0;
    possibleActions[8] << acceleration_limit*0.1f,  acceleration_limit;

    invalidAction << nanf("") , nanf("");
  }

  float actionLimit() const {
    return acceleration_limit;
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotTypeDubins2D& rt)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotTypeDubins2D(p_min=" << rt.p_min.format(fmt)
        << ",p_max=" << rt.p_max.format(fmt)
        << ",v_max=" << rt.velocity_limit
        << ",a_max=" << rt.acceleration_limit
        << ",tag_radius=" << sqrt(rt.tag_radiusSquared)
        << ",r_sense=" << sqrt(rt.r_senseSquared)
        << ")";
    return out;
  }

};

class Dubins2D
{
public:
  typedef RobotStateDubins2D State;
  typedef RobotActionDubins2D Action;
  typedef RobotTypeDubins2D Type;

  static constexpr int StateDim = 4;   // (x, y, theta, v)
  static constexpr int ActionDim = 2;  // (turn left/right, speed up/down)
};