#pragma once
#include "RobotState.hpp"
#include "RobotType.hpp"

// Uncomment the following line to clip the environment, rather than executing a validity check
// #define CLIP_ENVIRONMENT
// Uncomment the following line to scale the velocity
#define SCALE_VELOCITY


struct RobotStateDubins3D
  : public RobotState
{
public:
  RobotStateDubins3D() = default;

  RobotStateDubins3D(
    const Eigen::Matrix<float, 6, 1>& s)
    // const Eigen::Vector6f& s)
    : RobotState()
    , state(s)
  {
    status = Status::Active;
  }

  // x, y, z, phi, psi, V 
  Eigen::Matrix<float,6,1> state;
  // Eigen::Vector6f state;

  const auto position() const {
    return state.segment<3>(0);
  }

  auto position() {
    return state.segment<3>(0);
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

  const auto position_Z() const {
    return state.segment<1>(2);
  }

  auto position_Z() {
    return state.segment<1>(2);
  }  
  
  const auto phi() const {
    return state.segment<1>(3);
  }

  auto phi() {
    return state.segment<1>(3);
  }

  const auto psi() const {
    return state.segment<1>(4);
  }

  auto psi() {
    return state.segment<1>(4);
  }  

  const auto velocity() const {
    return state.segment<1>(5);
  }

  auto velocity() {
    return state.segment<1>(5);
  }  


  bool isApprox(const RobotStateDubins3D& other) const
  {
    const float epsilon = 1e-3;
    // const float epsilon = 1e-2;
    return status == other.status && (state - other.state).squaredNorm() < epsilon*epsilon;
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotStateDubins3D& s)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotState(p=" << s.position().format(fmt)
        << "," << s.status << ")";
    return out;
  }
};

typedef Eigen::Vector3f RobotActionDubins3D; // phidot, psidot, Vdot 

class RobotTypeDubins3D
  : public RobotType<3>
{
public:

  RobotTypeDubins3D() = default;

  RobotTypeDubins3D(
    const Eigen::Vector3f& p_min,
    const Eigen::Vector3f& p_max,
    float v_max,
    float a_max,
    float tag_radius,
    float r_sense,
    float radius)
    : RobotType(p_min, p_max, tag_radius, r_sense, radius)
    , velocity_limit(v_max)
    , acceleration_limit(a_max) // treat this one as an angular acceleration limit
  {
    init();
  }

  float velocity_limit;
  float acceleration_limit;
  RobotActionDubins3D invalidAction;

  void step(const RobotStateDubins3D& state,
    const RobotActionDubins3D& action,
    float dt,
    RobotStateDubins3D& result) const
  {
    // state: x,y,z,phi,psi,V
    // action: phidot, psidot, Vdot
    result.position_X() = state.position_X() + state.velocity()*cos(state.phi()(0))*sin(state.psi()(0)) * dt;
    result.position_Y() = state.position_Y() + state.velocity()*cos(state.phi()(0))*cos(state.psi()(0)) * dt;
    result.position_Z() = state.position_Z() - state.velocity()*sin(state.phi()(0)) * dt;
    result.phi()        = state.phi() + action.segment<1>(0) * dt;
    result.psi()        = state.psi() + action.segment<1>(1) * dt;

    auto velocity       = state.velocity() + action.segment<1>(2) * dt;
    float alpha = velocity.norm() / velocity_limit;
    result.velocity() = velocity / std::max(alpha, 1.0f);    
  }

  bool isStateValid(const RobotStateDubins3D& state) const
  {
    bool positionValid;
    bool velocityValid;
    // positionValid = (state.position_X() >= p_min.segment<1>(0) && state.position_Y() >= p_min.segment<1>(1) && state.position_X() <= p_max.segment<1>(0) && state.position_Y() <= p_max.segment<1>(1));
    positionValid = true;
    velocityValid = true;
    return positionValid && velocityValid;
  }

  void init()
  {
    invalidAction << nanf("") , nanf("");
  }

  float actionLimit() const {
    return acceleration_limit;
  }

  RobotActionDubins3D sampleActionUniform(std::default_random_engine& generator) const {
    // use uniform random sample (no deterministic option)
    // std::uniform_real_distribution<float> distTheta(0.0, 2*M_PI);
    // std::uniform_real_distribution<float> distMag(0.0, 1.0);
    // float theta = distTheta(generator);
    // float mag = sqrtf(distMag(generator)) * actionLimit();
    return RobotActionDubins3D(0, 0, 0);
  }

  friend std::ostream& operator<<(std::ostream& out, const RobotTypeDubins3D& rt)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotTypeDubins3D(p_min=" << rt.p_min.format(fmt)
        << ",p_max=" << rt.p_max.format(fmt)
        << ",v_max=" << rt.velocity_limit
        << ",a_max=" << rt.acceleration_limit
        << ",tag_radius=" << sqrt(rt.tag_radiusSquared)
        << ",r_sense=" << sqrt(rt.r_senseSquared)
        << ")";
    return out;
  }

};

class Dubins3D
{
public:
  typedef RobotStateDubins3D State;
  typedef RobotActionDubins3D Action;
  typedef RobotTypeDubins3D Type;

  static constexpr int StateDim = 6;
  static constexpr int ActionDim = 3;
};
