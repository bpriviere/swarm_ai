#pragma once
#include <iostream>
#include <Eigen/Dense>

struct RobotState
{
public:
  enum class Status
  {
    Active      = 0,
    Captured    = 1,
    ReachedGoal = 2,
  };

  Eigen::Vector2f position; // m
  Eigen::Vector2f velocity; // m/s
  Status status;

  friend std::ostream& operator<<(std::ostream& out, const RobotState& s)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "RobotState(p=" << s.position.format(fmt) << ",v=" << s.velocity.format(fmt) << ",";
    switch(s.status) {
      case RobotState::Status::Active:
        out << "Active";
        break;
      case RobotState::Status::Captured:
        out << "Captured";
        break;
      case RobotState::Status::ReachedGoal:
        out << "ReachedGoal";
        break;
    }
    out << ")";
    return out;
  }
};
