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
    Invalid     = 3, // collided or attempted to execute any invalid action
  };

  RobotState() = default;

  Status status;

  friend std::ostream& operator<<(std::ostream& out, const RobotState::Status& s)
  {
    switch(s) {
      case RobotState::Status::Active:
        out << "Active";
        break;
      case RobotState::Status::Captured:
        out << "Captured";
        break;
      case RobotState::Status::ReachedGoal:
        out << "ReachedGoal";
        break;
      case RobotState::Status::Invalid:
        out << "Invalid";
        break;
    }
    return out;
  }
};
