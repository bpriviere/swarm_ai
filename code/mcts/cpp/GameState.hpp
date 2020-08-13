#pragma once

#include "RobotState.hpp"

class GameState
{
public:
  enum class Turn
  {
    Attackers = 0,
    Defenders = 1,
  };

  Turn turn;
  std::vector<RobotState> attackers;
  float attackersReward;
  std::vector<RobotState> defenders;
  float defendersReward;

  size_t depth;
  // WARNING: this assumes we have less than 32 attackers
  uint32_t activeMask;

  friend std::ostream& operator<<(std::ostream& out, const GameState& s)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "GameState(turn=";
    switch(s.turn) {
      case GameState::Turn::Attackers:
        out << "Attackers";
        break;
      case GameState::Turn::Defenders:
        out << "Defenders";
        break;
    }
    out << ",attackers=";
    for(const auto& attacker : s.attackers) {
      out << attacker << ",";
    }
    out << "defenders=";
    for(const auto& defender : s.defenders) {
      out << defender << ",";
    }
    out << ")";
    return out;
  }
};
