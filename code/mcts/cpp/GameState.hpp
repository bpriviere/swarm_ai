#pragma once

#include "RobotState.hpp"

template <std::size_t NumAttackers, std::size_t NumDefenders>
class GameState
{
public:
  enum class Turn
  {
    Attackers = 0,
    Defenders = 1,
  };

  Turn turn;
  std::array<RobotState, NumAttackers> attackers;
  float attackersReward;
  std::array<RobotState, NumDefenders> defenders;
  float defendersReward;

  size_t depth;
  std::bitset<NumAttackers> activeMask;

  friend std::ostream& operator<<(std::ostream& out, const GameState<NumAttackers, NumDefenders>& s)
  {
    Eigen::IOFormat fmt(2, 0, ",", ";", "", "","[", "]");

    out << "GameState(turn=";
    switch(s.turn) {
      case GameState<NumAttackers, NumDefenders>::Turn::Attackers:
        out << "Attackers";
        break;
      case GameState<NumAttackers, NumDefenders>::Turn::Defenders:
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
