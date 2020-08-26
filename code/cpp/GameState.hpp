#pragma once

template<class Robot>
class GameState
{
public:
  typedef typename Robot::Type RobotTypeT;
  typedef typename Robot::State RobotStateT;

  enum class Turn
  {
    Attackers = 0,
    Defenders = 1,
  };

  GameState() = default;

  GameState(
    const Turn& turn,
    const std::vector<RobotStateT>& attackers,
    const std::vector<RobotStateT>& defenders)
    : turn(turn)
    , attackers(attackers)
    , attackersReward(0)
    , defenders(defenders)
    , defendersReward(0)
    , depth(0)
    , activeMask(0)
  {
    // update activeMask
    for (size_t i = 0; i < attackers.size(); ++i) {
      if (attackers[i].status == RobotStateT::Status::Active) {
        activeMask |= (1<<i);
      }
    }
  }

  Turn turn;
  std::vector<RobotStateT> attackers;
  float attackersReward;
  std::vector<RobotStateT> defenders;
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
