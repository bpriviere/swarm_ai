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
    , defenders(defenders)
    , depth(0)
    , cumulativeReward(0)
  {
  }

  Turn turn;
  std::vector<RobotStateT> attackers;
  std::vector<RobotStateT> defenders;

  size_t depth;
  float cumulativeReward;

  bool isApprox(const GameState<Robot>& other) const
  {
    if (turn != other.turn) { 
      return false;
    }

    for (size_t i = 0; i < attackers.size(); ++i) {
      if (!attackers[i].isApprox(other.attackers[i])) {
        return false;
      }
    }

    for (size_t i = 0; i < defenders.size(); ++i) {
      if (!defenders[i].isApprox(other.defenders[i])) {
        return false;
      }
    }
    return true;
  }

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
    out <<"depth=" << s.depth;
    out <<",cumReward=" << s.cumulativeReward;
    out << ")";
    return out;
  }
};
