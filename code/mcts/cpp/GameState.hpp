#pragma once
#include <array>
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

	Eigen::Vector2f position;	// m
	Eigen::Vector2f velocity;	// m/s
	Status status;
};

std::ostream& operator<<(std::ostream& out, const RobotState& s)
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
	std::array<RobotState, NumDefenders> defenders;
};

template <std::size_t NumAttackers, std::size_t NumDefenders>
std::ostream& operator<<(std::ostream& out, const GameState<NumAttackers, NumDefenders>& s)
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

template <std::size_t NumAttackers, std::size_t NumDefenders>
class GameLogic
{
public:
	typedef GameState<NumAttackers, NumDefenders> GameStateT;


	bool step(const GameStateT& state, const Eigen::Vector2f& action, float dt, GameStateT& result)
	{
		// copy current state
		result = state;
		// update active robots
		if (state.turn == GameStateT::Turn::Attackers) {
			for(auto& attacker : result.attackers) {
				if (attacker.status == RobotState::Status::Active) {
					step(attacker, action, dt, attacker);
				}
			}
			result.turn = GameStateT::Turn::Defenders;
		} else {
			for(auto& defender : result.defenders) {
				step(defender, action, dt, defender);
			}
			result.turn = GameStateT::Turn::Attackers;
		}
		return true;
	}

	bool isFinished(const GameStateT& state)
	{
		for (const auto& attacker : state.attackers) {
			if (attacker.status == RobotState::Status::Active) {
				return false;
			}
		}
		return true;
	}

	std::tuple<float,float> reward(const GameStateT& state)
	{
		float reward1 = 0;
		for (const auto& attacker : state.attackers) {
			switch (attacker.status) {
				case RobotState::Status::Active:
					reward1 += 0.5;
					break;
				case RobotState::Status::ReachedGoal:
					reward1 += 1.0;
					break;
				case RobotState::Status::Captured:
					break;
			}
		}
		reward1 /= state.attackers.size();
		return std::make_tuple(reward1, 1.0 - reward1);
	}

private:
	bool step(const RobotState& state, const Eigen::Vector2f& action, float dt, RobotState& result)
	{
		result.position = state.position + state.velocity * dt;
		result.velocity = state.velocity + action * dt;
		return true;
	}


};