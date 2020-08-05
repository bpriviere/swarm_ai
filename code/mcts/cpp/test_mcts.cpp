#include <iostream>

#include "GameState.hpp"

template <class T>
std::vector<std::vector<T>> cart_product(const std::vector<std::vector<T>>& v)
{
	std::vector<std::vector<T>> s = {{}};
	for (const auto& u : v) {
		std::vector<std::vector<int>> r;
		for (const auto& x : s) {
			for (const auto y : u) {
				r.push_back(x);
				r.back().push_back(y);
			}
		}
		s = std::move(r);
	}
	return s;
}


int main()
{
	size_t num_robots = 2;
	std::vector<size_t> actions = {0,1,2};
	size_t num_actions = pow(3, num_robots);

	auto result = cart_product<int>({{0,1,2},{0,1,2},{0,1,2}});
	for (const auto r : result) {
		for (const auto v : r) {
			std::cout << v << " ";
		}
		std::cout << std::endl;
	}


	return 0;


	// GameState<2,1> gs;
	// gs.turn = GameState<2,1>::Turn::Attackers;
	// gs.attackers[0].status = RobotState::Status::Active;
	// gs.attackers[0].position << 1,0;
	// gs.attackers[0].velocity << 0,0;
	// gs.attackers[1].status = RobotState::Status::Active;
	// gs.attackers[1].position << -1,0;
	// gs.attackers[1].velocity << 0,0;

	// gs.defenders[0].status = RobotState::Status::Active;
	// gs.defenders[0].position << 0,1;
	// gs.defenders[0].velocity << 0,0;

	// std::cout << gs << std::endl;

	// GameLogic<2,1> gl;

	// return 0;
}