#include <iostream>

#include "GameState.hpp"



int main()
{
	GameState<2,1> gs;
	gs.turn = GameState<2,1>::Turn::Attackers;
	gs.attackers[0].status = RobotState::Status::Active;
	gs.attackers[0].position << 1.23,0;
	gs.attackers[0].velocity << 0,0;

	std::cout << gs << std::endl;

	GameLogic<2,1> gl;

	return 0;
}