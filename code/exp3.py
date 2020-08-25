import argparse
import yaml
import sys
from collections import defaultdict
import numpy as np

# my packages
sys.path.append("glas/")
sys.path.append('mcts/cpp')

from param import Param
from grun import createGLAS, create_robot_type_cpp
from buildRelease import mctscpp
from mice import game_state_to_cpp_result
import datahandler as dh
import plotter


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("file", help="policy (*.pt) to test")
	parser.add_argument("team", help="a or b")
	args = parser.parse_args()

	with open('exp3.yaml') as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)

	if args.team == 'a':
		policies_a = [args.file]
		policies_b = [p['file'] for p in cfg["defenderPolicies"]]
	elif args.team == 'b':
		policies_a = [p['file'] for p in cfg["attackerPolicies"]]
		policies_b = [args.file]
	else:
		raise Exception("Unknown team type")

	param = Param()
	generator = mctscpp.createRandomGenerator(param.seed)

	robot_types = dict()
	robot_types['standard_robot'] = create_robot_type_cpp(param, 'standard_robot', 'a')

	score = 0
	for game in cfg['games']:
		for policy_a in policies_a:
			glas_a = createGLAS(policy_a, generator)
			for policy_b in policies_b:
				glas_b = createGLAS(policy_b, generator)
				
				attackerTypes = []
				attackers = []
				for robot in game["team_a"]:
					attackerTypes.append(robot_types[robot['type']])
					attackers.append(mctscpp.RobotState(robot["x0"][0:2],robot["x0"][2:]))
				defenderTypes = []
				defenders = []
				for robot in game["team_b"]:
					defenderTypes.append(robot_types[robot['type']])
					defenders.append(mctscpp.RobotState(robot["x0"][0:2],robot["x0"][2:]))

				param.num_nodes_A = len(attackers)
				param.num_nodes_B = len(defenders)
				param.num_nodes = param.num_nodes_A + param.num_nodes_B
				param.goal = game["goal"]

				max_depth = 10000000
				rollout_beta = 0
				g = mctscpp.Game(attackerTypes, defenderTypes, param.sim_dt, game["goal"], max_depth, generator, glas_a, glas_b, rollout_beta)
				gs = mctscpp.GameState(mctscpp.GameState.Turn.Attackers,attackers,defenders)

				results = []
				next_state = mctscpp.GameState()
				while True:
					gs.attackersReward = 0
					gs.defendersReward = 0
					gs.depth = 0
					
					deterministic = True
					action = mctscpp.computeActionsWithGLAS(glas_a, glas_b, gs, game["goal"], attackerTypes, defenderTypes, generator, deterministic)
					# step twice (once per team)
					success = g.step(gs, action, next_state)
					if success:
						gs.attackersReward = 0
						gs.defendersReward = 0
						gs.depth = 0
						success = g.step(next_state, action, next_state)

					if not success:
						break
					results.append(game_state_to_cpp_result(gs,action))

					if g.isTerminal(gs):
						break

					gs = next_state

				sim_result = dh.convert_cpp_data_to_sim_result(np.array(results),param)
				score += sim_result['rewards'][-1,0]
				plotter.plot_tree_results(sim_result)

	print("SCORE: ", score)


	print('saving and opening figs...')
	plotter.save_figs("exp3.pdf")
	plotter.open_figs("exp3.pdf")
