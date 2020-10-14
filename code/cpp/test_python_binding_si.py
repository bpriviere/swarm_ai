import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from buildRelease import mctscppsi as mctscpp

def loadFeedForwardNNWeights(ff, state_dict, name):
	l = 0
	while True:
		key1 = "{}.layers.{}.weight".format(name, l)
		key2 = "{}.layers.{}.bias".format(name, l)
		if key1 in state_dict and key2 in state_dict:
			ff.addLayer(state_dict[key1].numpy(), state_dict[key2].numpy())
		else:
			if l == 0:
				print("WARNING: No weights found for {}".format(name))
			break
		l += 1

def loadGLAS(glas, file):
	state_dict = torch.load(file)

	loadFeedForwardNNWeights(glas.deepSetA.phi, state_dict, "model_team_a.phi")
	loadFeedForwardNNWeights(glas.deepSetA.rho, state_dict, "model_team_a.rho")
	loadFeedForwardNNWeights(glas.deepSetB.phi, state_dict, "model_team_b.phi")
	loadFeedForwardNNWeights(glas.deepSetB.rho, state_dict, "model_team_b.rho")
	loadFeedForwardNNWeights(glas.psi, state_dict, "psi")
	loadFeedForwardNNWeights(glas.encoder, state_dict, "encoder")
	loadFeedForwardNNWeights(glas.decoder, state_dict, "decoder")
	loadFeedForwardNNWeights(glas.value, state_dict, "value")

	return glas

if __name__ == '__main__':
	mode = "MCTS_RANDOM" # one of "GLAS", "MCTS_RANDOM", "MCTS_GLAS"
	num_nodes = 10000
	export_dot = None # or "mcts.dot"

	# test RobotState
	rs = mctscpp.RobotState([0,1])
	print(rs)

	# test GameState
	attackers = [mctscpp.RobotState([0.05,0.25])]
	defenders = [mctscpp.RobotState([0.3,0.25])]
	gs = mctscpp.GameState(mctscpp.GameState.Turn.Attackers,attackers,defenders)
	print(gs)

	# test RobotType
	rt = mctscpp.RobotType([0.0, 0.0], [0.5, 0.5], 0.125, 0.025, 1.0, 0.025)
	# rt.p_min = [0.0, 0.0]
	# rt.p_max = [0.5, 0.5]
	# rt.velocity_limit = 0.125
	# rt.acceleration_limit = 0.125
	# rt.tag_radiusSquared = 0.025**2
	# rt.r_senseSquared = 1.0**2

	# test Game
	attackerTypes = [rt]
	defenderTypes = [rt]
	dt = 0.25
	goal = [0.25,0.25]
	max_depth = 100
	beta2 = 0.5 # 0 means pure random, 1.0 means pure GLAS
	Cp = 1.4
	pw_C = 1.0
	pw_alpha = 0.25
	beta1 = 0
	beta3 = 0
	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, max_depth)
	policyA = mctscpp.Policy('a')
	policyB = mctscpp.Policy('b')
	if "GLAS" in mode:
		loadGLAS(policyA.glas, "../../current/models/a1.pt")
		loadGLAS(policyB.glas, "../../current/models/b1.pt")
		policyA.beta2 = beta2
		policyB.beta2 = beta2
	if "RANDOM" in mode:
		policyA.beta2 = 0.0
		policyB.beta2 = 0.0
		vf_beta = 0
	print(policyA)
	print(policyB)
	print(g)

	next_state = mctscpp.GameState()
	success = g.step(gs, [[0,2],[0,1]], next_state)
	print(success, next_state)

	# Test Game Rollout
	fig, ax = plt.subplots()
	ax.add_patch(mpatches.Circle(goal, 0.025,alpha=0.5))

	result = []
	for d in range(max_depth):
		gs.depth = 0;
		# print(gs)
		result.append([
			[rs.state[0:2].copy() for rs in gs.attackers],
			[rs.state[0:2].copy() for rs in gs.defenders]])
		if "MCTS" in mode:
			if gs.turn == mctscpp.GameState.Turn.Attackers:
				myPolicy = policyA
				opponentPolicies = [policyB]
			else:
				myPolicy = policyB
				opponentPolicies = [policyA]

			mctsresult = mctscpp.search(g, gs, myPolicy, opponentPolicies, num_nodes, Cp, pw_C, pw_alpha, beta1, beta3, export_dot)
			if export_dot:
				print("Run 'dot -Tpng mcts.dot -o mcts.png' to visualize!")
				exit()
			if mctsresult.success:
				print(mctsresult.expectedReward)
				# print(mctsresult.valuePerAction)

				if gs.turn == mctscpp.GameState.Turn.Attackers:
					actionIdx = 0
					robots = gs.attackers
				else:
					actionIdx = 1
					robots = gs.defenders
				for robot in robots:
					x = robot.state[0]
					y = robot.state[1]
					for action, value in mctsresult.valuePerAction:
						dx = action[actionIdx][0]
						dy = action[actionIdx][1]
						p = value
						color = None
						if (action[actionIdx] == mctsresult.bestAction[actionIdx]).all():
							color = 'red'
						ax.arrow(x, y, dx * 0.01, dy*0.01, width=p*0.001, color=color)

				success = g.step(gs, mctsresult.bestAction, gs)
			else:
				break
		elif mode == "GLAS":
			deterministic = True
			action = mctscpp.eval(g, gs, deterministic)
			# step twice (once per team)
			success = g.step(gs, action, gs)
			if success:
				success = g.step(gs, action, gs)
		else:
			print("UNKNOWN MODE")
			exit()
		if success:
			if g.isTerminal(gs):
				print(gs)
				result.append([
					[rs.state[0:2].copy() for rs in gs.attackers],
					[rs.state[0:2].copy() for rs in gs.defenders]])
				break
		else:
			break

	print(result[0])
	# exit()
	result = np.array(result)
	# print(result[:,1,0,0])

	
	ax.axis('equal')
	for i in range(result.shape[2]):
		ax.plot(result[:,0,i,0], result[:,0,i,1], label="attacker {}".format(i))
	for i in range(result.shape[2]):
		ax.plot(result[:,1,i,0], result[:,1,i,1], label="defender {}".format(i))

	ax.legend()
	plt.show()

