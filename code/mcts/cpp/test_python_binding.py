import numpy as np
import matplotlib.pyplot as plt
import torch

from buildRelease import mctscpp

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

def createGLAS(file, generator):
	state_dict = torch.load(file)

	glas = mctscpp.GLAS(generator)
	den = glas.discreteEmptyNet
	loadFeedForwardNNWeights(den.deepSetA.phi, state_dict, "model_team_a.phi")
	loadFeedForwardNNWeights(den.deepSetA.rho, state_dict, "model_team_a.rho")
	loadFeedForwardNNWeights(den.deepSetB.phi, state_dict, "model_team_b.phi")
	loadFeedForwardNNWeights(den.deepSetB.rho, state_dict, "model_team_b.rho")
	loadFeedForwardNNWeights(den.psi, state_dict, "psi")

	return glas

if __name__ == '__main__':
	seed = 10
	mode = "MCTS_GLAS" # one of "GLAS", "MCTS_RANDOM", "MCTS_GLAS"
	num_nodes = 1000

	# test RobotState
	rs = mctscpp.RobotState([0,1],[2,3])
	print(rs)
	rs.position=[0.5,1.5]
	rs.velocity=np.array([2.5,3.5])
	rs.status = mctscpp.RobotState.Status.ReachedGoal
	print(rs)

	# test GameState
	attackers = [mctscpp.RobotState([0.05,0.25],[0,0])]
	defenders = [mctscpp.RobotState([0.3,0.25],[0,0])]
	gs = mctscpp.GameState(mctscpp.GameState.Turn.Attackers,attackers,defenders)
	print(gs)

	# test RobotType
	rt = mctscpp.RobotType([0.0, 0.0], [0.5, 0.5], 0.125, 0.125, 0.025, 1.0)
	# rt.p_min = [0.0, 0.0]
	# rt.p_max = [0.5, 0.5]
	# rt.velocity_limit = 0.125
	# rt.acceleration_limit = 0.125
	# rt.tag_radiusSquared = 0.025**2
	# rt.r_senseSquared = 1.0**2

	# test GLAS
	generator = mctscpp.createRandomGenerator(seed)
	if "GLAS" in mode:
		glas_a = createGLAS("../../../models/il_current_a.pt", generator)
		glas_b = createGLAS("../../../models/il_current_b.pt", generator)
	else:
		glas_a = glas_b = None

	# test Game
	attackerTypes = [rt]
	defenderTypes = [rt]
	dt = 0.25
	goal = [0.25,0.25]
	max_depth = 1000
	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, max_depth, generator, glas_a, glas_b)
	print(g)

	next_state = mctscpp.GameState()
	success = g.step(gs, [[0,2],[0,1]], next_state)
	print(success, next_state)

	# Test Game Rollout

	result = []
	while True:
		gs.attackersReward = 0;
		gs.defendersReward = 0;
		gs.depth = 0;
		# print(gs)
		result.append([
			[rs.position.copy() for rs in gs.attackers],
			[rs.position.copy() for rs in gs.defenders]])
		if "MCTS" in mode:
			action = mctscpp.search(g, gs, generator, num_nodes)
			if len(action) > 0:
				success = g.step(gs, action, gs)
			else:
				break
		elif mode == "GLAS":
			action = mctscpp.computeActionsWithGLAS(glas_a, glas_b, gs, goal, attackerTypes, defenderTypes, generator)
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
					[rs.position.copy() for rs in gs.attackers],
					[rs.position.copy() for rs in gs.defenders]])
				break
		else:
			break

	print(result[0])
	# exit()
	result = np.array(result)
	# print(result[:,1,0,0])

	fig, ax = plt.subplots()
	ax.axis('equal')
	for i in range(result.shape[2]):
		ax.plot(result[:,0,i,0], result[:,0,i,1], label="attacker {}".format(i))
	for i in range(result.shape[2]):
		ax.plot(result[:,1,i,0], result[:,1,i,1], label="defender {}".format(i))

	ax.legend()
	plt.show()

