import numpy as np

from buildRelease import mctscpp

if __name__ == '__main__':

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

	# # test GLAS
	generator = mctscpp.createRandomGenerator(1)
	useGLAS = False
	if useGLAS:
		glas_a, glas_b = mctscpp.createGLAS("nn.yaml", generator)
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

	# Test MCTS search
	num_nodes = 1000

	while True:
		gs.attackersReward = 0;
		gs.defendersReward = 0;
		gs.depth = 0;
		print(gs)
		action = mctscpp.search(g, gs, generator, num_nodes)
		print(action)
		if len(action) > 0:
			success = g.step(gs, action, next_state)
			if success:
				gs = next_state
				if g.isTerminal(gs):
					print(gs)
					break
			else:
				break
		else:
			break
