
import numpy as np 


def relative_state(nodes,abs_goal,flatten=False):

	# make neighbors
	observations = dict()

	for node_i in nodes:

		o_a = []
		o_b = [] 

		if node_i.state.ndim == 1: 
			abs_goal = np.array((abs_goal[0],abs_goal[1],0,0))
		elif node_i.state.ndim == 2: 
			abs_goal = np.array((abs_goal[0],abs_goal[1],0,0),dtype=float)[:,np.newaxis]
		
		goal = abs_goal - node_i.state

		for node_j in nodes: 
			if node_j is not node_i and np.linalg.norm(node_j.state[0:2] - node_i.state[0:2]) < node_i.r_sense: 
				if node_j.team_A: 
					o_a.append(node_j.state - node_i.state)
				elif node_j.team_B:
					o_b.append(node_j.state - node_i.state)

		if flatten:
			observations[node_i] = (np.array(o_a).flatten(),np.array(o_b).flatten(),np.array(goal).flatten())
		else:
			observations[node_i] = (o_a,o_b,goal)

	return observations