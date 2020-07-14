
import numpy as np 


def relative_state(nodes,r_sense):

	# make neighbors
	observations = dict()

	for node_i in nodes:

		o_a = []
		o_b = [] 

		for node_j in nodes: 
			if node_j is not node_i and np.linalg.norm(node_j.state[0:2] - node_i.state[0:2]) < r_sense: 
				if node_j.team_A: 
					o_a.append(node_j.state - node_i.state)
				elif node_j.team_B:
					o_b.append(node_j.state - node_i.state)

		observations[node_i] = (np.array(o_a).flatten(),np.array(o_b).flatten())

	return observations