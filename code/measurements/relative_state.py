
import numpy as np 


# def relative_state(nodes,abs_goal,flatten=False):

# 	# make neighbors
# 	observations = dict()

# 	for node_i in nodes:

# 		o_a = []
# 		o_b = [] 

# 		if node_i.state.ndim == 1: 
# 			abs_goal = np.array((abs_goal[0],abs_goal[1],0,0))
# 		elif node_i.state.ndim == 2: 
# 			abs_goal = np.array((abs_goal[0],abs_goal[1],0,0),dtype=float)[:,np.newaxis]
		
# 		goal = abs_goal - node_i.state

# 		for node_j in nodes: 
# 			if node_j is not node_i and np.linalg.norm(node_j.state[0:2] - node_i.state[0:2]) < node_i.r_sense: 
# 				if node_j.team_A: 
# 					o_a.append(node_j.state - node_i.state)
# 				elif node_j.team_B:
# 					o_b.append(node_j.state - node_i.state)

# 		if flatten:
# 			observations[node_i] = (np.array(o_a).flatten(),np.array(o_b).flatten(),np.array(goal).flatten())
# 		else:
# 			observations[node_i] = (o_a,o_b,goal)

# 	return observations

def relative_state(states,param,idx,flatten=False):

	n_robots, n_state_dim = states.shape

	goal = np.array([param.goal[0],param.goal[1],0,0])

	o_a = []
	o_b = [] 
	relative_goal = goal - states[idx,:]

	for idx_j in range(n_robots): 
		if idx_j != idx and np.linalg.norm(states[idx_j,0:2] - states[idx,0:2]) < param.robots[idx]["r_sense"]: 
			if idx_j in param.team_1_idxs:  
				o_a.append(states[idx_j,:] - states[idx,:])
			elif idx_j in param.team_2_idxs:
				o_b.append(states[idx_j,:] - states[idx,:])

	return np.array(o_a),np.array(o_b),np.array(relative_goal)

def relative_state_per_node(nodes,states,param,flatten=False):
	observations = dict()
	for node in nodes: 
		observations[node] = relative_state(states,param,node.idx,flatten=flatten)
	return observations