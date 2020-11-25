
import numpy as np 
import torch 


def global_to_local(states,param,idx):

	assert(np.isfinite(states[idx,:]).all())

	n_robots, n_state_dim = states.shape

	if param.dynamics["name"] == "double_integrator":
		pos_idx = np.arange(2)
		goal = np.array([param.goal[0],param.goal[1],0,0])
	elif param.dynamics["name"] == "single_integrator":
		pos_idx = np.arange(2)
		goal = np.array([param.goal[0],param.goal[1]])
	elif param.dynamics["name"] == "dubins_2d":
		pos_idx = np.arange(2)
		goal = np.array([param.goal[0],param.goal[1],0,0])
	else: 
		exit('learning interface dynamics not implemented')

	o_a = []
	o_b = [] 
	relative_goal = goal - states[idx,:]

	# projecting goal to radius of sensing 
	# alpha = np.linalg.norm(relative_goal[0:2]) / param.robots[idx]["r_sense"]
	# relative_goal[2:] = relative_goal[2:] / np.max((alpha,1))	

	for idx_j in range(n_robots):
		if idx_j != idx \
		and np.isfinite(states[idx_j,:]).all() \
		and np.linalg.norm(states[idx_j,pos_idx] - states[idx,pos_idx]) < param.robots[idx]["r_sense"]:
			if idx_j in param.team_1_idxs:  
				o_a.append(states[idx_j,:] - states[idx,:])
			elif idx_j in param.team_2_idxs:
				o_b.append(states[idx_j,:] - states[idx,:])

	return np.array(o_a),np.array(o_b),np.array(relative_goal)

def global_to_value(param,state): 

	# v_a = {s^j - g}
	# v_b = {s^j - g}
	# num-attackers, num-reached-goal, num-defenders

	n_robots, n_state_dim = state.shape

	if param.dynamics["name"] == "double_integrator":
		goal = np.array([param.goal[0],param.goal[1],0,0])
	elif param.dynamics["name"] == "single_integrator":
		goal = np.array([param.goal[0],param.goal[1]])
	elif param.dynamics["name"] == "dubins_2d":
		goal = np.array([param.goal[0],param.goal[1],0,0])
	else: 
		exit('learning interface dynamics not implemented')

	v_a = []
	v_b = [] 

	for idx_j in range(n_robots):
		if np.isfinite(state[idx_j,:]).all(): 
			if idx_j in param.team_1_idxs:  
				v_a.append(state[idx_j,:] - goal)
			elif idx_j in param.team_2_idxs:
				v_b.append(state[idx_j,:] - goal)

	return np.array(v_a),np.array(v_b)

def local_to_global(param,o_a,o_b,relative_goal,team):

	# relative_goal = {[g_x,g_y,0,0] - s^i} 
	# o_a = {s^j - s^i}
	# o_b = {s^j - s^i}

	if param.dynamics["name"] == "double_integrator":
		abs_goal = np.array([param.goal[0],param.goal[1],0,0])
	elif param.dynamics["name"] == "single_integrator":
		abs_goal = np.array([param.goal[0],param.goal[1]])
	elif param.dynamics["name"] == "dubins_2d":
		abs_goal = np.array([param.goal[0],param.goal[1],0,0])
	else: 
		exit('learning interface dynamics not implemented')

	# assume knowledge of self-team 
	if team == "a":
		i_a = 1
		i_b = 0 
	elif team == "b":
		i_a = 0 
		i_b = 1 

	# robot team composition 
	# assume knowledge of state_dim
	state_dim = param.dynamics["state_dim"]
	num_a = o_a.shape[0] + i_a
	num_b = o_b.shape[0] + i_b
	robot_team_composition = {
		'a': {'standard_robot': num_a},
		'b': {'standard_robot': num_b}
		}

	# state 
	s_i = abs_goal - relative_goal

	team_1_idxs = []
	team_2_idxs = [] 
	state = np.zeros((num_a + num_b,state_dim)) 
	for i in range(num_a + num_b):
		if team == "a" and i == 0:
			state[i,:] = s_i 
			self_idx = i 
			team_1_idxs.append(i)
		elif team == "b" and i == num_a: 
			state[i,:] = s_i
			self_idx = i 
			team_2_idxs.append(i)
		elif i < num_a:
			o_a_idx = i - i_a
			state[i,:] = o_a[o_a_idx,:] + s_i 
			team_1_idxs.append(i)
		else: 
			o_b_idx = i - i_b - num_a 
			state[i,:] = o_b[o_b_idx,:] + s_i 
			team_2_idxs.append(i)

	return state, robot_team_composition, self_idx, team_1_idxs, team_2_idxs

def format_data(o_a,o_b,goal):
	# input: [num_a/b, dim_state_a/b] np array 
	# output: 1 x something torch float tensor

	# make 0th dim (this matches batch dim in training)
	if o_a.shape[0] == 0:
		o_a = np.expand_dims(o_a,axis=0)
	if o_b.shape[0] == 0:
		o_b = np.expand_dims(o_b,axis=0)
	goal = np.expand_dims(goal,axis=0)

	# reshape if more than one element in set
	if o_a.shape[0] > 1: 
		o_a = np.reshape(o_a,(1,np.size(o_a)))
	if o_b.shape[0] > 1: 
		o_b = np.reshape(o_b,(1,np.size(o_b)))

	o_a = torch.from_numpy(o_a).float() 
	o_b = torch.from_numpy(o_b).float()
	goal = torch.from_numpy(goal).float()

	return o_a,o_b,goal

def format_data_value(v_a,v_b,n_a,n_b,n_rg):
	# input: [num_a/b, dim_state_a/b] np array 
	# output: 1 x something torch float tensor

	v_a = np.array(v_a,ndmin=2)
	v_b = np.array(v_b,ndmin=2)
	n_a = np.array(n_a,ndmin=2)
	n_b = np.array(n_b,ndmin=2)
	n_rg = np.array(n_rg,ndmin=2)

	# reshape if more than one element in set
	if v_a.shape[0] > 1: 
		v_a = np.reshape(v_a,(1,np.size(v_a)))
	if v_b.shape[0] > 1: 
		v_b = np.reshape(v_b,(1,np.size(v_b)))

	v_a = torch.from_numpy(v_a).float() 
	v_b = torch.from_numpy(v_b).float()
	n_a = torch.from_numpy(n_a).float()
	n_b = torch.from_numpy(n_b).float()
	n_rg = torch.from_numpy(n_rg).float()

	return v_a,v_b,n_a,n_b,n_rg
