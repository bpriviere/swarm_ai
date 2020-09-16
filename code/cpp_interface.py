
# standard 
import torch
import numpy as np
import tqdm
from collections import defaultdict
from multiprocessing import current_process

# ours
from cpp.buildRelease import mctscpp
from benchmark.panagou import PanagouPolicy
import datahandler as dh

from learning.continuous_emptynet import ContinuousEmptyNet

def create_cpp_robot_type(param, robot_type):
	p_min = [param.env_xlim[0], param.env_ylim[0]]
	p_max = [param.env_xlim[1], param.env_ylim[1]]
	velocity_limit = param.__dict__[robot_type]["speed_limit"]
	acceleration_limit = param.__dict__[robot_type]["acceleration_limit"]
	tag_radius = param.__dict__[robot_type]["tag_radius"]
	r_sense = param.__dict__[robot_type]["r_sense"]
	radius = param.__dict__[robot_type]["radius"]
	rt = mctscpp.RobotType(p_min,p_max,velocity_limit,acceleration_limit,tag_radius,r_sense,radius)
	return rt

def robot_composition_to_cpp_robot_types(param,team):
	types = [] 
	for robot_type, num in param.robot_team_composition[team].items():
		rt = create_cpp_robot_type(param, robot_type)
		for _ in range(num):
			types.append(rt)
	return types

def param_to_cpp_game(param):
	
	attackerTypes = robot_composition_to_cpp_robot_types(param,"a") 
	defenderTypes = robot_composition_to_cpp_robot_types(param,"b") 
	
	dt = param.sim_dt 

	goal = param.goal 
	max_depth = param.mcts_rollout_horizon

	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, max_depth)
	loadGLAS(g.glasA, param.path_glas_model_a)
	loadGLAS(g.glasB, param.path_glas_model_b)

	return g

def loadGLAS(glas, file):
	state_dict = torch.load(file)

	den = glas.discreteEmptyNet
	loadFeedForwardNNWeights(den.deepSetA.phi, state_dict, "model_team_a.phi")
	loadFeedForwardNNWeights(den.deepSetA.rho, state_dict, "model_team_a.rho")
	loadFeedForwardNNWeights(den.deepSetB.phi, state_dict, "model_team_b.phi")
	loadFeedForwardNNWeights(den.deepSetB.rho, state_dict, "model_team_b.rho")
	loadFeedForwardNNWeights(den.psi, state_dict, "psi")

	return glas

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

def cpp_state_to_pstate(gs):

	pstate = np.zeros((len(gs.attackers)+len(gs.defenders),4))
	idx = 0 
	for rs in gs.attackers:
		pstate[idx,:] = rs.state
		idx += 1 
	for rs in gs.defenders:
		pstate[idx,:] = rs.state
		idx += 1 		
	return pstate

def state_to_cpp_game_state(param,state,turn):

	param.state = state
	param.make_robot_teams()
	param.assign_initial_condition()

	attackers = [] 
	defenders = [] 
	for robot in param.robots: 
		rs = mctscpp.RobotState(robot["x0"])
		if robot["team"] == "a":
			attackers.append(rs)
		elif robot["team"] == "b":
			defenders.append(rs)		

	if turn == "a":
		turn = mctscpp.GameState.Turn.Attackers
	elif turn == "b": 
		turn = mctscpp.GameState.Turn.Defenders
	else: 
		exit('training team not recognized') 

	game_state = mctscpp.GameState(turn,attackers,defenders)

	game_state.depth = 0

	return game_state 

def expected_value(param):
	g = param_to_cpp_game(param) 
	state = np.array(param.state)
	gs = state_to_cpp_game_state(param,state,"a")
	mctsresult = mctscpp.search(g, gs,
		param.mcts_tree_size, param.mcts_rollout_beta, param.mcts_c_param,
		param.mcts_pw_C, param.mcts_pw_alpha)
	return mctsresult.expectedReward

def self_play(param,deterministic=True):

	param.policy_a_dict = {
		"sim_mode" : param.sim_mode, 
	} 
	param.policy_b_dict = {
		"sim_mode" : param.sim_mode, 
	} 

	if param.sim_mode in ["GLAS","MCTS_RANDOM","MCTS_GLAS"]:
		param.policy_a_dict["path_glas_model_a"] = param.path_glas_model_a
		param.policy_b_dict["path_glas_model_b"] = param.path_glas_model_b
	if param.sim_mode in ["MCTS_RANDOM","MCTS_GLAS"]:
		param.policy_a_dict["mcts_tree_size"] = param.mcts_tree_size
		param.policy_a_dict["mcts_rollout_beta"] = param.mcts_rollout_beta
		param.policy_a_dict["mcts_c_param"] = param.mcts_c_param
		param.policy_a_dict["mcts_pw_C"] = param.mcts_pw_C
		param.policy_a_dict["mcts_pw_alpha"] = param.mcts_pw_alpha
		
		param.policy_b_dict["mcts_tree_size"] = param.mcts_tree_size
		param.policy_b_dict["mcts_rollout_beta"] = param.mcts_rollout_beta
		param.policy_b_dict["mcts_c_param"] = param.mcts_c_param
		param.policy_b_dict["mcts_pw_C"] = param.mcts_pw_C
		param.policy_b_dict["mcts_pw_alpha"] = param.mcts_pw_alpha

	return play_game(param,deterministic=deterministic)

def relative_state(states,param,idx):

	n_robots, n_state_dim = states.shape

	goal = np.array([param.goal[0],param.goal[1],0,0])

	o_a = []
	o_b = [] 
	relative_goal = goal - states[idx,:]

	# projecting goal to radius of sensing 
	alpha = np.linalg.norm(relative_goal[0:2]) / param.robots[idx]["r_sense"]
	relative_goal[2:] = relative_goal[2:] / np.max((alpha,1))	

	for idx_j in range(n_robots): 
		if idx_j != idx and np.linalg.norm(states[idx_j,0:2] - states[idx,0:2]) < param.robots[idx]["r_sense"]: 
			if idx_j in param.team_1_idxs:  
				o_a.append(states[idx_j,:] - states[idx,:])
			elif idx_j in param.team_2_idxs:
				o_b.append(states[idx_j,:] - states[idx,:])

	return np.array(o_a),np.array(o_b),np.array(relative_goal)


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


def play_game(param,deterministic=True): 

	if param.policy_a_dict["sim_mode"] in ["GLAS" , "MCTS_GLAS" , "MCTS_RANDOM"]:
		param.path_glas_model_a = param.policy_a_dict["path_glas_model_a"]

	if param.policy_b_dict["sim_mode"] in ["GLAS" , "MCTS_GLAS" , "MCTS_RANDOM"]:
		param.path_glas_model_b = param.policy_b_dict["path_glas_model_b"]

	if param.policy_a_dict["sim_mode"] == "PANAGOU" or param.policy_b_dict["sim_mode"] == "PANAGOU":
		pp = PanagouPolicy(param)
		pp.init_sim(param.state)

	# similar to rollout 
	g = param_to_cpp_game(param) 

	results = []
	actions = [] 

	state = param.state
	gs = state_to_cpp_game_state(param,state,"a")
	count = 0 
	while True:
		gs.depth = 0

		if gs.turn == mctscpp.GameState.Turn.Attackers:
			action = [] 
			policy_dict = param.policy_a_dict
			idxs = param.team_1_idxs
		elif gs.turn == mctscpp.GameState.Turn.Defenders:
			policy_dict = param.policy_b_dict
			idxs = param.team_2_idxs 

		if "MCTS" in policy_dict["sim_mode"]:
			
			mctsresult = mctscpp.search(g, gs, \
				policy_dict["mcts_tree_size"],
				policy_dict["mcts_rollout_beta"],
				policy_dict["mcts_c_param"],
				policy_dict["mcts_pw_C"],
				policy_dict["mcts_pw_alpha"])
			if mctsresult.success: 
				team_action = mctsresult.bestAction
				success = g.step(gs, team_action, gs)
			else:
				success = False		

		elif policy_dict["sim_mode"] == "GLAS":

			# action = mctscpp.eval(g, gs, deterministic)
			
			# temp 
			# use python to eval model 
			state = cpp_state_to_pstate(gs)
			action = np.nan*np.ones((param.num_nodes,2))
			model = ContinuousEmptyNet(param, "cpu")
			if gs.turn == mctscpp.GameState.Turn.Attackers: 
				model.load_state_dict(torch.load(param.path_glas_model_a))
			else: 
				model.load_state_dict(torch.load(param.path_glas_model_b))

			for robot_idx in idxs: 
				o_a,o_b,goal = relative_state(state,param,robot_idx)
				o_a,o_b,goal = format_data(o_a,o_b,goal)
				value, policy = model(o_a,o_b,goal)
				action[robot_idx, :] = policy.detach().numpy().flatten()

			success = g.step(gs, action, gs)

		elif policy_dict["sim_mode"] == "PANAGOU":

			pstate = cpp_state_to_pstate(gs)
			pp.init_sim(pstate)
			action = pp.eval(pstate)
			success = g.step(gs, action, gs)

		else: 
			exit('sim mode {} not recognized'.format(policy_dict["sim_mode"]))


		# for idx in idxs: 
		# 	action.append(team_action[idx])
		
		results.append(game_state_to_cpp_result(g,gs,None))
		count += 1 

		if success:
			if g.isTerminal(gs):
				results.append(game_state_to_cpp_result(g,gs,None))
				break 
		else:
			break

	if len(results) == 0:
		results.append(game_state_to_cpp_result(g,gs,None))
		
	sim_result = dh.convert_cpp_data_to_sim_result(np.array(results),param)
	return sim_result


def game_state_to_cpp_result(g,gs,action):

	if action is None:
		action = np.nan*np.ones((len(gs.attackers) + len(gs.defenders),2))
		
	idx = 0 
	result = np.empty((len(gs.attackers) + len(gs.defenders))*6+2, dtype=np.float32)
	for rs in gs.attackers:
		result[idx*6+0:idx*6+4] = rs.state
		result[idx*6+4:idx*6+6] = action[idx]
		idx += 1
	for rs in gs.defenders:
		result[idx*6+0:idx*6+4] = rs.state
		result[idx*6+4:idx*6+6] = action[idx]
		idx += 1
	result[idx*6+0] = g.computeReward(gs)
	result[idx*6+1] = 1 - result[idx*6+0]
	return result

def evaluate_expert(states,param,quiet_on=True,progress=None):
	
	if not quiet_on:
		print('   running expert for instance {}'.format(param.dataset_fn))

	if progress is not None:
		progress_pos = current_process()._identity[0] - progress
		enumeration = tqdm.tqdm(states, desc=param.dataset_fn, leave=False, position=progress_pos)
	else:
		enumeration = states

	game = param_to_cpp_game(param)

	sim_result = {
		'states' : [],
		'policy_dists' : [],
		'values' : [],
		'param' : param.to_dict()
		}

	for state in enumeration:
		game_state = state_to_cpp_game_state(param,state,param.training_team)
		mctsresult = mctscpp.search(game, game_state,
			param.mcts_tree_size, param.mcts_rollout_beta, param.mcts_c_param,
			param.mcts_pw_C, param.mcts_pw_alpha)
		if mctsresult.success: 
			policy_dist = valuePerAction_to_policy_dist(param,mctsresult.valuePerAction) # 
			value = mctsresult.expectedReward[0]
			sim_result["states"].append(state) # total number of robots x state dimension per robot 
			sim_result["policy_dists"].append(policy_dist)  
			sim_result["values"].append(value)

	sim_result["states"] = np.array(sim_result["states"])
	sim_result["policy_dists"] = np.array(sim_result["policy_dists"])
	dh.write_sim_result(sim_result,param.dataset_fn)

	if not quiet_on:
		print('   completed instance {} with {} dp.'.format(param.dataset_fn,sim_result["states"].shape[0]))
	

def valuePerAction_to_policy_dist(param,valuePerAction):

	if param.training_team == "a":
		num_robots = param.num_nodes_A
		robot_idxs = param.team_1_idxs
	elif param.training_team == "b":
		num_robots = param.num_nodes_B
		robot_idxs = param.team_2_idxs

	dist = defaultdict(list)
	for action,value in valuePerAction:

		action = np.array(action)
		action = action.flatten()

		for robot_idx in robot_idxs:
			action_idx = robot_idx * 2 + np.arange(2)
			dist[robot_idx].append([action[action_idx],value])

	for key in dist.keys():
		dist[key] = np.array(dist[key])

	return dist	


