
# standard 
import torch
import numpy as np
import tqdm
from collections import defaultdict
from multiprocessing import current_process

# ours
# from learning.discrete_emptynet import DiscreteEmptyNet
# from mice import format_data, relative_state
from cpp.buildRelease import mctscpp
from benchmark.panagou import PanagouPolicy
import datahandler as dh

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

def play_game(param,deterministic=True): 

	if param.policy_a_dict["sim_mode"] in ["GLAS" , "MCTS_GLAS" , "MCTS_RANDOM"]:
		param.path_glas_model_a = param.policy_a_dict["path_glas_model_a"]
		# glas_model_a = DiscreteEmptyNet(param, "cpu")
		# glas_model_a.load_state_dict(torch.load(param.path_glas_model_a))

	if param.policy_b_dict["sim_mode"] in ["GLAS" , "MCTS_GLAS" , "MCTS_RANDOM"]:
		param.path_glas_model_b = param.policy_b_dict["path_glas_model_b"]
		# glas_model_b = DiscreteEmptyNet(param, "cpu")
		# glas_model_b.load_state_dict(torch.load(param.path_glas_model_b))		

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
		# gs.attackersReward = 0
		# gs.defendersReward = 0
		gs.depth = 0

		if gs.turn == mctscpp.GameState.Turn.Attackers:
			action = [] 
			policy_dict = param.policy_a_dict
			idxs = param.team_1_idxs
			# if param.policy_a_dict["sim_mode"] in ["GLAS" , "MCTS_GLAS" , "MCTS_RANDOM"]:
			# 	model = glas_model_a 

		elif gs.turn == mctscpp.GameState.Turn.Defenders:
			policy_dict = param.policy_b_dict
			idxs = param.team_2_idxs 
			# if param.policy_b_dict["sim_mode"] in ["GLAS" , "MCTS_GLAS" , "MCTS_RANDOM"]:
			# 	model = glas_model_b 

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

			action = mctscpp.eval(g, gs, deterministic)

			# pstate = cpp_state_to_pstate(gs)
			# action = np.zeros((len(param.robots),2))
			# for idx in idxs: 
			# 	o_a,o_b,goal = relative_state(state,param,idx)
			# 	o_a,o_b,goal = format_data(o_a,o_b,goal)
			# 	value,action[idx,:] = model(o_a,o_b,goal)

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
		'actions' : [],
		'values' : [],
		'param' : param.to_dict()
		}

	for state in enumeration:
		game_state = state_to_cpp_game_state(param,state,param.training_team)
		mctsresult = mctscpp.search(game, game_state,
			param.mcts_tree_size, param.mcts_rollout_beta, param.mcts_c_param,
			param.mcts_pw_C, param.mcts_pw_alpha)
		if mctsresult.success: 
			action = value_to_dist(param,mctsresult.valuePerAction) # 
			value = mctsresult.expectedReward[0]
			sim_result["states"].append(state) # total number of robots x state dimension per robot 
			sim_result["actions"].append(action) # total number of robots x action dimension per robot 
			sim_result["values"].append(value)

	sim_result["states"] = np.array(sim_result["states"])
	sim_result["actions"] = np.array(sim_result["actions"])
	dh.write_sim_result(sim_result,param.dataset_fn)

	if not quiet_on:
		print('   completed instance {} with {} dp.'.format(param.dataset_fn,sim_result["states"].shape[0]))
	

def value_to_dist(param,valuePerAction):

	if param.training_team == "a":
		num_robots = param.num_nodes_A
		robot_idxs = param.team_1_idxs
	elif param.training_team == "b":
		num_robots = param.num_nodes_B
		robot_idxs = param.team_2_idxs

	dist = np.zeros((param.num_nodes,9))
	values = defaultdict(list) 
	for value_action,value in valuePerAction:

		value_action = np.array(value_action)
		value_action = value_action.flatten()

		for robot_idx in robot_idxs:

			class_action = np.zeros(2)
			if value_action[robot_idx*2] > 0: 
				class_action[0] = 1 
			elif value_action[robot_idx*2] < 0: 
				class_action[0] = -1 
			if value_action[robot_idx*2+1] > 0: 
				class_action[1] = 1 
			elif value_action[robot_idx*2+1] < 0: 
				class_action[1] = -1 

			action_idx = np.where(np.all(param.actions == class_action,axis=1))[0][0]
			values[robot_idx,action_idx].append(value) 

	for robot_idx in robot_idxs: 
		for action_idx, robot_action in enumerate(param.actions):
			dist[robot_idx,action_idx] = np.sum(values[robot_idx,action_idx])

	return dist	