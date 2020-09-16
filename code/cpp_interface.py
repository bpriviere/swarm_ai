
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

def param_to_cpp_game(param, policy_dict_a, policy_dict_b):
	
	attackerTypes = robot_composition_to_cpp_robot_types(param,"a") 
	defenderTypes = robot_composition_to_cpp_robot_types(param,"b") 
	
	dt = param.sim_dt 

	goal = param.goal 

	# ideally, max_depth should be a policy parameter, not a game parameter 
	if policy_dict_a["sim_mode"] == "MCTS":
		max_depth = policy_dict_a["mcts_rollout_horizon"]
	elif policy_dict_b["sim_mode"] == "MCTS":
		max_depth = policy_dict_b["mcts_rollout_horizon"]
	else: 
		max_depth = param.df_mcts_rollout_horizon

	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, max_depth)
	
	if policy_dict_a["sim_mode"] in ["GLAS","MCTS"]:
		loadGLAS(g.glasA, policy_dict_a["path_glas_model_a"])

	if policy_dict_b["sim_mode"] in ["GLAS","MCTS"]:
		loadGLAS(g.glasB, policy_dict_b["path_glas_model_b"])

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

def expected_value(param,state,policy_dict):
	g = param_to_cpp_game(param,policy_dict,policy_dict) 
	gs = state_to_cpp_game_state(param,state,"a")
	mctsresult = mctscpp.search(g, gs,
		policy_dict["mcts_tree_size"],
		policy_dict["mcts_rollout_beta"],
		policy_dict["mcts_c_param"],
		policy_dict["mcts_pw_C"],
		policy_dict["mcts_pw_alpha"])
	return mctsresult.expectedReward

def self_play(param,deterministic=True):

	if is_valid_policy_dict(param.policy_dict):
		policy_dict_a = param.policy_dict
		policy_dict_b = param.policy_dict 
	else: 
		print('bad policy dict')
		exit()

	return play_game(param,policy_dict_a,policy_dict_b,deterministic=deterministic)

def play_game(param,policy_dict_a,policy_dict_b,deterministic=True): 

	if policy_dict_a["sim_mode"] == "PANAGOU" or policy_dict_b["sim_mode"] == "PANAGOU":
		pp = PanagouPolicy(param)
		pp.init_sim(param.state)

	g = param_to_cpp_game(param,policy_dict_a,policy_dict_b) 

	sim_result = {
		'param' : param.to_dict(),
		'states' : [],
		'actions' : [],
		'times' : None,
		'rewards' : []
	}

	gs = state_to_cpp_game_state(param,param.state,"a")
	count = 0
	finished = False
	invalid_team_action = [np.nan*np.ones(2) for _ in range(param.num_nodes)]
	team_action = list(invalid_team_action)
	while not finished:
		gs.depth = 0

		if gs.turn == mctscpp.GameState.Turn.Attackers:
			policy_dict = policy_dict_a
			team_idx = param.team_1_idxs
		elif gs.turn == mctscpp.GameState.Turn.Defenders:
			policy_dict = policy_dict_b
			team_idx = param.team_2_idxs

		if policy_dict["sim_mode"] == "MCTS":
			
			mctsresult = mctscpp.search(g, gs, \
				policy_dict["mcts_tree_size"],
				policy_dict["mcts_rollout_beta"],
				policy_dict["mcts_c_param"],
				policy_dict["mcts_pw_C"],
				policy_dict["mcts_pw_alpha"])
			if mctsresult.success: 
				action = mctsresult.bestAction
				success = g.step(gs, action, gs)
			else:
				success = False

		elif policy_dict["sim_mode"] == "GLAS":

			action = mctscpp.eval(g, gs, deterministic)
			success = g.step(gs, action, gs)

		elif policy_dict["sim_mode"] == "PANAGOU":

			pstate = cpp_state_to_pstate(gs)
			pp.init_sim(pstate)
			action = pp.eval(pstate)
			success = g.step(gs, action, gs)

		else: 
			exit('sim mode {} not recognized'.format(policy_dict["sim_mode"]))

		isTerminal = g.isTerminal(gs)
		if success:
			if count % 2 == 0 or isTerminal:
				# update sim_result
				sim_result['states'].append([rs.state.copy() for rs in gs.attackers + gs.defenders])
				sim_result['actions'].append(team_action.copy())
				r = g.computeReward(gs)
				sim_result['rewards'].append([r, 1 - r])
				# prepare for next update
				team_action = invalid_team_action

		finished = not success or isTerminal

		for idx in team_idx:
			team_action[idx] = action[idx].copy()

		count += 1

	# delete the first (nan) action pair and add it at the end.
	# This way the action is the action that needs to be taken to reach the next state
	# and action and state arrays will have the same dimension
	del sim_result['actions'][0]
	sim_result['actions'].append(invalid_team_action)
	# convert sim_result to numpy arrays
	sim_result['states'] = np.array(sim_result['states'], dtype=np.float32)
	sim_result['actions'] = np.array(sim_result['actions'], dtype=np.float32)
	sim_result['times'] = param.sim_dt*np.arange(sim_result['states'].shape[0])
	sim_result['rewards'] = np.array(sim_result['rewards'], dtype=np.float32)

	return sim_result


def evaluate_expert(states,param,quiet_on=True,progress=None):
	
	if not quiet_on:
		print('   running expert for instance {}'.format(param.dataset_fn))

	if progress is not None:
		progress_pos = current_process()._identity[0] - progress
		enumeration = tqdm.tqdm(states, desc=param.dataset_fn, leave=False, position=progress_pos)
	else:
		enumeration = states

	if is_valid_policy_dict(param.policy_dict):
		policy_dict = param.policy_dict
	else: 
		print('bad policy dict')
		exit()

	game = param_to_cpp_game(param,policy_dict,policy_dict)

	sim_result = {
		'states' : [],
		'actions' : [],
		'values' : [],
		'param' : param.to_dict()
		}

	for state in enumeration:
		game_state = state_to_cpp_game_state(param,state,param.training_team)
		mctsresult = mctscpp.search(game, game_state,
			policy_dict["mcts_tree_size"],
			policy_dict["mcts_rollout_beta"],
			policy_dict["mcts_c_param"],
			policy_dict["mcts_pw_C"],
			policy_dict["mcts_pw_alpha"])
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

def bad_key(some_dict,some_key):
	if some_key not in some_dict.keys():
		print('no {} specified'.format(some_key))
		return True
	return False

def is_valid_policy_dict(policy_dict):

	if policy_dict["sim_mode"] == "GLAS":
		if bad_key(policy_dict,"path_glas_model_a") or \
			bad_key(policy_dict,"path_glas_model_b"):
			return False 

	elif policy_dict["sim_mode"] == "MCTS":
		if bad_key(policy_dict,"path_glas_model_a") or \
			bad_key(policy_dict,"path_glas_model_b") or \
			bad_key(policy_dict,"mcts_tree_size") or \
			bad_key(policy_dict,"mcts_rollout_horizon") or \
			bad_key(policy_dict,"mcts_rollout_beta") or \
			bad_key(policy_dict,"mcts_c_param") or \
			bad_key(policy_dict,"mcts_pw_C") or \
			bad_key(policy_dict,"mcts_pw_alpha"): 
			return False

	elif policy_dict["sim_mode"] == "PANAGOU":
		pass 

	else: 
		print('sim_mode not recognized')
		return False

	return True 	