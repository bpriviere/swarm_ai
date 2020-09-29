
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

def param_to_cpp_game(param, policy_dict_a, policy_dict_b):
	attackerTypes = robot_composition_to_cpp_robot_types(param,"a") 
	defenderTypes = robot_composition_to_cpp_robot_types(param,"b") 

	dt = param.sim_dt
	goal = param.goal
	rollout_horizon = param.rollout_horizon

	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, rollout_horizon)
	return g

def create_cpp_policy(policy_dict, team):
	policy = mctscpp.Policy()
	if policy_dict["sim_mode"] in ["GLAS","MCTS"]:
		file = policy_dict["path_glas_model_{}".format(team)]
		if file is not None:
			loadGLAS(policy.glas, file)
			policy.rolloutBeta = policy_dict["mcts_rollout_beta"]
			return policy
	policy.rolloutBeta = 0.0
	return policy

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
		policy_dict["mcts_pw_alpha"],
		policy_dict["mcts_vf_beta"])
	return mctsresult.expectedReward

def self_play(param,deterministic=True):

	if is_valid_policy_dict(param.policy_dict):
		policy_dict_a = param.policy_dict
		policy_dict_b = param.policy_dict 
	else: 
		print('bad policy dict')
		exit()

	return play_game(param,policy_dict_a,policy_dict_b,deterministic=deterministic)

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


def play_game(param,policy_dict_a,policy_dict_b,deterministic=True): 

	if policy_dict_a["sim_mode"] == "PANAGOU" or policy_dict_b["sim_mode"] == "PANAGOU":
		pp = PanagouPolicy(param)
		pp.init_sim(param.state)

	g = param_to_cpp_game(param,policy_dict_a,policy_dict_b)
	policy_a = create_cpp_policy(policy_dict_a, 'a')
	policy_b = create_cpp_policy(policy_dict_b, 'b')

	sim_result = {
		'param' : param.to_dict(),
		'states' : [],
		'actions' : [],
		'times' : None,
		'rewards' : []
	}

	gs = state_to_cpp_game_state(param,param.state,"a")
	count = 0
	invalid_team_action = [np.nan*np.ones(2) for _ in range(param.num_nodes)]
	team_action = list(invalid_team_action)
	while True:
		gs.depth = 0

		if gs.turn == mctscpp.GameState.Turn.Attackers:
			policy_dict = policy_dict_a
			team_idx = param.team_1_idxs
			my_policy = policy_a
			other_policies = [policy_b]
		elif gs.turn == mctscpp.GameState.Turn.Defenders:
			policy_dict = policy_dict_b
			team_idx = param.team_2_idxs
			my_policy = policy_b
			other_policies = [policy_a]

		# output result
		isTerminal = g.isTerminal(gs)
		if count % 2 == 0 or isTerminal:
			# update sim_result
			sim_result['states'].append([rs.state.copy() for rs in gs.attackers + gs.defenders])
			sim_result['actions'].append(team_action.copy())
			r = g.computeReward(gs)
			sim_result['rewards'].append([r, 1 - r])
			# prepare for next update
			team_action = list(invalid_team_action)

		if isTerminal:
			break

		if policy_dict["sim_mode"] == "MCTS":
			
			mctsresult = mctscpp.search(g, gs, \
				my_policy,
				other_policies,
				policy_dict["mcts_tree_size"],
				policy_dict["mcts_c_param"],
				policy_dict["mcts_pw_C"],
				policy_dict["mcts_pw_alpha"],
				policy_dict["mcts_vf_beta"])
			if mctsresult.success: 
				action = mctsresult.bestAction
				success = g.step(gs, action, gs)
			else:
				success = False

		elif policy_dict["sim_mode"] == "GLAS":
			policy_a.rolloutBeta = 1.0
			policy_b.rolloutBeta = 1.0
			action = mctscpp.eval(g, gs, policy_a, policy_b, deterministic)

			# testing 
			# deterministic = False
			# num_samples = 1000

			# state = cpp_state_to_pstate(gs)
			# model = ContinuousEmptyNet(param, "cpu")
			# if gs.turn == mctscpp.GameState.Turn.Attackers: 
			# 	model.load_state_dict(torch.load(policy_dict["path_glas_model_a"]))
			# else: 
			# 	model.load_state_dict(torch.load(policy_dict["path_glas_model_b"]))

			# cpp_actions = []
			# python_actions = [] 
			# for i_sample in range(num_samples):

			# 	cpp_action = mctscpp.eval(g, gs, deterministic)
			# 	cpp_actions.append(cpp_action)

			# 	python_action = np.nan*np.ones((param.num_nodes,2))
			# 	for robot_idx in range(param.num_nodes): 
			# 		o_a,o_b,goal = relative_state(state,param,robot_idx)
			# 		o_a,o_b,goal = format_data(o_a,o_b,goal)					
			# 		value, policy = model(o_a,o_b,goal)
			# 		python_action[robot_idx, :] = policy.detach().numpy().flatten()

			# 	python_actions.append(python_action)

			# action = python_action

			# print('cpp_action',cpp_action)
			# print('python_action',python_action)

			# cpp_actions = np.array(cpp_actions)
			# python_actions = np.array(python_actions)

			# import matplotlib.pyplot as plt 
			# fig, axs = plt.subplots(1,2,sharex=True,sharey=True)
			# axs[0].scatter(cpp_actions[:,0,0], cpp_actions[:,0,1], alpha=0.5)
			# axs[0].set_title('cpp')
			# axs[1].scatter(python_actions[:,0,0], python_actions[:,0,1], alpha=0.5)
			# axs[1].set_title('python')

			# plt.show()
			# exit()

			# print('cpp', action)

			# # use python to eval model 
			# state = cpp_state_to_pstate(gs)
			# action = np.nan*np.ones((param.num_nodes,2))
			# model = ContinuousEmptyNet(param, "cpu")
			# if gs.turn == mctscpp.GameState.Turn.Attackers: 
			# 	model.load_state_dict(torch.load(policy_dict["path_glas_model_a"]))
			# else: 
			# 	model.load_state_dict(torch.load(policy_dict["path_glas_model_b"]))

			# for robot_idx in team_idx: 
			# 	o_a,o_b,goal = relative_state(state,param,robot_idx)
			# 	o_a,o_b,goal = format_data(o_a,o_b,goal)
			# 	value, policy = model(o_a,o_b,goal)
			# 	action[robot_idx, :] = policy.detach().numpy().flatten()

			# print('python', action)


			success = g.step(gs, action, gs)

		elif policy_dict["sim_mode"] == "PANAGOU":

			pstate = cpp_state_to_pstate(gs)
			pp.init_sim(pstate)
			action = pp.eval(pstate)
			success = g.step(gs, action, gs)

		elif policy_dict["sim_mode"] == "RANDOM":
			policy_a.rolloutBeta = 0.0
			policy_b.rolloutBeta = 0.0
			action = mctscpp.eval(g, gs, policy_a, policy_b, False)
			success = g.step(gs, action, gs)

		else: 
			exit('sim mode {} not recognized'.format(policy_dict["sim_mode"]))

		if not success:
			break

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

	game = param_to_cpp_game(param,None,None)

	my_policy = create_cpp_policy(param.my_policy_dict, param.training_team)

	other_team = "b" if param.training_team == "a" else "a"
	other_policies = []
	for other_policy_dict in param.other_policy_dicts:
		other_policies.append(create_cpp_policy(other_policy_dict,other_team))

	sim_result = {
		'states' : [],
		'policy_dists' : [],
		'values' : [],
		'param' : param.to_dict()
		}

	for state in enumeration:
		game_state = state_to_cpp_game_state(param,state,param.training_team)
		mctsresult = mctscpp.search(game, game_state, \
			my_policy,
			other_policies,
			param.my_policy_dict["mcts_tree_size"],
			param.my_policy_dict["mcts_c_param"],
			param.my_policy_dict["mcts_pw_C"],
			param.my_policy_dict["mcts_pw_alpha"],
			param.my_policy_dict["mcts_vf_beta"])
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

def test_evaluate_expert(states,param,testing,quiet_on=True,progress=None):
	
	alphas = np.random.randint(low=0,high=len(testing),size=len(states))
	
	if not quiet_on:
		print('   running expert for instance {}'.format(param.dataset_fn))

	if progress is not None:
		progress_pos = current_process()._identity[0] - progress
		# enumeration = tqdm.tqdm(states, desc=param.dataset_fn, leave=False, position=progress_pos)
		enumeration = tqdm.tqdm(zip(alphas,states), desc=param.dataset_fn, leave=False, position=progress_pos)
	else:
		enumeration = zip(alphas,states)

	if is_valid_policy_dict(param.policy_dict):
		policy_dict = param.policy_dict
	else: 
		print('bad policy dict')
		exit()

	game = param_to_cpp_game(param,policy_dict,policy_dict)

	sim_result = {
		'states' : [],
		'policy_dists' : [],
		'values' : [],
		'param' : param.to_dict()
		}

	for alpha, state in enumeration:

		test_state = np.array(testing[alpha]["test_state"])
		test_value = testing[alpha]["test_value"]

		test_valuePerAction = []
		for valuePerAction in testing[alpha]["test_valuePerAction"]:
			x = ((np.array((valuePerAction[0],valuePerAction[1]),dtype=float),np.array((np.nan,np.nan),dtype=float)),valuePerAction[-1])
			test_valuePerAction.append(x)

		test_policy_dist = valuePerAction_to_policy_dist(param,test_valuePerAction) # 

		sim_result["states"].append(test_state)
		sim_result["policy_dists"].append(test_policy_dist)  		
		sim_result["values"].append(test_value)

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


	if param.l_subsample_on: 

		# make distribution
		weights = np.array([v for _,v in valuePerAction])
		weights /= sum(weights)
		dist = dict()
		for robot_idx in robot_idxs: 
			action_idx = robot_idx * 2 + np.arange(2)
			actions = np.array([np.array(a).flatten()[action_idx] for a,v in valuePerAction])
			choice_idxs = np.random.choice(actions.shape[0],param.l_num_subsamples,p=weights)
			dist[robot_idx] = np.array([(actions[choice_idx,:],weights[choice_idx]) for choice_idx in choice_idxs])

	else: 

		dist = defaultdict(list)

		actions = [a for a,_ in valuePerAction]
		values = [v for _,v in valuePerAction]

		actions = [x for _,x in sorted(zip(values,actions), key=lambda pair: -pair[0])]
		values = sorted(values)

		if len(actions) > param.l_num_samples:
			actions = actions[0:param.l_num_samples]
			values = values[0:param.l_num_samples]

		values = [v/sum(values) for v in values] 

		for action,value in zip(actions,values):

			action = np.array(action)
			action = action.flatten()

			for robot_idx in robot_idxs:
				action_idx = robot_idx * 2 + np.arange(2)
				# dist[robot_idx].append(np.array([action[action_idx],value]))
				dist[robot_idx].append([action[action_idx],value])

		# print(dist)
		# exit()

		for robot_idx in dist.keys():
			dist[robot_idx] = np.array(dist[robot_idx])

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
			bad_key(policy_dict,"mcts_rollout_beta") or \
			bad_key(policy_dict,"mcts_c_param") or \
			bad_key(policy_dict,"mcts_pw_C") or \
			bad_key(policy_dict,"mcts_pw_alpha"): 
			return False

	elif policy_dict["sim_mode"] == "PANAGOU":
		pass

	elif policy_dict["sim_mode"] == "RANDOM":
		pass

	else:
		print('sim_mode not recognized')
		return False

	return True