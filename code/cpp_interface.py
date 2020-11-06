
# standard 
import torch
import numpy as np
import tqdm
import os 
from queue import Empty
from collections import defaultdict
from multiprocessing import current_process

# ours
from cpp.buildRelease import mctscpp
from benchmark.panagou import PanagouPolicy
import datahandler as dh

from learning.continuous_emptynet import ContinuousEmptyNet
from learning_interface import local_to_global, global_to_local

def create_cpp_robot_type(robot_type, env_xlim, env_ylim):
	p_min = [env_xlim[0], env_ylim[0]]
	p_max = [env_xlim[1], env_ylim[1]]
	velocity_limit = robot_type["speed_limit"]
	acceleration_limit = robot_type["acceleration_limit"]
	tag_radius = robot_type["tag_radius"]
	r_sense = robot_type["r_sense"]
	radius = robot_type["radius"]
	rt = mctscpp.RobotType(p_min,p_max,velocity_limit,acceleration_limit,tag_radius,r_sense,radius)
	return rt	


def robot_composition_to_cpp_robot_types(robot_team_composition,robot_types,team,env_xlim,env_ylim):
	types = [] 
	for robot_type_name, num in robot_team_composition[team].items():
		rt = create_cpp_robot_type(robot_types[robot_type_name], env_xlim, env_ylim)
		for _ in range(num):
			types.append(rt)
	return types


def param_to_cpp_game(robot_team_composition,robot_types,env_xlim,env_ylim,dt,goal,rollout_horizon):
	attackerTypes = robot_composition_to_cpp_robot_types(robot_team_composition,robot_types,"a",env_xlim,env_ylim)
	defenderTypes = robot_composition_to_cpp_robot_types(robot_team_composition,robot_types,"b",env_xlim,env_ylim)
	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, rollout_horizon)
	return g

def create_cpp_policy(policy_dict, team):
	policy = mctscpp.Policy('None')
	if policy_dict["sim_mode"] in ["MCTS","D_MCTS"]:
		file = policy_dict["path_glas_model_{}".format(team)]
		if file is not None:
			loadPolicy(policy.glas, file)
			policy.name = file
			policy.beta2 = policy_dict["mcts_beta2"]
			return policy
	elif policy_dict["sim_mode"] in ["GLAS"]:
		file = policy_dict["path_glas_model"]
		loadPolicy(policy.glas, file)
		policy.name = file
		policy.beta2 = 1.0
		return policy

	policy.beta2 = 0.0
	return policy

def create_cpp_value(file):
	value = mctscpp.ValuePredictor('None')
	if file is not None: 
		loadValue(value,file)
	return value 

def loadValue(value,file):
	state_dict = torch.load(file)

	loadFeedForwardNNWeights(value.deepSetA.phi, state_dict, "model_team_a.phi")
	loadFeedForwardNNWeights(value.deepSetA.rho, state_dict, "model_team_a.rho")
	loadFeedForwardNNWeights(value.deepSetB.phi, state_dict, "model_team_b.phi")
	loadFeedForwardNNWeights(value.deepSetB.rho, state_dict, "model_team_b.rho")
	loadFeedForwardNNWeights(value.value, state_dict, "value")

	return value 

def loadPolicy(policy, file):
	state_dict = torch.load(file)

	loadFeedForwardNNWeights(policy.deepSetA.phi, state_dict, "model_team_a.phi")
	loadFeedForwardNNWeights(policy.deepSetA.rho, state_dict, "model_team_a.rho")
	loadFeedForwardNNWeights(policy.deepSetB.phi, state_dict, "model_team_b.phi")
	loadFeedForwardNNWeights(policy.deepSetB.rho, state_dict, "model_team_b.rho")
	loadFeedForwardNNWeights(policy.policy, state_dict, "policy")

	return policy	

def loadFeedForwardNNWeights(ff, state_dict, name):
	l = 0
	while True:
		key1 = "{}.layers.{}.weight".format(name, l)
		key2 = "{}.layers.{}.bias".format(name, l)
		if key1 in state_dict and key2 in state_dict:
			ff.addLayer(state_dict[key1].numpy(), state_dict[key2].numpy())
		else:
			# if l == 0:
				# print("WARNING: No weights found for {}".format(name))
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

def state_to_cpp_game_state(state,turn,team_1_idxs,team_2_idxs):

	attackers = [] 
	defenders = [] 
	for robot_idx, robot_state in enumerate(state): 
		rs = mctscpp.RobotState(robot_state)
		if np.isnan(robot_state).any():
			rs.status = mctscpp.RobotState.Status.Invalid
		elif np.isposinf(robot_state).any():
			rs.status = mctscpp.RobotState.Status.ReachedGoal
		elif np.isneginf(robot_state).any():
			rs.status = mctscpp.RobotState.Status.Captured
		else:
			rs.status = mctscpp.RobotState.Status.Active

		if robot_idx in team_1_idxs: 
			attackers.append(rs)
		elif robot_idx in team_2_idxs:
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

def expected_value(param,state,policy_dict,team):

	g = param_to_cpp_game(param.robot_team_composition,param.robot_types,param.env_xlim,param.env_ylim,\
		param.sim_dt,param.goal,param.rollout_horizon)	
	gs = state_to_cpp_game_state(state,team,param.team_1_idxs,param.team_2_idxs)

	policy_a = create_cpp_policy(policy_dict, 'a')
	policy_b = create_cpp_policy(policy_dict, 'b')

	my_policy = policy_a 
	other_policies = [policy_b]
	valuePredictor = create_cpp_value(policy_dict["path_value_fnc"])	

	mctssettings = mctscpp.MCTSSettings()
	mctssettings.num_nodes = policy_dict["mcts_tree_size"]
	mctssettings.Cp = policy_dict["mcts_c_param"]
	mctssettings.pw_C = policy_dict["mcts_pw_C"]
	mctssettings.pw_alpha = policy_dict["mcts_pw_alpha"]
	mctssettings.beta1 = policy_dict["mcts_beta1"]
	mctssettings.beta3 = policy_dict["mcts_beta3"]

	mctsresult = mctscpp.search(g, gs, \
		my_policy,
		other_policies,
		valuePredictor,
		mctssettings)

	return mctsresult.expectedReward[0], mctsresult.bestAction

def self_play(param):

	if is_valid_policy_dict(param.policy_dict):
		policy_dict_a = param.policy_dict
		policy_dict_b = param.policy_dict 
		param.policy_dict_a = policy_dict_a 
		param.policy_dict_b = policy_dict_b 
	else: 
		print('bad policy dict')
		exit()

	return play_game(param,policy_dict_a,policy_dict_b)


def play_game(param,policy_dict_a,policy_dict_b): 

	if policy_dict_a["sim_mode"] == "PANAGOU" or policy_dict_b["sim_mode"] == "PANAGOU":

		# set panagou robots to zero physical radius to prevent self collisions 
		robot_idxs = []
		if policy_dict_a["sim_mode"] == "PANAGOU": 
			robot_idxs.extend(param.team_1_idxs)
		elif policy_dict_b["sim_mode"] == "PANAGOU": 
			robot_idxs.extend(param.team_2_idxs)

		for robot_idx in robot_idxs: 
			param.robots[robot_idx]["radius"] = 0.0 

		pp = PanagouPolicy(param)
		pp.init_sim(param.state)

	g = param_to_cpp_game(param.robot_team_composition,param.robot_types,param.env_xlim,param.env_ylim,\
		param.sim_dt,param.goal,param.rollout_horizon)
	policy_a = create_cpp_policy(policy_dict_a, 'a')
	policy_b = create_cpp_policy(policy_dict_b, 'b')

	valuePredictor_a = create_cpp_value(None)
	if policy_dict_a["sim_mode"] in ["MCTS","D_MCTS"]:
		if policy_dict_a["path_glas_model_a"] is not None:
			model_num = int(os.path.basename(policy_dict_a["path_glas_model_a"])[1])
		if policy_dict_a["path_value_fnc"] is not None:
			valuePredictor_a = create_cpp_value(policy_dict_a["path_value_fnc"])

	valuePredictor_b = create_cpp_value(None)
	if policy_dict_b["sim_mode"] in ["MCTS","D_MCTS"]:
		value_path_b = None 
		if policy_dict_b["path_glas_model_b"] is not None:
			model_num = int(os.path.basename(policy_dict_b["path_glas_model_b"])[1])
		if policy_dict_b["path_value_fnc"] is not None:
			valuePredictor_b = create_cpp_value(policy_dict_b["path_value_fnc"])

	sim_result = {
		'param' : param.to_dict(),
		'states' : [],
		'actions' : [],
		'times' : None,
		'rewards' : [],
		'trees': [],
		'tree_params': [],
		'n_rgs': [],
	}

	gs = state_to_cpp_game_state(param.state,"a",param.team_1_idxs,param.team_2_idxs)
	count = 0
	invalid_team_action = [np.nan*np.ones(2) for _ in range(param.num_nodes)]
	team_action = list(invalid_team_action)
	gs.depth = 0
	while True:

		if gs.turn == mctscpp.GameState.Turn.Attackers:
			policy_dict = policy_dict_a
			team_idx = param.team_1_idxs
			my_policy = policy_a
			other_policies = [policy_b]
			valuePredictor = valuePredictor_a
			team = 'a'
		elif gs.turn == mctscpp.GameState.Turn.Defenders:
			policy_dict = policy_dict_b
			team_idx = param.team_2_idxs
			my_policy = policy_b
			other_policies = [policy_a]
			valuePredictor = valuePredictor_b
			team = 'b'

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

			# calc reachedgoal
			reachedGoal = 0.0 
			for rs in gs.attackers:
				if rs.status == mctscpp.RobotState.Status.ReachedGoal:
					reachedGoal += 1.0
			sim_result['n_rgs'].append(reachedGoal)

		if isTerminal:
			# calc reachedgoal
			reachedGoal = 0.0 
			for rs in gs.attackers:
				if rs.status == mctscpp.RobotState.Status.ReachedGoal:
					reachedGoal += 1.0
			sim_result['reached_goal'] = reachedGoal / len(gs.attackers)
			break

		if policy_dict["sim_mode"] in ["MCTS", "D_MCTS"]:
			mctssettings = mctscpp.MCTSSettings()
			mctssettings.num_nodes = policy_dict["mcts_tree_size"]
			mctssettings.Cp = policy_dict["mcts_c_param"]
			mctssettings.pw_C = policy_dict["mcts_pw_C"]
			mctssettings.pw_alpha = policy_dict["mcts_pw_alpha"]
			mctssettings.beta1 = policy_dict["mcts_beta1"]
			mctssettings.beta3 = policy_dict["mcts_beta3"]
			mctssettings.export_tree = (count // 2) % param.tree_timestep == 0  and param.plot_tree_on 

		if policy_dict["sim_mode"] == "MCTS":
			depth = gs.depth
			gs.depth = 0
			mctsresult = mctscpp.search(g, gs, \
				my_policy,
				other_policies,
				valuePredictor,
				mctssettings)
			gs.depth = depth
			if mctsresult.success: 
				action = mctsresult.bestAction
				success = g.step(gs, action, gs)
				if mctssettings.export_tree:
					sim_result['trees'].append(mctsresult.tree)
					sim_result['tree_params'].append({
							'tree_team_1_idxs' : param.team_1_idxs,
							'tree_team_2_idxs' : param.team_2_idxs,
							'time' : param.sim_dt*len(sim_result['states']),
							'robot_idx' : 0,
						})
			else:
				success = False

		elif policy_dict["sim_mode"] == "D_MCTS": 
			state = np.array([rs.state.copy() for rs in gs.attackers + gs.defenders])
			action = np.nan*np.zeros((state.shape[0],2))
			for robot_idx in team_idx: 
				if not np.isfinite(state[robot_idx,:]).all(): # non active robot 
					continue
				o_a,o_b,goal = global_to_local(state,param,robot_idx)
				state_i, robot_team_composition_i, self_idx, team_1_idxs_i, team_2_idxs_i = \
					local_to_global(param,o_a,o_b,goal,team)
				game_i = param_to_cpp_game(robot_team_composition_i,param.robot_types,param.env_xlim,param.env_ylim,\
					param.sim_dt,param.goal,param.rollout_horizon)
				gamestate_i = state_to_cpp_game_state(state_i,team,team_1_idxs_i,team_2_idxs_i)
				gamestate_i.depth = 0
				mctsresult = mctscpp.search(game_i, gamestate_i, \
					my_policy,
					other_policies,
					valuePredictor,
					mctssettings)

				if mctsresult.success: 
					action_i = mctsresult.bestAction
					action[robot_idx,:] = action_i[self_idx]
					if mctssettings.export_tree:
						sim_result['trees'].append(mctsresult.tree)
						sim_result['tree_params'].append({
							'tree_team_1_idxs' : team_1_idxs_i,
							'tree_team_2_idxs' : team_2_idxs_i,
							'time' : param.sim_dt*len(sim_result['states']),
							'robot_idx' : robot_idx,
							})
				else: 
					action[robot_idx,:] = np.zeros(2) 

			success = g.step(gs,action,gs)

		elif policy_dict["sim_mode"] == "GLAS":
			action = mctscpp.eval(g, gs, policy_a, policy_b, policy_dict["deterministic"])
			success = g.step(gs, action, gs)

		elif policy_dict["sim_mode"] == "PANAGOU":

			pstate = cpp_state_to_pstate(gs)
			pp.init_sim(pstate)
			action = pp.eval(pstate)
			success = g.step(gs, action, gs)

		elif policy_dict["sim_mode"] == "RANDOM":
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


def evaluate_expert(rank, queue, total, states,param,quiet_on=True):
	
	if not quiet_on:
		print('   running expert for instance {}'.format(param.dataset_fn))

	if rank == 0:
		pbar = tqdm.tqdm(total=total)

	game = param_to_cpp_game(param.robot_team_composition,param.robot_types,param.env_xlim,param.env_ylim,\
		param.sim_dt,param.goal,param.rollout_horizon)

	my_policy = create_cpp_policy(param.my_policy_dict, param.training_team)
	value_path = param.l_value_model_fn.format(\
		DATADIR=param.path_current_models,\
		ITER=param.i) if param.i > 0 else None 
	valuePredictor = create_cpp_value(value_path) 

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

	mctssettings = mctscpp.MCTSSettings()
	mctssettings.num_nodes = param.my_policy_dict["mcts_tree_size"]
	mctssettings.Cp = param.my_policy_dict["mcts_c_param"]
	mctssettings.pw_C = param.my_policy_dict["mcts_pw_C"]
	mctssettings.pw_alpha = param.my_policy_dict["mcts_pw_alpha"]
	mctssettings.beta1 = param.my_policy_dict["mcts_beta1"]
	mctssettings.beta3 = param.my_policy_dict["mcts_beta3"]

	for state in states:
		game_state = state_to_cpp_game_state(state,param.training_team,param.team_1_idxs,param.team_2_idxs)
		game_state.depth = 0 
		# print(game_state)
		mctsresult = mctscpp.search(game, game_state, \
			my_policy,
			other_policies,
			valuePredictor,
			mctssettings)
		if mctsresult.success: 
			policy_dist = valuePerAction_to_policy_dist(param,mctsresult.valuePerAction,mctsresult.bestAction) # 
			sim_result["states"].append(state) # total number of robots x state dimension per robot 
			sim_result["policy_dists"].append(policy_dist)  

		# update status
		if rank == 0:
			count = 1
			try:
				while True:
					count += queue.get_nowait()
			except Empty:
				pass
			pbar.update(count)
		else:
			queue.put_nowait(1)

	sim_result["states"] = np.array(sim_result["states"])
	sim_result["policy_dists"] = np.array(sim_result["policy_dists"])
	dh.write_sim_result(sim_result,param.dataset_fn)

	if not quiet_on:
		print('   completed instance {} with {} dp.'.format(param.dataset_fn,sim_result["states"].shape[0]))


def evaluate_expert_value(rank, queue, total, states, param, policy_fn_a, policy_fn_b, quiet_on=True):

	if not quiet_on:
		print('   running expert for instance {}'.format(param.dataset_fn))

	if rank == 0:
		pbar = tqdm.tqdm(total=total)

	sim_result = {
		'states' : [],
		'policy_dists' : [],
		'values' : [],
		'n_rgs' : [],
		'param' : param.to_dict()
		}

	if param.l_glas_rollout_on:
		policy_dict_a = {
			'sim_mode': 'GLAS',
			'path_glas_model' : policy_fn_a,
			'deterministic': False,
		}

		policy_dict_b = {
			'sim_mode': 'GLAS',
			'path_glas_model' : policy_fn_b,
			'deterministic': False,
		}
		num_rollouts = 10 

	else: 
		if param.i > 0:
			path_value_fnc = param.l_value_model_fn.format(\
								DATADIR=param.path_current_models,\
								ITER=param.i)
		else: 
			path_value_fnc = None

		policy_dict_a = {
			'sim_mode' : 				"D_MCTS", 
			'path_glas_model_a' : 		policy_fn_a, 	
			'path_glas_model_b' : 		policy_fn_b, 	
			'path_value_fnc' : 			path_value_fnc, 	
			'mcts_tree_size' : 			param.l_num_learner_nodes,
			'mcts_c_param' : 			param.l_mcts_c_param,
			'mcts_pw_C' : 				param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			param.l_mcts_pw_alpha,
			'mcts_beta1' : 				param.l_mcts_beta1,
			'mcts_beta2' : 				param.l_mcts_beta2,
			'mcts_beta3' : 				param.l_mcts_beta3,
		}
		policy_dict_b = policy_dict_a.copy() 
		num_rollouts = 1 


	for state in states:
		param.state = state

		values = [] 
		for _ in range(num_rollouts):
			glas_rollout_sim_result = play_game(param,policy_dict_a,policy_dict_b)
			values.append(glas_rollout_sim_result["rewards"][-1,0])
			# values = glas_rollout_sim_result["rewards"][-1,0] * np.ones((glas_rollout_sim_result["states"].shape[0],1)) 

			# sim_result["states"].extend(glas_rollout_sim_result["states"])
			# sim_result["n_rgs"].extend(glas_rollout_sim_result["n_rgs"])
			# sim_result["values"].extend(values)

			# sim_result["states"].append(glas_rollout_sim_result["states"][0]) 
			# sim_result["n_rgs"].append(glas_rollout_sim_result["n_rgs"][0])
			# sim_result["values"].append(values[[0]])			

		sim_result["states"].append(glas_rollout_sim_result["states"][0]) 
		sim_result["n_rgs"].append(glas_rollout_sim_result["n_rgs"][0])
		sim_result["values"].append(sum(values)/len(values))			

		# update status
		if rank == 0:
			count = 1
			try:
				while True:
					count += queue.get_nowait()
			except Empty:
				pass
			pbar.update(count)
		else:
			queue.put_nowait(1)			

	dh.write_sim_result(sim_result,param.value_dataset_fn)

def valuePerAction_to_policy_dist(param,valuePerAction,bestAction):

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

	elif param.l_gaussian_on: 

		# "first" option 
		actions = np.array([np.array(a).flatten() for a,_ in valuePerAction])
		values = np.array([v for _,v in valuePerAction])

		dist = defaultdict(list)
		for robot_idx in robot_idxs: 
			action_idx = robot_idx * 2 + np.arange(2)

			# average = np.average(actions[:,action_idx], weights=values, axis=0)
			average = np.array(bestAction).flatten()[action_idx]
			variance = np.average((actions[:,action_idx]-average)**2, weights=values, axis=0)
			dist[robot_idx] = np.array([[average,variance]])


		# "second" option
		# valuePerActionSorted = sorted(valuePerAction, key=lambda p: p[1], reverse=True)
		# mean_action = np.array(valuePerActionSorted[0][0]).flatten() # (num_robots,2) -> (2*num_robots,) 
		# mean_value = np.array(valuePerActionSorted[0][1])

		# dist = defaultdict(list)
		# for robot_idx in robot_idxs: 
		# 	action_idx = robot_idx * 2 + np.arange(2)

		# 	mean_action_i = mean_action[action_idx]
		# 	var_action_i = np.zeros((2))
		# 	for action,value in valuePerAction: 
		# 		var_action_i += np.power(np.array(action).flatten()[action_idx] - mean_action_i,2)
		# 		# var_action_i += value*np.power(np.array(action).flatten()[action_idx] - mean_action_i,2)
		# 	# var_action_i = var_action_i / sum([value for _,value in valuePerAction])

		# 	var_action_i /= len(valuePerAction)
		# 	dist[robot_idx] = np.array([[mean_action_i,var_action_i]])

	else: 
		# sort so that highest values are first
		valuePerActionSorted = sorted(valuePerAction, key=lambda p: p[1], reverse=True)

		# limit to l_num_samples
		if len(valuePerActionSorted) > param.l_num_samples:
			valuePerActionSorted = valuePerActionSorted[0:param.l_num_samples]

		# renormalize
		# norm = sum([value for _, value in valuePerActionSorted])

		normalization = "linear"
		if normalization == "linear": 
			norm = sum([value for _, value in valuePerActionSorted])
			valuePerActionSorted = [ (a, v/norm) for a,v in valuePerActionSorted]
		elif normalization == "softmax": 
			beta = 1.0 
			norm = sum([np.exp(beta*value) for _, value in valuePerActionSorted])
			valuePerActionSorted = [ (a, np.exp(beta*v) / norm) for a,v in valuePerActionSorted] 

		dist = defaultdict(list)
		for action,value in valuePerActionSorted:

			action = np.array(action)
			action = action.flatten()

			for robot_idx in robot_idxs:
				action_idx = robot_idx * 2 + np.arange(2)
				# v = value/norm if norm > 0 else 1/len(valuePerActionSorted)
				# dist[robot_idx].append([action[action_idx],v])
				dist[robot_idx].append([action[action_idx],value])

		# dist = defaultdict(list)
		# action = np.array(bestAction)
		# action = action.flatten()
		# for robot_idx in robot_idxs:
		# 	action_idx = robot_idx * 2 + np.arange(2)
		# 	dist[robot_idx].append([action[action_idx],1.0])

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
		if bad_key(policy_dict,"path_glas_model") or \
			bad_key(policy_dict, "deterministic"):
			return False 

	elif policy_dict["sim_mode"] in ["MCTS","D_MCTS"]:
		if bad_key(policy_dict,"path_glas_model_a") or \
			bad_key(policy_dict,"path_glas_model_b") or \
			bad_key(policy_dict,"mcts_tree_size") or \
			bad_key(policy_dict,"mcts_beta2") or \
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