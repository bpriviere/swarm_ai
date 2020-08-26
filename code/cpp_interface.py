
# standard 
import torch 
import numpy as np 
from collections import defaultdict

# ours
from cpp.buildRelease import mctscpp
import datahandler as dh

def create_cpp_robot_type(param, robot_type):
	p_min = [param.env_xlim[0], param.env_ylim[0]]
	p_max = [param.env_xlim[1], param.env_ylim[1]]
	velocity_limit = param.__dict__[robot_type]["speed_limit"]
	acceleration_limit = param.__dict__[robot_type]["acceleration_limit"]
	tag_radius = param.__dict__[robot_type]["tag_radius"]
	r_sense = param.__dict__[robot_type]["r_sense"]
	rt = mctscpp.RobotType(p_min,p_max,velocity_limit,acceleration_limit,tag_radius,r_sense)
	return rt

def robot_composition_to_cpp_robot_types(param,team):
	types = [] 
	for robot_type, num in param.robot_team_composition[team].items():
		rt = create_cpp_robot_type(param, robot_type)
		for _ in range(num):
			types.append(rt)
	return types

def param_to_cpp_game(param,generator):
	
	attackerTypes = robot_composition_to_cpp_robot_types(param,"a") 
	defenderTypes = robot_composition_to_cpp_robot_types(param,"b") 
	
	dt = param.sim_dt 
	rollout_beta = param.mcts_rollout_beta 
	goal = param.goal 
	max_depth = param.mcts_rollout_horizon
	generator = generator

	if "GLAS" in param.sim_mode:
		glas_a = createGLAS(param.path_glas_model_a, generator)
		glas_b = createGLAS(param.path_glas_model_b, generator)
	else:
		glas_a = glas_b = mctscpp.GLAS(generator)

	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, max_depth, generator, glas_a, glas_b, rollout_beta)

	return g,glas_a,glas_b,attackerTypes,defenderTypes

def createGLAS(file, generator):
	state_dict = torch.load(file)

	glas = mctscpp.GLAS(generator)
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


def state_to_cpp_game_state(param,state,turn):

	param.state = state
	param.make_robot_teams()
	param.assign_initial_condition()

	attackers = [] 
	defenders = [] 
	for robot in param.robots: 
		rs = mctscpp.RobotState(robot["x0"][0:2],robot["x0"][2:])
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

	game_state.attackersReward = 0
	game_state.defendersReward = 0
	game_state.depth = 0

	return game_state 

def rollout(param):
	# rollout adapted from test_python_bindings

	generator = mctscpp.createRandomGenerator(param.seed)
	g,glas_a,glas_b,attackerTypes,defenderTypes = param_to_cpp_game(param,generator) # this calls param.glas_model_a

	deterministic = True
	goal = param.goal 

	results = []
	action = []

	state = param.state
	gs = state_to_cpp_game_state(param,state,"a")
	next_state = state_to_cpp_game_state(param,state,"a")
	while True:

		if "MCTS" in param.sim_mode:
			
			next_state.attackersReward = 0
			next_state.defendersReward = 0
			next_state.depth = 0

			mctsresult = mctscpp.search(g, next_state, generator, param.mcts_tree_size)
			if mctsresult.success: 
				team_action = mctsresult.bestAction
				if next_state.turn == mctscpp.GameState.Turn.Attackers:
					for idx in param.team_1_idxs: 
						action.append(team_action[idx])
				elif next_state.turn == mctscpp.GameState.Turn.Defenders:
					for idx in param.team_2_idxs: 
						action.append(team_action[idx])
				success = g.step(next_state, team_action, next_state)

				if success and next_state.turn == mctscpp.GameState.Turn.Attackers:
					print(action)
					results.append(game_state_to_cpp_result(gs,action))
					action = []
					gs = next_state
			else:
				success = False

		elif param.sim_mode == "GLAS":

			gs.attackersReward = 0
			gs.defendersReward = 0
			gs.depth = 0

			action = mctscpp.computeActionsWithGLAS(glas_a, glas_b, gs, goal, attackerTypes, defenderTypes, generator, deterministic)

			# step twice (once per team)
			success = g.step(gs, action, next_state)
			if success:
				next_state.attackersReward = 0
				next_state.defendersReward = 0
				next_state.depth = 0
				success = g.step(next_state, action, next_state)

			if not success:
				break

			results.append(game_state_to_cpp_result(gs,action))
			gs = next_state

		if success:
			if g.isTerminal(next_state):
				results.append(game_state_to_cpp_result(next_state,None))
				break 
		else:
			break

	if len(results) == 0:
		results.append(game_state_to_cpp_result(gs,None))
		
	sim_result = dh.convert_cpp_data_to_sim_result(np.array(results),param)
	return sim_result	

def game_state_to_cpp_result(gs,action):

	if action is None:
		action = np.nan*np.ones((len(gs.attackers) + len(gs.defenders),2))
		
	idx = 0 
	result = []
	for rs in gs.attackers:
		result.append(rs.position.copy())
		result.append(rs.velocity.copy())
		result.append(action[idx].copy())
		idx += 1 
	for rs in gs.defenders: 
		result.append(rs.position.copy())
		result.append(rs.velocity.copy())
		result.append(action[idx].copy()) 
		idx += 1 
	result.append([gs.attackersReward, gs.defendersReward])

	return np.array(result).flatten()	

def evaluate_expert(states,param):
	# print('   running expert for instance {}'.format(param.dataset_fn))

	generator = mctscpp.createRandomGenerator(param.seed)
	game,_,_,_,_ = param_to_cpp_game(param,generator) 
	sim_result = {
		'states' : [],
		'actions' : [],
		'param' : param.to_dict()
		}

	for state in states:
		game_state = state_to_cpp_game_state(param,state,param.training_team)
		mctsresult = mctscpp.search(game, game_state, generator, param.mcts_tree_size)
		if mctsresult.success: 
			action = value_to_dist(param,mctsresult.valuePerAction) # 
			sim_result["states"].append(state) # total number of robots x state dimension per robot 
			sim_result["actions"].append(action) # total number of robots x action dimension per robot 

	sim_result["states"] = np.array(sim_result["states"])
	sim_result["actions"] = np.array(sim_result["actions"])
	dh.write_sim_result(sim_result,param.dataset_fn)
	# print('   completed instance {} with {} dp.'.format(param.dataset_fn,sim_result["states"].shape[0]))

def evaluate_glas(states,param):
	print('   running glas for instance {}'.format(param.dataset_fn))

	generator = mctscpp.createRandomGenerator(param.seed)
	g,glas_a,glas_b,attackerTypes,defenderTypes = param_to_cpp_game(param,generator) # this calls param.glas_model_a

	sim_result = {
		'states' : [],
		'actions' : [],
		'param' : param.to_dict()
		}

	deterministic = False

	for state in states:
		gs = state_to_cpp_game_state(param,state,param.training_team)
		action = mctscpp.computeActionsWithGLAS(glas_a, glas_b, gs, param.goal, attackerTypes, defenderTypes, generator, deterministic)
		sim_result["states"].append(state) # total number of robots x state dimension per robot 
		sim_result["actions"].append(action) # total number of robots x action dimension per robot 

	sim_result["states"] = np.array(sim_result["states"])
	sim_result["actions"] = np.array(sim_result["actions"])
	dh.write_sim_result(sim_result,param.dataset_fn)
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