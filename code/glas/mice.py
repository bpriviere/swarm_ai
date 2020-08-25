

import os,sys,glob,shutil
import numpy as np 
from collections import defaultdict
from itertools import repeat 
from multiprocessing import cpu_count, Pool 
from grun import uniform_sample, config_to_game, state_to_game_state, createGLAS
from grun import value_to_dist, make_labelled_data, write_labelled_data, train_model, robot_composition_to_cpp_types
from gparam import Gparam

sys.path.append("../")
import datahandler as dh
from param import Param 
from learning.discrete_emptynet import DiscreteEmptyNet

sys.path.append("../mcts/cpp")
from buildRelease import mctscpp

def get_uniform_samples(gparam,params):
	print('getting uniform samples...')
	states = []
	for param in params: 
		states.append(uniform_sample(param,param.num_points_per_file))
	print('uniform samples collection completed.')
	return states, params

def get_self_play_samples(gparam,params):
	print('getting self-play samples...')
	states = []
	for param in params: 
		sim_result = rollout(param)
		states.append(sim_result["states"])
	print('self-play sample collection completed.')
	return states, params 

def rollout(param):
	# rollout adapted from test_python_bindings
	generator = mctscpp.createRandomGenerator(param.seed)

	g = config_to_game(param,generator) # this calls param.glas_model_a

	deterministic = True
	goal = param.goal 
	attackerTypes = robot_composition_to_cpp_types(param,"a")
	defenderTypes = robot_composition_to_cpp_types(param,"b")

	glas_a = createGLAS(param.glas_model_A, generator)
	glas_b = createGLAS(param.glas_model_B, generator)	

	results = []
	while len(results) < param.num_points_per_file: 

		state = uniform_sample(param,1)[0]
		gs = state_to_game_state(param,state)
		while True:
			gs.attackersReward = 0;
			gs.defendersReward = 0;
			gs.depth = 0;

			action = mctscpp.computeActionsWithGLAS(glas_a, glas_b, gs, goal, attackerTypes, defenderTypes, generator, deterministic)
			results.append(game_state_to_cpp_result(gs,action))

			# step twice (once per team)
			success = g.step(gs, action, gs)
			if success:
				success = g.step(gs, action, gs)

			if success:
				if g.isTerminal(gs):
					# print(gs)
					results.append(game_state_to_cpp_result(gs,action))
			else:
				break

	sim_result = dh.convert_cpp_data_to_sim_result(np.array(results),param)
	return sim_result

def game_state_to_cpp_result(gs,action):

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
	
def increment(gparam):
	exit('not implemented')

def make_dataset(states,params):
	print('making dataset...')

	if gparam.serial_on:
		for states_per_file, param in zip(states, params): 
			evaluate_expert(states_per_file, param)
	else:
		ncpu = cpu_count()
		print('ncpu: ', ncpu)
		with Pool(ncpu-1) as p:
			# p.starmap(evaluate_expert, states, params)
			p.starmap(evaluate_expert, list(zip(states, params)))

	# labelled dataset 
	print('make labelled data...')
	sim_result_fns = glob.glob('{}**raw**'.format(gparam.demonstration_data_dir))
	oa_pairs_by_size = defaultdict(list) 
	for sim_result in sim_result_fns: 
		sim_result = dh.load_sim_result(sim_result)
		oa_pairs_by_size = make_labelled_data(sim_result,oa_pairs_by_size)

	# make actual batches and write to file 
	write_labelled_data(gparam,oa_pairs_by_size)
	print('labelling data completed.')
	print('dataset completed.')

def evaluate_expert(states,param):
	print('   running expert for instance {}'.format(param.dataset_fn))

	generator = mctscpp.createRandomGenerator(param.seed)
	game = config_to_game(param,generator) 
	sim_result = {
		'states' : [],
		'actions' : [],
		'param' : param.to_dict()
		}

	for state in states:
		game_state = state_to_game_state(param,state)
		mctsresult = mctscpp.search(game, game_state, generator, param.tree_size)
		if mctsresult.success: 
			action = value_to_dist(param,mctsresult.valuePerAction) # 
			sim_result["states"].append(state) # total number of robots x state dimension per robot 
			sim_result["actions"].append(action) # total number of robots x action dimension per robot 

	sim_result["states"] = np.array(sim_result["states"])
	sim_result["actions"] = np.array(sim_result["actions"])
	dh.write_sim_result(sim_result,param.dataset_fn)
	print('   completed instance {} with {} dp.'.format(param.dataset_fn,sim_result["states"].shape[0]))

def increment():
	pass 

def get_model_fn(training_team,iter_i):
	return os.getcwd() + '/models/{}{}.pt'.format(training_team,iter_i)

def get_and_make_datadir(training_team,iter_i):
	datadir = os.getcwd() + '/data/{}{}/'.format(training_team,iter_i)
	os.makedirs(datadir,exist_ok=True)
	# os.makedirs(datadir)
	return datadir

def prepare_raw_data_gen(gparam,training_team,iter_i):

	params = []

	for robot_team_composition in gparam.robot_team_composition_cases:

		num_nodes_A = 0 
		for robot_type,number in robot_team_composition["a"].items():
			num_nodes_A += number

		num_nodes_B = 0 
		for robot_type,number in robot_team_composition["b"].items():
			num_nodes_B += number

		start = len(glob.glob('{}raw_{}train_{}a_{}b**'.format(\
				gparam.demonstration_data_dir,training_team,num_nodes_A,num_nodes_B)))

		for trial in range(gparam.num_trials): 

			param = Param()
			param.seed = int.from_bytes(os.urandom(4), sys.byteorder)
			param.robot_team_composition = robot_team_composition 
			param.num_points_per_file = gparam.num_points_per_file
			param.mode = gparam.mode 
			param.tree_size = gparam.tree_size

			# 0 means pure random, 1.0 means pure GLAS
			if gparam.mode == "DAgger" or gparam.mode == "IL":
				param.rollout_beta = 0.0
				param.mode = "MCTS_RANDOM"
			elif gparam.mode == "ExIt" or gparam.mode == "Mice":
				param.rollout_beta = gparam.rollout_beta
				param.mode = "MCTS_GLAS"
			else: 
				exit('gparam.mode not recognized')

			param.training_team = training_team
			param.iter_i = iter_i 
			param.glas_model_A = get_model_fn("a",iter_i-1)
			param.glas_model_B = get_model_fn("b",iter_i-1)
			param.dataset_fn = '{}raw_{}train_{}a_{}b_{}trial'.format(\
				gparam.demonstration_data_dir,training_team,num_nodes_A,num_nodes_B,trial+start) 
			param.update() 

			params.append(param)

	return params	

def get_dataset(gparam,training_team):
	batched_files = []
	for datadir in glob.glob(os.path.join(os.getcwd(),"data") + "/*"):
		for batched_file in glob.glob(get_batch_fn(datadir,training_team,'*','*','*')):
			batched_files.append(batched_file)
	return batched_files

def get_batch_fn(datadir,team,num_a,num_b,batch_num):
	return '{}/labelled_{}train_{}a_{}b_{}trial.npy'.format(datadir,team,num_a,num_b,batch_num)

def format_dir():

	datadir = os.path.join(os.getcwd(),"data")
	if os.path.exists(datadir):
		for sub_datadir in glob.glob(datadir + "/*"):
			shutil.rmtree(sub_datadir)
	else:
		os.makedirs(datadir)

	modeldir = os.path.join(os.getcwd(),"models")
	if os.path.exists(modeldir):
		for model in glob.glob(modeldir + "/*"):
			os.remove(model)
	else: 
		os.makedirs(modeldir)

if __name__ == '__main__':

	gparam = Gparam() 
	gparam.num_iterations = 5
	gparam.mode = "DAgger" # IL, DAgger, ExIt, Mice 

	format_dir()
	
	# 
	for iter_i in range(gparam.num_iterations):
		for training_team in gparam.training_teams: 

			print('iter: {}/{}, training team: {}'.format(iter_i,gparam.num_iterations,training_team))

			gparam.demonstration_data_dir = get_and_make_datadir(training_team,iter_i)
			params = prepare_raw_data_gen(gparam,training_team,iter_i)

			if iter_i == 0 or gparam.mode == "IL":
				states, params = get_uniform_samples(gparam,params)
			else: 
				states, params = get_self_play_samples(gparam,params) 

			make_dataset(states,params)
			train_model(gparam,get_dataset(gparam,training_team),training_team,get_model_fn(training_team,iter_i))

			if gparam.mode == "Mice":
				increment(gparam)

			