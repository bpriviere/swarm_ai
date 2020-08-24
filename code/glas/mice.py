

import os,sys,glob,shutil
import numpy as np 
from collections import defaultdict
from grun import prepare_raw_data_gen, uniform_sample, config_to_game, state_to_game_state
from grun import value_to_dist, make_labelled_data, write_labelled_data, train_model, get_batch_fn
from gparam import Gparam

sys.path.append("../")
import datahandler as dh
from learning.discrete_emptynet import DiscreteEmptyNet

sys.path.append("../mcts/cpp")
from buildRelease import mctscpp

def get_uniform_samples(gparam):
	print('getting uniform samples...')
	params = prepare_raw_data_gen(gparam) 
	states = []
	for param in params: 
		states.append(uniform_sample(param))
	print('uniform samples collection completed.')
	return states, params

def get_self_play_samples(model_a,model_b):
	pass 

def increment(gparam):
	pass 

def make_dataset(states,params,model_a,model_b,datadir):
	print('making dataset...')

	# raw datagen 
	for param in params: 
		param.dataset_fn = os.path.join(datadir,os.path.basename(param.dataset_fn))
		if model_a is None or model_b is None: 
			param.mode = "MCTS_RANDOM"
		else:
			param.mode = "MCTS_GLAS"

	if gparam.serial_on:
		for states_per_file, param in zip(states, params): 
			evaluate_expert(states_per_file, param)
	else:
		ncpu = cpu_count()
		print('ncpu: ', ncpu)
		with Pool(ncpu-1) as p:
			p.starmap(evaluate_expert, params, states)

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
	print('running expert for instance {}'.format(param.dataset_fn))

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
	print('completed instance {} with {} dp.'.format(param.dataset_fn,sim_result["states"].shape[0]))

def increment():
	pass 

def get_model_fn(training_team,iter_i):
	return os.getcwd() + '/models/{}{}.pt'.format(training_team,iter_i)

def get_and_make_datadir(training_team,iter_i):
	datadir = os.getcwd() + '/data/{}{}/'.format(training_team,iter_i)
	os.makedirs(datadir)
	return datadir

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
	gparam.num_iterations = 1

	format_dir()
	
	# 
	for iter_i in range(gparam.num_iterations):
		for training_team in gparam.training_teams: 

			datadir = get_and_make_datadir(training_team,iter_i)
			gparam.demonstration_data_dir = datadir

			if iter_i == 0:
				model_a, model_b = None, None
				states, params = get_uniform_samples(gparam)
			else: 
				model_a = DiscreteEmptyNet(gparam, device).load_state_dict(torch.load(get_model_fn("a",iter_i-1)))
				model_b = DiscreteEmptyNet(gparam, device).load_state_dict(torch.load(get_model_fn("b",iter_i-1)))
				states, params = get_self_play_samples(gparam,model_a,model_b) 

			make_dataset(states,params,model_a,model_b,datadir)
			dataset = glob.glob(get_batch_fn(gparam.demonstration_data_dir,training_team,'*','*','*'))
			train_model(gparam,dataset,training_team,get_model_fn(training_team,iter_i))

			