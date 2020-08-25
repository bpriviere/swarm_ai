
# standard
import os, sys, glob, shutil 
import itertools
import random # for vis samples of training data  
import numpy as np 
import torch 
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multiprocessing import cpu_count, Pool 
import concurrent.futures
import tempfile 
import subprocess
from collections import defaultdict
import argparse

# project
sys.path.append("../")
from env import Swarm
from run import run_sim
from param import Param 
from gparam import Gparam
from learning.emptynet import EmptyNet
from learning.discrete_emptynet import DiscreteEmptyNet
from measurements.relative_state import relative_state
from utilities import load_module
import datahandler as dh
import plotter 

sys.path.append("../mcts/cpp")
from buildRelease import mctscpp

def train(model,optimizer,loader):
	
	# loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	# loss_func = torch.nn.CrossEntropyLoss()  
	loss_func = torch.nn.MultiLabelSoftMarginLoss()  
	epoch_loss = 0
	for step, (o_a,o_b,goal,action) in enumerate(loader): 
		prediction = model(o_a,o_b,goal,training=True)
		# loss = loss_func(prediction, action.flatten())  # for cross entopy
		loss = loss_func(prediction, action) # for anything else 
		optimizer.zero_grad()   
		loss.backward()         
		optimizer.step()        
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def test(model,optimizer,loader):
	
	# loss_func = torch.nn.MSELoss()  
	# loss_func = torch.nn.CrossEntropyLoss()  
	loss_func = torch.nn.MultiLabelSoftMarginLoss()  
	epoch_loss = 0
	for step, (o_a,o_b,goal,action) in enumerate(loader): 
		prediction = model(o_a,o_b,goal,training=True)     
		# loss = loss_func(prediction, action.flatten()) # for cross entropy
		loss = loss_func(prediction, action) # for others 
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def create_robot_type_cpp(param, robot_type, team):
	if team == "a":
		p_min = [param.reset_xlim_A[0], param.reset_ylim_A[0]]
		p_max = [param.reset_xlim_A[1], param.reset_ylim_A[1]]
	elif team == "b":
		p_min = [param.reset_xlim_B[0], param.reset_ylim_B[0]]
		p_max = [param.reset_xlim_B[1], param.reset_ylim_B[1]]

	velocity_limit = param.__dict__[robot_type]["speed_limit"]
	acceleration_limit = param.__dict__[robot_type]["acceleration_limit"]
	tag_radius = param.__dict__[robot_type]["tag_radius"]
	r_sense = param.__dict__[robot_type]["r_sense"]
	rt = mctscpp.RobotType(p_min,p_max,velocity_limit,acceleration_limit,tag_radius,r_sense)
	return rt

def robot_composition_to_cpp_types(param,team):
	types = [] 
	for robot_type, num in param.robot_team_composition[team].items():
		rt = create_robot_type_cpp(param, robot_type, team)
		for _ in range(num):
			types.append(rt)
	return types


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


def config_to_game(param,generator):
	
	attackerTypes = robot_composition_to_cpp_types(param,"a") 
	defenderTypes = robot_composition_to_cpp_types(param,"b") 
	
	dt = param.sim_dt 
	goal = param.goal 
	max_depth = param.rollout_horizon
	generator = generator

	if "GLAS" in param.mode:
		glas_a = createGLAS(param.glas_model_A, generator)
		glas_b = createGLAS(param.glas_model_B, generator)
	else:
		glas_a = glas_b = mctscpp.GLAS(generator)

	rollout_beta = param.rollout_beta 

	g = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, max_depth, generator, glas_a, glas_b, rollout_beta)

	return g 


def state_to_game_state(param,state):

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

	if param.training_team == "a":
		turn = mctscpp.GameState.Turn.Attackers
	elif param.training_team == "b": 
		turn = mctscpp.GameState.Turn.Defenders
	else: 
		exit('training team not recognized') 

	game_state = mctscpp.GameState(turn,attackers,defenders)

	game_state.attackersReward = 0
	game_state.defendersReward = 0
	game_state.depth = 0

	return game_state 


def uniform_sample(param,n):
	states = [] 
	for _ in range(n):
		param.make_initial_condition() 
		states.append(param.state)
	return states


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



def collect_uniform_demonstrations(param): 

	print('running instance {}'.format(param.dataset_fn))

	# init 
	generator = mctscpp.createRandomGenerator(param.seed)
	game = config_to_game(param,generator) 
	states = uniform_sample(param,param.num_points_per_file)

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

	print('completed instance {} with {} dp'.format(param.dataset_fn,sim_result["states"].shape[0]))


def get_class_actions():
	u_xs = [-1,0,1]
	u_ys = [-1,0,1]

	master_lst = []
	# for _ in range(num_robots):
	for _ in range(1):
		master_lst.extend([u_xs,u_ys])
	master_lst = list(itertools.product(*master_lst))

	return np.array(master_lst)


def prepare_raw_data_gen(gparam):

	params = []

	for robot_team_composition in gparam.robot_team_composition_cases:

		num_nodes_A = 0 
		for robot_type,number in robot_team_composition["a"].items():
			num_nodes_A += number

		num_nodes_B = 0 
		for robot_type,number in robot_team_composition["b"].items():
			num_nodes_B += number

		for training_team in gparam.training_teams:

			start = len(glob.glob('{}raw_{}train_{}a_{}b**'.format(\
					gparam.demonstration_data_dir,training_team,num_nodes_A,num_nodes_B)))

			for trial in range(gparam.num_trials): 

				param = Param()
				param.seed = int.from_bytes(os.urandom(4), sys.byteorder)
				param.robot_team_composition = robot_team_composition 
				param.num_points_per_file = gparam.num_points_per_file
				param.mode = gparam.mode 
				param.tree_size = gparam.tree_size
				param.rollout_beta = gparam.rollout_beta
				param.training_team = training_team
				param.actions = get_class_actions()
				param.update() 

				param.dataset_fn = '{}raw_{}train_{}a_{}b_{}trial'.format(\
					gparam.demonstration_data_dir,training_team,param.num_nodes_A,param.num_nodes_B,trial+start) 

				params.append(param)

	return params


def make_labelled_data(sim_result,oa_pairs_by_size):

	param = load_param(sim_result["param"])
	states = sim_result["states"] # nt x nrobots x nstate_per_robot
	actions = sim_result["actions"] # nt x nrobots x 9 

	if param.training_team == "a":
		robot_idxs = param.team_1_idxs
	elif param.training_team == "b":
		robot_idxs = param.team_2_idxs

	for timestep,(state,action) in enumerate(zip(states,actions)):
		for robot_idx in robot_idxs:
			o_a, o_b, goal = relative_state(state,param,robot_idx)
			key = (param.training_team,len(o_a),len(o_b))
			oa_pairs_by_size[key].append((o_a, o_b, goal, action[robot_idx,:]))

	return oa_pairs_by_size


def write_labelled_data(gparam,oa_pairs_by_size):

	for (team, num_a, num_b), oa_pairs in oa_pairs_by_size.items():
		batch_num = 0 
		batched_dataset = [] 
		for (o_a, o_b, goal, action) in oa_pairs:
			data = np.concatenate((np.array(o_a).flatten(),np.array(o_b).flatten(),np.array(goal).flatten(),np.array(action).flatten()))
			batched_dataset.append(data)
			if len(batched_dataset) > gparam.il_batch_size:
				batch_fn = get_batch_fn(gparam.demonstration_data_dir,team,num_a,num_b,batch_num)
				dh.write_oa_batch(batched_dataset,batch_fn) 
				batch_num += 1 
				batched_dataset = [] 
		batch_fn = get_batch_fn(gparam.demonstration_data_dir,team,num_a,num_b,batch_num)
		dh.write_oa_batch(batched_dataset,batch_fn) 	


def load_param(some_dict):
	param = Param()
	param.from_dict(some_dict)
	return param 


def get_batch_fn(datadir,team,num_a,num_b,batch_num):
	return '{}labelled_{}train_{}a_{}b_{}trial.npy'.format(datadir,team,num_a,num_b,batch_num)


def clean_files_containing_str_from_dir(string,directory):

	if not os.path.exists(directory):
		print('no directory exists: ', directory)
		exit()

	print('cleaning files with str: {} in dir {}'.format(string,directory))

	files = glob.glob(directory + '**{}**'.format(string))
	for f in files: 
		os.remove(f)


def get_dataset_size(gparam,batched_files):
	n_points = 0 
	for batched_file in batched_files:
		print('batched_file',batched_file)
		# exit()

		o_a,o_b,goal,action = dh.read_oa_batch(batched_file,gparam.demonstration_data_dir)
		n_points += action.shape[0]
	n_points = np.min((n_points, gparam.il_n_points))
	return n_points


def make_loaders(gparam,batched_files,n_points):

	train_loader = [] # lst of batches 
	test_loader = [] 
	curr_points, train_dataset_size, test_dataset_size = 0,0,0
	for batched_file in batched_files: 
		o_a,o_b,goal,action = dh.read_oa_batch(batched_file,gparam.demonstration_data_dir)
		if curr_points < gparam.il_test_train_ratio * n_points: 
			train_loader.append([
				torch.from_numpy(o_a).float().to(gparam.device),
				torch.from_numpy(o_b).float().to(gparam.device),
				torch.from_numpy(goal).float().to(gparam.device),
				torch.from_numpy(action).float().to(gparam.device)])
				# torch.from_numpy(action).type(torch.long).to(gparam.device)])
			train_dataset_size += action.shape[0]

		elif curr_points < n_points:
			test_loader.append([
				torch.from_numpy(o_a).float().to(gparam.device),
				torch.from_numpy(o_b).float().to(gparam.device),
				torch.from_numpy(goal).float().to(gparam.device),
				torch.from_numpy(action).float().to(gparam.device)]) # for others
				# torch.from_numpy(action).type(torch.long).to(gparam.device)]) # for cross entopy
			test_dataset_size += action.shape[0]
		curr_points += action.shape[0]	

	return train_loader,test_loader, train_dataset_size, test_dataset_size


def train_model(gparam,batched_files,training_team,model_fn):

	n_points = get_dataset_size(gparam,batched_files)
	train_loader,test_loader,train_dataset_size,test_dataset_size = make_loaders(gparam,batched_files,n_points)

	print('train dataset size: ', train_dataset_size)
	print('test dataset size: ', test_dataset_size)

	model = DiscreteEmptyNet(gparam,gparam.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=gparam.il_lr, weight_decay=gparam.il_wd)
	
	# train 
	losses = []
	with open(model_fn + ".csv", 'w') as log_file:
		log_file.write("time,epoch,train_loss,test_loss\n")
		start_time = time.time()
		best_test_loss = np.Inf
		scheduler = ReduceLROnPlateau(optimizer, 'min')
		for epoch in range(1,gparam.il_n_epoch+1):
			train_epoch_loss = train(model,optimizer,train_loader)
			test_epoch_loss = test(model,optimizer,test_loader)
			scheduler.step(test_epoch_loss)
			losses.append((train_epoch_loss,test_epoch_loss))
			if epoch%gparam.il_log_interval==0:
				print('epoch: ', epoch)
				print('   Train Epoch Loss: ', train_epoch_loss)
				print('   Test Epoch Loss: ', test_epoch_loss)
				if test_epoch_loss < best_test_loss:
					best_test_loss = test_epoch_loss
					print('      saving @ best test loss:', best_test_loss)
					torch.save(model.state_dict(), model_fn)
					model.to(gparam.device)
			log_file.write("{},{},{},{}\n".format(time.time() - start_time, epoch, train_epoch_loss, test_epoch_loss))
	plotter.plot_loss(losses,training_team)


def get_team_from_batch_fn(batch_fn):
	batch_fn = os.path.basename(batch_fn)
	batch_fn = batch_fn.split("train")
	team = batch_fn[0][-1]
	return team 

def get_team_from_raw_fn(raw_fn):
	batch_fn = os.path.basename(batch_fn)
	batch_fn = batch_fn.split("train")
	team = batch_fn[0][-1]
	return team 	

if __name__ == '__main__':

	# parse 
	parser = argparse.ArgumentParser()
	parser.add_argument("-file", default=None, required=False)
	args = parser.parse_args()

	# default stuff 
	gparam = Gparam()

	# run expert and write (state, action) pairs into files 
	if gparam.make_raw_data_on:
		print('making raw data...')

		if gparam.clean_raw_data_on: 
			print('cleaning training data...')
			clean_files_containing_str_from_dir("raw",gparam.demonstration_data_dir)
			clean_files_containing_str_from_dir("labelled",gparam.demonstration_data_dir)

		params = prepare_raw_data_gen(gparam)
		if gparam.serial_on:
			for param in params:
				collect_uniform_demonstrations(param)
		else:
			ncpu = cpu_count()
			print('ncpu: ', ncpu)
			# with Pool(ncpu-1) as p:
			with Pool(ncpu) as p:
				# todo 
				# p.starmap(collect_uniform_demonstrations, params)
				p.map(collect_uniform_demonstrations, params)

	# load (state,action) files, apply measurement model, and write (observation,action) binary files
	if gparam.make_labelled_data_on: 
		print('make labelled data...')

		if gparam.clean_labelled_data_on: 
			print('cleaning labelled data...')
			clean_files_containing_str_from_dir("labelled",gparam.demonstration_data_dir)

		sim_result_fns = glob.glob('{}**raw**'.format(gparam.demonstration_data_dir))
		oa_pairs_by_size = defaultdict(list) 
		for sim_result in sim_result_fns: 
			sim_result = dh.load_sim_result(sim_result)
			oa_pairs_by_size = make_labelled_data(sim_result,oa_pairs_by_size)

		# make actual batches and write to file 
		write_labelled_data(gparam,oa_pairs_by_size)
		print('completed labelling data...')

	# load (observation,action) binary files, train a model, and write model to file 
	if gparam.train_model_on: 
		for training_team in gparam.training_teams:
			print('training model for team {}...'.format(training_team))
			batched_files = glob.glob(get_batch_fn(gparam.demonstration_data_dir,training_team,'*','*','*'))
			train_model(gparam,batched_files,training_team,gparam.il_train_model_fn.format(training_team))

	# visualize training data 
	if gparam.dbg_vis_on: 

		num_points_per_file = 9 

		if not args.file is None: 

			fn = args.file 

			if not os.path.exists(fn):
				exit('fn {} does not exist'.format(fn))

			if "raw" in fn:
				sim_result = dh.load_sim_result(fn)
				state_action_pairs = list(zip(sim_result["states"],sim_result["actions"]))
				sampled_sa_pairs = random.sample(state_action_pairs,num_points_per_file)
				training_team = get_team_from_raw_fn(fn)
				plotter.plot_sa_pairs(sampled_sa_pairs,sim_result,training_team)

			elif "labelled" in fn:
				param = Param()
				rsense = param.standard_robot["r_sense"]
				abs_goal = param.goal 
				action_list = param.actions
				training_team = get_team_from_batch_fn(fn)

				o_a,o_b,goal,actions = dh.read_oa_batch(fn,gparam.demonstration_data_dir)
				oa_pairs = list(zip(o_a,o_b,goal,actions))
				sampled_oa_pairs = random.sample(oa_pairs,num_points_per_file)
				plotter.plot_oa_pairs(sampled_oa_pairs,abs_goal,training_team,rsense,action_list)

		else: 
				
			num_files = 1

			for training_team in gparam.training_teams: 

				# plot state action pairs 
				sim_result_fns = glob.glob('{}**raw**{}train**'.format(gparam.demonstration_data_dir,training_team))
				sampled_raw_fns = random.sample(sim_result_fns,num_files)

				for sim_result_fn in sampled_raw_fns:
					sim_result = dh.load_sim_result(sim_result_fn)
					state_action_pairs = list(zip(sim_result["states"],sim_result["actions"]))
					sampled_sa_pairs = random.sample(state_action_pairs,num_points_per_file)
					plotter.plot_sa_pairs(sampled_sa_pairs,sim_result,training_team)


				# plot observation action pairs 
				batch_fns = glob.glob(get_batch_fn(gparam.demonstration_data_dir,training_team,'*','*','*'))
				sampled_batch_fns = random.sample(batch_fns,num_files)

				param = Param()
				rsense = param.standard_robot["r_sense"]
				abs_goal = param.goal 
				action_list = param.actions

				for batch_fn in sampled_batch_fns:
					o_a,o_b,goal,actions = dh.read_oa_batch(batch_fn,gparam.demonstration_data_dir)
					oa_pairs = list(zip(o_a,o_b,goal,actions))
					sampled_oa_pairs = random.sample(oa_pairs,num_points_per_file)
					plotter.plot_oa_pairs(sampled_oa_pairs,abs_goal,training_team,rsense,action_list)

	if plotter.has_figs():
		plotter.save_figs('plots.pdf')
		plotter.open_figs('plots.pdf')
