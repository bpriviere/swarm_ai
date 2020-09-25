
# standard
import os,sys,glob,shutil
import numpy as np 
import time 
import torch 
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn 
from collections import defaultdict
from itertools import repeat
from multiprocessing import cpu_count, Pool, freeze_support
from tqdm import tqdm
import itertools
import yaml 
import random

# custom 
import plotter
import datahandler as dh
from cpp_interface import evaluate_expert, test_evaluate_expert, self_play
from param import Param 
# from learning.discrete_emptynet import DiscreteEmptyNet
from learning.continuous_emptynet import ContinuousEmptyNet

def my_loss(value, policy, target_value, target_policy, weight, mu, sd,l_subsample_on):
	# value \& policy network : https://www.nature.com/articles/nature24270
	# for kl loss : https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
	# for wmse : https://stackoverflow.com/questions/57004498/weighted-mse-loss-in-pytorch

	if l_subsample_on:
		criterion = nn.MSELoss(reduction='sum')
		mse = criterion(policy,target_policy) + criterion(value,target_value)
		kldiv = 0.5 * torch.sum(- 1 - torch.log(sd.pow(2)) + mu.pow(2) + sd.pow(2)) 
	else: 
		criterion = nn.MSELoss(reduction='none')
		# mse = torch.sum(weight*criterion(value, target_value) + weight*criterion(policy, target_policy))
		mse = torch.sum(weight*(criterion(value, target_value) + criterion(policy, target_policy)))
		kldiv = 0.5 * torch.sum( weight * (- 1 - torch.log(sd.pow(2)) + mu.pow(2) + sd.pow(2))) 

	kld_weight = 1e-4
	loss = mse + kld_weight * kldiv
	loss = loss / value.shape[0]

	return loss

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


def train(model,optimizer,loader,l_subsample_on,l_sync_every):

	epoch_loss = 0
	for step, (o_a,o_b,goal,target_value,target_policy,weight) in enumerate(loader): 

		if step % l_sync_every == 0:
			model.require_backward_grad_sync = True
			model.require_forward_param_sync = True
		else:
			model.require_backward_grad_sync = False
			model.require_forward_param_sync = False

		value, policy, mu, sd = model(o_a,o_b,goal,x=target_policy)
		loss = my_loss(value, policy, target_value, target_policy, weight, mu, sd,l_subsample_on)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += float(loss)
	return epoch_loss


def test(model,loader,l_subsample_on):
	epoch_loss = 0
	with torch.no_grad():
		for o_a,o_b,goal,target_value,target_policy,weight in loader:
			value, policy, mu, sd = model(o_a,o_b,goal,x=target_policy)
			loss = my_loss(value, policy, target_value, target_policy, weight, mu, sd,l_subsample_on)
			epoch_loss += float(loss)
	return epoch_loss


def uniform_sample(param,n):
	states = [] 
	for _ in range(n):
		state = param.make_initial_condition() 
		states.append(state)
	return states


def get_uniform_samples(params):
	print('getting uniform samples...')
	states = []
	for param in params:
		states.append(uniform_sample(param,param.l_num_points_per_file))
	print('uniform samples collection completed.')
	return states


def get_self_play_samples(params):
	print('getting self-play samples...')
	self_play_states = []

	for param in params:
		param.policy_dict["sim_mode"] = "GLAS"

	for param in params: 
		states_per_file = [] 
		while len(states_per_file) < param.l_num_points_per_file:
			param.state = param.make_initial_condition()
			sim_result = self_play(param,deterministic=False)

			# clean data
			idxs = np.logical_not(np.isnan(sim_result["states"]).any(axis=2).any(axis=1))
			states = sim_result["states"][idxs]

			if len(states) > param.l_num_points_per_file:
				states = states[0:param.l_num_points_per_file]
			
			states_per_file.extend(states)
		self_play_states.append(states_per_file)
	print('self-play sample collection completed.')
	return self_play_states
	

def increment():
	exit('not implemented')


def make_labelled_data(sim_result,oa_pairs_by_size):

	param = load_param(sim_result["param"])
	states = sim_result["states"] # nt x nrobots x nstate_per_robot
	policy_dists = sim_result["policy_dists"]  
	values = sim_result["values"] # nt 

	if param.training_team == "a":
		robot_idxs = param.team_1_idxs
	elif param.training_team == "b":
		robot_idxs = param.team_2_idxs

	for timestep,(state,policy_dist,value) in enumerate(zip(states,policy_dists,values)):
		for robot_idx in robot_idxs:
			o_a, o_b, goal = relative_state(state,param,robot_idx)
			key = (param.training_team,len(o_a),len(o_b))

			for action, weight in zip(policy_dist[robot_idx][:,0],policy_dist[robot_idx][:,1]):

				# o_a2 = o_a + cov_s * np.random.normal(size=o_a.shape)
				# o_b2 = o_b + cov_s * np.random.normal(size=o_b.shape)
				# goal2 = goal + cov_s * np.random.normal(size=goal.shape)
				# action2 = action + cov_a * np.random.normal(size=action.shape)
				# oa_pairs_by_size[key].append((o_a2, o_b2, goal2, value, action2, weight))

				oa_pairs_by_size[key].append((o_a, o_b, goal, value, action, weight))

	return oa_pairs_by_size


def write_labelled_data(df_param,oa_pairs_by_size):

	for (team, num_a, num_b), oa_pairs in oa_pairs_by_size.items():
		batch_num = 0 
		batched_dataset = [] 

		random.shuffle(oa_pairs)

		for (o_a, o_b, goal, value, action, weight) in oa_pairs:
			data = np.concatenate((np.array(o_a).flatten(),\
				np.array(o_b).flatten(),np.array(goal).flatten(),np.array(value).flatten(),\
				np.array(action).flatten(),np.array(weight).flatten()))

			batched_dataset.append(data)
			if len(batched_dataset) >= df_param.l_batch_size:
				batch_fn = df_param.l_labelled_fn.format(DATADIR=df_param.path_current_data,\
					TEAM=team,NUM_A=num_a,NUM_B=num_b,IDX_TRIAL=batch_num)
				dh.write_oa_batch(batched_dataset,batch_fn) 
				batch_num += 1 
				batched_dataset = [] 

		if len(batched_dataset) > 0:
			batch_fn = df_param.l_labelled_fn.format(DATADIR=df_param.path_current_data,\
				TEAM=team,NUM_A=num_a,NUM_B=num_b,IDX_TRIAL=batch_num)
			dh.write_oa_batch(batched_dataset,batch_fn) 	


def make_loaders(df_param,batched_files):

	train_loader = [] # lst of batches 
	test_loader = [] 
	train_dataset_size, test_dataset_size = 0,0
	num_train_batches = int(len(batched_files) * df_param.l_test_train_ratio)
	random.shuffle(batched_files)
	for k, batched_file in enumerate(batched_files):

		o_a,o_b,goal,value,action,weight = dh.read_oa_batch(batched_file)
		data = [
			torch.from_numpy(o_a).float().to(df_param.device),
			torch.from_numpy(o_b).float().to(df_param.device),
			torch.from_numpy(goal).float().to(df_param.device),
			torch.from_numpy(value).float().to(df_param.device).unsqueeze(1),
			torch.from_numpy(action).float().to(df_param.device),
			torch.from_numpy(weight).float().to(df_param.device).unsqueeze(1),
			]
		
		if k < num_train_batches:
			train_loader.append(data)
			train_dataset_size += value.shape[0]
		else:
			test_loader.append(data)
			test_dataset_size += value.shape[0]

	return train_loader,test_loader, train_dataset_size, test_dataset_size

def train_model_parallel(rank, world_size, df_param, batched_files, training_team, model_fn, parallel=True):

	if parallel:
		torch.set_num_threads(1)

		# initialize the process group
		os.environ['MASTER_ADDR'] = 'localhost'
		os.environ['MASTER_PORT'] = '12355'
		torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

	num_batches = len(batched_files) // world_size
	my_batched_files = batched_files[rank:num_batches*world_size:world_size]

	train_loader,test_loader,train_dataset_size,test_dataset_size = make_loaders(df_param,my_batched_files)
	num_train_batches = len(train_loader)
	num_test_batches = len(test_loader)
	print(rank, num_train_batches, num_test_batches)

	dataset_size = torch.tensor([train_dataset_size, test_dataset_size, num_train_batches, num_test_batches])

	if parallel:
		torch.distributed.reduce(dataset_size, 0)

	if rank == 0:
		print('train dataset size: ', int(dataset_size[0]))
		print('test dataset size: ', int(dataset_size[1]))
		if parallel:
			print('device: multi-cpu: ', world_size)
		else:
			print('device: ', df_param.device)
		num_train_batches = int(dataset_size[2])
		num_test_batches = int(dataset_size[3])
		losses = []
		pbar = tqdm(range(1,df_param.l_n_epoch+1))
		best_test_loss = np.Inf
	else:
		pbar = range(1,df_param.l_n_epoch+1)

	if parallel:
		torch.distributed.barrier()

	start_time = time.time()

	single_model = ContinuousEmptyNet(df_param,df_param.device)

	if parallel:
		model = torch.nn.parallel.DistributedDataParallel(single_model, find_unused_parameters=True)
	else:
		model = single_model

	optimizer = torch.optim.Adam(model.parameters(), lr=df_param.l_lr, weight_decay=df_param.l_wd)

	for epoch in pbar:

		random.shuffle(train_loader)
		random.shuffle(test_loader)

		train_epoch_loss = train(model,optimizer,train_loader,df_param.l_subsample_on,df_param.l_sync_every)
		test_epoch_loss = test(model,test_loader,df_param.l_subsample_on)

		epoch_loss = torch.tensor([train_epoch_loss, test_epoch_loss])

		if parallel:
			torch.distributed.all_reduce(epoch_loss)
			train_epoch_loss = float(epoch_loss[0]) / num_train_batches
			test_epoch_loss = float(epoch_loss[1]) / num_test_batches
			if np.isnan(train_epoch_loss):
				if rank == 0:
					print("WARNING: NAN encountered during training! Aborting.")
				break

		if rank == 0:
			losses.append((train_epoch_loss,test_epoch_loss))

			if epoch%df_param.l_log_interval==0:
				if test_epoch_loss < best_test_loss:
					best_test_loss = test_epoch_loss
					pbar.set_description("Best Test Loss: {:.5f}".format(best_test_loss))
					torch.save(single_model.to('cpu').state_dict(), model_fn)
					single_model.to(df_param.device)

	# if parallel:
		# torch.distributed.barrier()
	
	if rank == 0:
		print("time for training: ", time.time() - start_time)
		plotter.plot_loss(losses,training_team)
		plotter.save_figs(model_fn + '.pdf')
		# plotter.open_figs('plots/model.pdf')
		print('training model complete for {}'.format(model_fn))

	if parallel:
		torch.distributed.destroy_process_group()


def train_model(df_param,batched_files,training_team,model_fn):

	print('training model... {}'.format(model_fn))

	if df_param.device == 'cpu' and df_param.num_cpus is not None:
		torch.multiprocessing.spawn(train_model_parallel,
			args=(df_param.num_cpus, df_param, batched_files, training_team, model_fn, True),
			nprocs=df_param.num_cpus,
			join=True)
	else:
		train_model_parallel(0, 1, df_param,batched_files,training_team,model_fn, False)


def load_param(some_dict):
	param = Param()
	param.from_dict(some_dict)
	return param 

def evaluate_expert_wrapper(arg):
	evaluate_expert(*arg)

def test_evaluate_expert_wrapper(arg):
	test_evaluate_expert(*arg)

def make_dataset(states,params,df_param,testing=None):
	print('making dataset...')

	for param in params:
		param.policy_dict["sim_mode"] = "MCTS"

	if not df_param.l_parallel_on:
		if df_param.mice_testing_on:
			for states_per_file, param in zip(states, params): 
				test_evaluate_expert(states_per_file,param,testing,quiet_on=True,progress=None) 				
		else:
			for states_per_file, param in zip(states, params): 
				evaluate_expert(states_per_file, param, quiet_on=False)
	else:
		global pool_count
		freeze_support()
		ncpu = cpu_count()
		print('ncpu: ', ncpu)
		num_workers = min(ncpu-1, len(params))
		with Pool(num_workers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
			if df_param.mice_testing_on:
				args = list(zip(states, params, itertools.repeat(testing), itertools.repeat(True), itertools.repeat(pool_count)))
				r = list(tqdm(p.imap_unordered(test_evaluate_expert_wrapper, args), total=len(args), position=0))
			else:
				args = list(zip(states, params, itertools.repeat(True), itertools.repeat(pool_count)))
				r = list(tqdm(p.imap_unordered(evaluate_expert_wrapper, args), total=len(args), position=0))
			# p.starmap(evaluate_expert, states, params)
			# p.starmap(evaluate_expert, list(zip(states, params)))
		pool_count += num_workers

def make_labelled_dataset(df_param):
	# labelled dataset 
	print('cleaning labelled data...')
	labelled_data_fns = df_param.l_labelled_fn.format(DATADIR=df_param.path_current_data,TEAM='**',NUM_A='**',NUM_B='**',IDX_TRIAL='**')
	for file in glob.glob(labelled_data_fns+'**'):
		os.remove(file)

	print('making labelled data...')
	sim_result_fns = df_param.l_raw_fn.format(DATADIR=df_param.path_current_data,TEAM='**',NUM_A='**',NUM_B='**',IDX_TRIAL='**')
	oa_pairs_by_size = defaultdict(list) 
	for sim_result_fn in tqdm(glob.glob(sim_result_fns+'**')): 
		sim_result = dh.load_sim_result(sim_result_fn)
		oa_pairs_by_size = make_labelled_data(sim_result,oa_pairs_by_size)

	# make actual batches and write to file 
	write_labelled_data(df_param,oa_pairs_by_size)
	print('labelling data completed.')
	print('dataset completed.')

def get_start(df_param,robot_team_composition):
	num_nodes_A = 0 
	for robot_type,number in robot_team_composition["a"].items():
		num_nodes_A += number
	num_nodes_B = 0 
	for robot_type,number in robot_team_composition["b"].items():
		num_nodes_B += number
	base_fn = df_param.l_raw_fn.format(DATADIR=df_param.path_current_data,TEAM=training_team,\
		NUM_A=num_nodes_A,NUM_B=num_nodes_B,IDX_TRIAL="*")
	start = len(glob.glob(base_fn+'*'))
	return start

def get_params(df_param,training_team,iter_i):

	params = []

	for robot_team_composition in df_param.l_robot_team_composition_cases:

		start = get_start(df_param,robot_team_composition)

		for trial in range(df_param.l_num_file_per_iteration): 

			param = Param() # random seed 
			param.robot_team_composition = robot_team_composition 
			param.l_num_points_per_file = df_param.l_num_points_per_file
			param.training_team = training_team
			param.iter_i = iter_i 

			param.policy_dict["sim_mode"] = "MCTS"

			if "a" in df_param.l_training_teams:
				param.policy_dict["path_glas_model_a"] = df_param.l_model_fn.format(\
					DATADIR=df_param.path_current_models,TEAM="a",ITER=iter_i)
			else: 
				param.policy_dict["path_glas_model_a"] = df_param.l_model_fn.format(\
					DATADIR=df_param.path_current_models,TEAM="a",ITER=0)
			if "b" in df_param.l_training_teams:
				param.policy_dict["path_glas_model_b"] = df_param.l_model_fn.format(\
					DATADIR=df_param.path_current_models,TEAM="b",ITER=iter_i)
			else: 
				param.policy_dict["path_glas_model_b"] = df_param.l_model_fn.format(\
					DATADIR=df_param.path_current_models,TEAM="b",ITER=0)
			
			if df_param.l_mode == "IL" or df_param.l_mode == "DAgger":
				param.policy_dict["mcts_rollout_beta"] = 0.0 
			elif df_param.l_mode == "ExIt" or df_param.l_mode == "MICE":
				param.policy_dict["mcts_rollout_beta"] = df_param.policy_dict["mcts_rollout_beta"]

			param.update() 
			param.dataset_fn = df_param.l_raw_fn.format(DATADIR=df_param.path_current_data,TEAM=training_team,\
				NUM_A=param.num_nodes_A,NUM_B=param.num_nodes_B,IDX_TRIAL=trial+start)

			params.append(param)

	return params	

def format_dir(df_param):

	if df_param.clean_data_on:
		datadir = df_param.path_current_data
		if os.path.exists(datadir):
			for file in glob.glob(datadir + "/raw_*"):
				os.remove(file)
		os.makedirs(datadir,exist_ok=True)

	if df_param.clean_models_on:
		modeldir = df_param.path_current_models
		if os.path.exists(modeldir):
			for file in glob.glob(modeldir + "/*"):
				os.remove(file)
		os.makedirs(modeldir,exist_ok=True)	

def test_model(param,model_fn,testing):

	print('testing model: {}'.format(model_fn))

	stats = {} 
	model = ContinuousEmptyNet(param, "cpu")
	model.load_state_dict(torch.load(model_fn))

	robot_idx = 0 
	n_samples = 1000

	for alpha in range(len(testing)):
		
		stats_per_condition = {
			'learned' : [],
			'test' : [],
			'latent' : [], 
		}

		test_state = np.array(testing[alpha]["test_state"])
		o_a,o_b,goal = relative_state(test_state,param,robot_idx)
		o_a,o_b,goal = format_data(o_a,o_b,goal)

		value, _ = model(o_a,o_b,goal)

		print('test num: ', alpha)
		print('   value: ',value)
		print('   valuePerAction: ', testing[alpha]["test_valuePerAction"])

		# learned distribution 
		for i_sample in range(n_samples):
			_, learned_sample = model(o_a,o_b,goal)
			stats_per_condition["learned"].append(learned_sample.detach().numpy())

		# test distribution 
		weights = []
		actions = [] 
		for action in testing[alpha]["test_valuePerAction"]:
			weights.append(action[-1])
			actions.append(np.array((action[0],action[1])))

		weights = np.array(weights)
		actions = np.array(actions)
		weights /= sum(weights) 
		choice_idxs = np.random.choice(actions.shape[0],n_samples,p=weights)
		for choice_idx in choice_idxs:
			mu = actions[choice_idx,:]
			test_sample = mu + cov_a * np.random.normal(size=(2,)) 
			stats_per_condition["test"].append(test_sample)

		stats_per_condition["learned"] = np.array(stats_per_condition["learned"]).squeeze()
		stats_per_condition["test"] = np.array(stats_per_condition["test"]).squeeze()

		stats[alpha] = stats_per_condition

	return stats

def read_testing_yaml(fn):
	
	with open(fn) as f:
		testing_cfg = yaml.load(f, Loader=yaml.FullLoader)

	testing = []
	for test in testing_cfg["test_continuous_glas"]:
		testing.append(test)
	return testing

if __name__ == '__main__':

	cov_a = 1e-2
	cov_s = 1e-2

	pool_count = 0

	df_param = Param() 
	df_param.clean_data_on = True
	df_param.clean_models_on = True
	df_param.make_data_on = True
	df_param.make_labelled_data_on = True
	df_param.mice_testing_on = False

	if df_param.mice_testing_on:
		testing = read_testing_yaml("testing/test_continuous_glas.yaml")
	else:
		testing = None 

	print('Clean old data on: {}'.format(df_param.clean_data_on))
	print('Make new data on: {}'.format(df_param.make_data_on))

	format_dir(df_param)

	# create randomly initialized models for use in the first iteration
	model = ContinuousEmptyNet(df_param,df_param.device)
	for training_team in ["a","b"]:
		model_fn = df_param.l_model_fn.format(\
			DATADIR=df_param.path_current_models,TEAM=training_team,ITER=0)
		torch.save(model.to('cpu').state_dict(), model_fn)
	del model

	# training loop 
	for iter_i in range(df_param.l_num_iterations):
		for training_team in df_param.l_training_teams:

			print('iter: {}/{}, training team: {}'.format(iter_i,df_param.l_num_iterations,training_team))

			if df_param.make_data_on: 
				
				params = get_params(df_param,training_team,iter_i)
				# if iter_i == 0 or df_param.l_mode == "IL":
				if df_param.l_mode == "IL":
					states = get_uniform_samples(params)
				else: 
					states = get_self_play_samples(params)
				
				make_dataset(states,params,df_param,testing=testing)

			if df_param.make_labelled_data_on:
				make_labelled_dataset(df_param)

			model_fn = df_param.l_model_fn.format(\
					DATADIR=df_param.path_current_models,TEAM=training_team,ITER=iter_i+1)
			batched_fns = glob.glob(df_param.l_labelled_fn.format(\
					DATADIR=df_param.path_current_data,NUM_A='**',NUM_B='**',IDX_TRIAL='**',TEAM=training_team,ITER='**'))

			train_model(df_param,\
				batched_fns, \
				training_team,\
				model_fn)

			if df_param.mice_testing_on: 
				stats = test_model(df_param,model_fn,testing)
				plotter.plot_test_model(df_param,stats)
				plotter.save_figs('plots/model.pdf')
				plotter.open_figs('plots/model.pdf')
				exit()

			if df_param.l_mode == "Mice":
				increment(df_param)

		if not df_param.make_data_on:
			break 

	print('done!')