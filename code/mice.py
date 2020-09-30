
# standard
import os,sys,glob,shutil
import numpy as np 
import time 
import torch 
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
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


def train(model,optimizer,loader,l_subsample_on,l_sync_every,epoch, scheduler=None):

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
		if scheduler is not None:
			scheduler.step(epoch + step/len(loader))
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

		# delta-uniform sampling for curriculum 
		robot_team_composition, _, _, env_l = sample_curriculum(param.curriculum)

		# update
		param.robot_team_composition = robot_team_composition
		param.env_l = env_l
		param.update()

		states.append(uniform_sample(param,param.l_num_points_per_file))
	print('uniform samples collection completed.')
	return states


def get_self_play_samples(params):
	# When using multiprocessing, load cpp_interface per process
	from cpp_interface import play_game

	print('getting self-play samples...')
	self_play_states = []

	for param in params:

		# delta-uniform sampling for curriculum 
		robot_team_composition, skill_a, skill_b, env_l = sample_curriculum(param.curriculum)

		# update
		param.robot_team_composition = robot_team_composition
		param.env_l = env_l
		param.update()

		# dict a 
		if param.i == 0 or (param.training_team == "b" and skill_a == None):
			param.policy_dict_a = {
				"sim_mode" : "RANDOM"
			}

		elif param.training_team == "a": 
			path_glas_model_a = param.l_model_fn.format(\
						DATADIR=param.path_current_models,\
						TEAM="a",\
						ITER=param.i)
			param.policy_dict_a = {
				"sim_mode" : "GLAS",
				"path_glas_model_a" : path_glas_model_a, 
				"path_glas_model_b" : None ,
				"mcts_rollout_beta" : 1.0 
			}			

		else: 
			path_glas_model_a = param.l_model_fn.format(\
						DATADIR=param.path_current_models,\
						TEAM="a",\
						ITER=skill_a)
			param.policy_dict_a = {
				"sim_mode" : "GLAS",
				"path_glas_model_a" : path_glas_model_a, 
				"path_glas_model_b" : None ,
				"mcts_rollout_beta" : 1.0 
			}

		# dict b 
		if param.i == 0 or (param.training_team == "a" and skill_b == None):
			param.policy_dict_b = {
				"sim_mode" : "RANDOM"
			}

		elif param.training_team == "b": 
			path_glas_model_b = param.l_model_fn.format(\
						DATADIR=param.path_current_models,\
						TEAM="b",\
						ITER=param.i)
			param.policy_dict_b = {
				"sim_mode" : "GLAS",
				"path_glas_model_a" : None, 
				"path_glas_model_b" : path_glas_model_b,
				"mcts_rollout_beta" : 1.0 
			}			

		else: 
			path_glas_model_b = param.l_model_fn.format(\
						DATADIR=param.path_current_models,\
						TEAM="b",\
						ITER=skill_b)
			param.policy_dict_b = {
				"sim_mode" : "GLAS",
				"path_glas_model_a" : None, 
				"path_glas_model_b" : path_glas_model_b,
				"mcts_rollout_beta" : 1.0 
			}		

	for param in params: 
		states_per_file = []
		remaining_plots_per_file = 2
		while len(states_per_file) < param.l_num_points_per_file:
			param.state = param.make_initial_condition()
			# sim_result = self_play(param,deterministic=False)
			sim_result = play_game(param,param.policy_dict_a,param.policy_dict_b,deterministic=False)

			if remaining_plots_per_file > 0:
				plotter.plot_tree_results(sim_result)
				remaining_plots_per_file -= 1

			# clean data
			idxs = np.logical_not(np.isnan(sim_result["states"]).any(axis=2).any(axis=1))
			states = sim_result["states"][idxs]
			states_per_file.extend(states)
		self_play_states.append(states_per_file[0:param.l_num_points_per_file])
	print('self-play sample collection completed.')

	plotter.save_figs('plots/self_play_samples_team{}_iter{}.pdf'.format(params[0].training_team, params[0].i))
	return self_play_states


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

				oa_pairs_by_size[key].append((o_a, o_b, goal, value, action, weight))

	return oa_pairs_by_size


def write_labelled_data(df_param,oa_pairs_by_size,i):

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
				batch_fn = df_param.l_labelled_fn.format(
					DATADIR=df_param.path_current_data,\
					TEAM=team,
					LEARNING_ITER=i,
					NUM_A=num_a,
					NUM_B=num_b,
					NUM_FILE=batch_num)
				dh.write_oa_batch(batched_dataset,batch_fn) 
				batch_num += 1 
				batched_dataset = [] 

		# last batch 
		if len(batched_dataset) > 0:
			batch_fn = df_param.l_labelled_fn.format(\
				DATADIR=df_param.path_current_data,\
				TEAM=team,
				LEARNING_ITER=i,
				NUM_A=num_a,
				NUM_B=num_b,
				NUM_FILE=batch_num)
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
		lrs = []
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

	if df_param.l_lr_scheduler == 'ReduceLROnPlateau':
		scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=1e-5, verbose=rank==0)
		train_scheduler = None
	elif df_param.l_lr_scheduler == 'CosineAnnealingWarmRestarts':
		scheduler = CosineAnnealingWarmRestarts(optimizer, int(max(75, df_param.l_n_epoch / 20)), 1, 1e-5)
		train_scheduler = scheduler
	else:
		scheduler = None
		train_scheduler = None

	for epoch in pbar:

		random.shuffle(train_loader)
		random.shuffle(test_loader)

		train_epoch_loss = train(model,optimizer,train_loader,df_param.l_subsample_on,df_param.l_sync_every,epoch,train_scheduler)
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

		if df_param.l_lr_scheduler == 'ReduceLROnPlateau':
			scheduler.step(test_epoch_loss)

		if rank == 0:
			losses.append((train_epoch_loss,test_epoch_loss))
			if df_param.l_lr_scheduler == 'ReduceLROnPlateau':
				lrs.append(scheduler._last_lr)
			elif df_param.l_lr_scheduler == 'CosineAnnealingWarmRestarts':
				lrs.append(scheduler.get_last_lr())
			else:
				lrs.append(df_param.l_lr)

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
		plotter.plot_loss(losses,lrs,training_team)
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
	# When using multiprocessing, load cpp_interface per process
	from cpp_interface import evaluate_expert
	evaluate_expert(*arg)

def test_evaluate_expert_wrapper(arg):
	# When using multiprocessing, load cpp_interface per process
	from cpp_interface import test_evaluate_expert
	test_evaluate_expert(*arg)

def make_dataset(states,params,df_param,testing=None):
	print('making dataset...')

	for param in params:

		# delta-uniform sampling for curriculum 
		robot_team_composition, skill_a, skill_b, env_l = sample_curriculum(param.curriculum)

		# update
		param.robot_team_composition = robot_team_composition
		param.env_l = env_l
		param.update()

		# my policy 
		param.my_policy_dict = param.policy_dict.copy()
		if param.i == 0 or param.l_mode in ["IL","DAgger"]:
			param.my_policy_dict["path_glas_model_{}".format(param.training_team)] = None  
			param.my_policy_dict["mcts_rollout_beta"] = 0.0 
		else:
			param.my_policy_dict["path_glas_model_{}".format(param.training_team)] = param.l_model_fn.format(\
				DATADIR=param.path_current_models,\
				TEAM=param.training_team,\
				ITER=param.i)

		opponents_key = "Skill_B" if param.training_team == "a" else "Skill_A"
		opponents_team = "b" if param.training_team == "a" else "a"
		param.other_policy_dicts = []
		for other_policy_skill in param.curriculum[opponents_key]:
			other_policy_dict = param.policy_dict.copy()
			if param.i == 0 or param.l_mode in ["IL","DAgger"] or other_policy_skill is None:
				other_policy_dict["path_glas_model_{}".format(opponents_team)] = None  
				other_policy_dict["mcts_rollout_beta"] = 0.0 
			else:
				other_policy_dict["path_glas_model_{}".format(opponents_team)] = param.l_model_fn.format(\
					DATADIR=param.path_current_models,\
					TEAM=opponents_team,\
					ITER=other_policy_skill)
			param.other_policy_dicts.append(other_policy_dict)

		# param.policy_dict["sim_mode"] = "MCTS" 

	if not df_param.l_parallel_on:
		from cpp_interface import evaluate_expert, test_evaluate_expert
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

def make_labelled_dataset(df_param,i):
	# labelled dataset 
	print('cleaning labelled data...')
	labelled_data_fns = df_param.l_labelled_fn.format(\
		DATADIR=df_param.path_current_data,
		TEAM='**',
		LEARNING_ITER=i,
		NUM_A='**',
		NUM_B='**',
		NUM_FILE='**')
	for file in glob.glob(labelled_data_fns+'**'):
		os.remove(file)

	print('making labelled data...')
	sim_result_fns = df_param.l_raw_fn.format(\
		DATADIR=df_param.path_current_data,
		TEAM='**',\
		LEARNING_ITER=i,
		NUM_FILE='**')	
	oa_pairs_by_size = defaultdict(list) 
	for sim_result_fn in tqdm(glob.glob(sim_result_fns+'**')): 
		sim_result = dh.load_sim_result(sim_result_fn)
		oa_pairs_by_size = make_labelled_data(sim_result,oa_pairs_by_size)

	# make actual batches and write to file 
	write_labelled_data(df_param,oa_pairs_by_size,i)
	print('labelling data completed.')
	print('dataset completed.')

def get_start(df_param,i):
	base_fn = df_param.l_raw_fn.format(
		DATADIR=df_param.path_current_data,\
		TEAM=training_team,\
		LEARNING_ITER=i,\
		NUM_FILE='*')
	start = len(glob.glob(base_fn+'*'))
	return start

def get_params(df_param,training_team,i,curriculum):

	params = []

	start = get_start(df_param,i)

	for trial in range(df_param.l_num_file_per_iteration): 

		param = Param() # random seed 

		param.curriculum = curriculum 
		param.training_team = training_team
		param.i = i 

		param.update() 

		param.dataset_fn = df_param.l_raw_fn.format(
			DATADIR=df_param.path_current_data,
			TEAM=training_team,\
			LEARNING_ITER=i,
			NUM_FILE=trial+start)

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
			test_sample = mu + 0.01 * np.random.normal(size=(2,)) 
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

def sample_curriculum(curriculum):

	# 'naive' curriculum learning 
	num_a = curriculum["NumA"][-1]
	num_b = curriculum["NumB"][-1]
	skill_a = curriculum["Skill_A"][-1]
	skill_b = curriculum["Skill_B"][-1]
	env_l = curriculum["EnvironmentLength"][-1]

	robot_team_composition = {
		'a': {'standard_robot':num_a,'evasive_robot':0},
		'b': {'standard_robot':num_b,'evasive_robot':0}
	}

	return robot_team_composition, skill_a, skill_b, env_l 

def initialCurriculum(df_param):
	curriculum = {
		'Skill_A' : [None],
		'Skill_B' : [None],
		'EnvironmentLength' : [0.5],
		'NumA' : [1],
		'NumB' : [1],
	}
	return curriculum 

def isTrainingConverged(df_param,i):
	if df_param.l_mode in ["IL"]:
		return True 
	elif df_param.l_mode in ["ExIt","MICE","DAgger"]: 
		return i >= df_param.l_num_iterations
		# return True
	else: 
		print('not recognized: ', df_param.l_mode)
		exit()

def isCurriculumConverged(df_param,curriculum,desired_game):
	return True 
	# for key, desired_game_value in desired_game.items():
	# 	if not desired_game_value in curriculum[key]:
	# 		return False
	# return True  

def incrementCurriculum(df_param,curriculum):
	# for now only increment policy skill 
	curriculum["Skill_A"].append(len(curriculum["Skill_A"]))
	curriculum["Skill_B"].append(len(curriculum["Skill_B"]))
	return curriculum 

if __name__ == '__main__':

	# parameters
	pool_count = 0
	df_param = Param() 
	df_param.clean_data_on = True
	df_param.clean_models_on = True
	df_param.make_data_on = True
	df_param.make_labelled_data_on = True
	df_param.mice_testing_on = False

	# testing 
	if df_param.mice_testing_on:
		testing = read_testing_yaml("testing/test_continuous_glas.yaml")
	else:
		testing = None 

	# prints 
	print('Learning Mode: {}'.format(df_param.l_mode))
	print('Clean old data on: {}'.format(df_param.clean_data_on))
	print('Clean old models on: {}'.format(df_param.clean_models_on))
	print('Make new data on: {}'.format(df_param.make_data_on))
	print('Testing on: {}'.format(df_param.mice_testing_on))

	# format directory 
	format_dir(df_param)

	# specify desired : for now isolate curriculum to skill of policy 
	desired_game = {
		'Skill_A' : 'a1.pt',
		'Skill_B' : 'b1.pt',
		'EnvironmentLength' : 0.5,
		'NumA' : 1,
		'NumB' : 1,
	}

	# initial curriculum 
	curriculum = initialCurriculum(df_param)

	i = 0 
	k = 0 
	# curriculum loop  
	while True: 

		# training loop 
		while True: 

			# team loop 
			for training_team in df_param.l_training_teams:

				print('k: {}, i: {}, training team: {}'.format(k,i,training_team))

				if df_param.make_data_on: 

					params = get_params(df_param,training_team,i,curriculum)

					if df_param.l_mode == "IL":
						states = get_uniform_samples(params)
					else: 
						states = get_self_play_samples(params)
					
					make_dataset(states,params,df_param,testing=testing)

				if df_param.make_labelled_data_on:
					make_labelled_dataset(df_param,i)

				# model to be trained 
				model_fn = df_param.l_model_fn.format(\
						DATADIR=df_param.path_current_models,\
						TEAM=training_team,\
						ITER=i+1)

				# data to be used 
				batched_fns = glob.glob(df_param.l_labelled_fn.format(\
						DATADIR=df_param.path_current_data,\
						TEAM=training_team,\
						LEARNING_ITER=i,\
						NUM_A='**',\
						NUM_B='**',\
						NUM_FILE='**'))

				train_model(df_param,batched_fns,training_team,model_fn)

				if df_param.mice_testing_on: 
					stats = test_model(df_param,model_fn,testing)
					plotter.plot_test_model(df_param,stats)
					plotter.save_figs('plots/model.pdf')
					plotter.open_figs('plots/model.pdf')
					exit()

			i = i + 1 

			if isTrainingConverged(df_param,i):
				curriculum = incrementCurriculum(df_param,curriculum)
				k = k + 1
				break 

		if isCurriculumConverged(df_param,curriculum,desired_game):
			break 

	print('done!')