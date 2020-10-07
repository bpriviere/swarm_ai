
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
# from multiprocessing import cpu_count, Pool, freeze_support, Queue
import multiprocessing as mp
from queue import Empty
from tqdm import tqdm
import itertools
import yaml 
import random
import pickle 

# custom 
from testing.test_continuous_glas import test_model, test_evaluate_expert_wrapper, read_testing_yaml, test_evaluate_expert
import plotter
import datahandler as dh
from param import Param 
# from learning.discrete_emptynet import DiscreteEmptyNet
from learning.continuous_emptynet import ContinuousEmptyNet
from learning.gaussian_emptynet import GaussianEmptyNet
from learning_interface import format_data, global_to_local 

def my_loss(value, policy, target_value, target_policy, weight, mu, logvar, l_subsample_on, l_gaussian_on):
	# value \& policy network : https://www.nature.com/articles/nature24270
	# for kl loss : https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
	# for wmse : https://stackoverflow.com/questions/57004498/weighted-mse-loss-in-pytorch

	if l_gaussian_on: 
		# train distribution parameters, mean and variance where target_policy is mean and weight is variance 
		criterion = nn.MSELoss(reduction='none')
		action_dim = 2 
		mse = torch.sum((criterion(value, target_value) + \
			criterion(mu, target_policy) + criterion(torch.exp(logvar),weight)))
		loss = mse / value.shape[0]

	else:
		if l_subsample_on:
			criterion = nn.MSELoss(reduction='sum')
			mse = criterion(policy,target_policy) + criterion(value,target_value)
			# kldiv = 0.5 * torch.sum(- 1 - torch.log(sd.pow(2)) + mu.pow(2) + sd.pow(2)) 
			kldiv = 0.5 * torch.sum(- 1 - logvar + mu.pow(2) + torch.exp(logvar)) 
		else: 
			criterion = nn.MSELoss(reduction='none')
			# mse = torch.sum(weight*criterion(value, target_value) + weight*criterion(policy, target_policy))
			mse = torch.sum(weight*(criterion(value, target_value) + criterion(policy, target_policy)))
			# kldiv = 0.5 * torch.sum( weight * (- 1 - torch.log(sd.pow(2)) + mu.pow(2) + sd.pow(2))) 
			kldiv = 0.5 * torch.sum( weight * (- 1 - logvar + mu.pow(2) + torch.exp(logvar))) 

		kld_weight = 1e-4
		loss = mse + kld_weight * kldiv
		loss = loss / value.shape[0]

	return loss


def train(model,optimizer,loader,l_subsample_on,l_gaussian_on,l_sync_every,epoch, scheduler=None):

	epoch_loss = 0
	for step, (o_a,o_b,goal,target_value,target_policy,weight) in enumerate(loader): 

		if step % l_sync_every == 0:
			model.require_backward_grad_sync = True
			model.require_forward_param_sync = True
		else:
			model.require_backward_grad_sync = False
			model.require_forward_param_sync = False

		if l_gaussian_on: 
			value, policy, mu, logvar = model(o_a,o_b,goal,training=True)
		else:
			value, policy, mu, logvar = model(o_a,o_b,goal,x=target_policy)

		loss = my_loss(value, policy, target_value, target_policy, weight, mu, logvar, l_subsample_on, l_gaussian_on)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if scheduler is not None:
			scheduler.step(epoch + step/len(loader))
		epoch_loss += float(loss)

		if torch.isnan(loss).any():
			print('WARNING: NAN FOUND IN TRAIN')
			if torch.isnan(o_a).any():
				print(' in o_a')
			if torch.isnan(o_b).any():
				print(' in o_b')
			if torch.isnan(goal).any():
				print(' in goal')
			if torch.isnan(target_value).any():
				print(' in target_value')
			if torch.isnan(target_policy).any():
				print(' in target_policy')
			if torch.isnan(weight).any():
				print(' in weight')
			break

	return epoch_loss


def test(model,loader,l_subsample_on,l_gaussian_on):
	epoch_loss = 0
	with torch.no_grad():
		for o_a,o_b,goal,target_value,target_policy,weight in loader:
			if l_gaussian_on: 
				value, policy, mu, logvar = model(o_a,o_b,goal,training=True)
			else:
				value, policy, mu, logvar = model(o_a,o_b,goal,x=target_policy)
			loss = my_loss(value, policy, target_value, target_policy, weight, mu, logvar, l_subsample_on, l_gaussian_on)
			epoch_loss += float(loss)

			if torch.isnan(loss).any():
				print('WARNING: NAN FOUND IN TEST')
				if torch.isnan(o_a).any():
					print(' in o_a')
				if torch.isnan(o_b).any():
					print(' in o_b')
				if torch.isnan(goal).any():
					print(' in goal')
				if torch.isnan(target_value).any():
					print(' in target_value')
				if torch.isnan(target_policy).any():
					print(' in target_policy')
				if torch.isnan(weight).any():
					print(' in weight')
				break

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

		if param.i == 0:
			path_glas_model_a = None
			path_glas_model_b = None
		else: 
			if param.training_team == "a" : 
				path_glas_model_a = param.l_model_fn.format(\
							DATADIR=param.path_current_models,\
							TEAM="a",\
							ITER=param.i)
				if skill_b is None: 
					path_glas_model_b = None
				else: 
					path_glas_model_b = param.l_model_fn.format(\
								DATADIR=param.path_current_models,\
								TEAM="b",\
								ITER=skill_b)

			elif param.training_team == "b" : 
				path_glas_model_b = param.l_model_fn.format(\
							DATADIR=param.path_current_models,\
							TEAM="b",\
							ITER=param.i)
				if skill_a is None: 
					path_glas_model_a = None
				else: 
					path_glas_model_a = param.l_model_fn.format(\
								DATADIR=param.path_current_models,\
								TEAM="a",\
								ITER=skill_a)

		param.policy_dict_a = {
			'sim_mode' : 				"D_MCTS", 
			'path_glas_model_a' : 		path_glas_model_a, 	
			'path_glas_model_b' : 		path_glas_model_b, 	
			'mcts_tree_size' : 			param.l_num_learner_nodes,
			'mcts_rollout_beta' : 		param.l_mcts_rollout_beta,
			'mcts_c_param' : 			1.4,
			'mcts_pw_C' : 				1.0,
			'mcts_pw_alpha' : 			0.25,
			'mcts_vf_beta' : 			0.0,
		}
		param.policy_dict_b = param.policy_dict_a.copy() 


	# print policies 
	print('self-play policies...')
	for param in params: 
		print('param.policy_dict_a: ',param.policy_dict_a)
		print('param.policy_dict_b: ',param.policy_dict_b)


	# get self play states 
	if not df_param.l_parallel_on:
		for param in params: 
			instance_self_play(0, None, param)

	else:
		ncpu = mp.cpu_count()
		print('ncpu: ', ncpu)
		num_workers = min(ncpu-1, len(params))
		with mp.Pool(num_workers) as pool:
			manager = mp.Manager()
			queue = manager.Queue()
			args = list(zip(itertools.count(), itertools.repeat(queue), itertools.repeat(len(params)), params))
			for _ in pool.imap_unordered(instance_self_play_wrapper, args):
				pass

	self_play_states = [] 
	for param in params: 
		fn = '{}/states_for_{}.pickle'.format(\
			os.path.dirname(param.dataset_fn),os.path.basename(param.dataset_fn))
		with open(fn, 'rb') as h:
			states_per_file = pickle.load(h)
		self_play_states.append(states_per_file[0:param.l_num_points_per_file]) 
	

	# for param in params: 
	# 	states_per_file = []
	# 	remaining_plots_per_file = 2
	# 	while len(states_per_file) < param.l_num_points_per_file:
	# 		param.state = param.make_initial_condition()
	# 		# sim_result = self_play(param,deterministic=False)
	# 		sim_result = play_game(param,param.policy_dict_a,param.policy_dict_b,deterministic=False)

	# 		# clean data
	# 		idxs = np.logical_not(np.isnan(sim_result["states"]).any(axis=2).any(axis=1))
	# 		sim_result["states"] = sim_result["states"][idxs]
	# 		sim_result["actions"] = sim_result["actions"][idxs]
	# 		sim_result["times"] = sim_result["times"][idxs]
	# 		sim_result["rewards"] = sim_result["rewards"][idxs]

	# 		if remaining_plots_per_file > 0:
	# 			title = policy_title(param.policy_dict_a,"a") + " vs " + policy_title(param.policy_dict_b,"b")
	# 			plotter.plot_tree_results(sim_result, title)
	# 			remaining_plots_per_file -= 1

	# 		if np.isnan(sim_result["states"]).any(axis=2).any(axis=1).any():
	# 			print('WARNING: NANS found in self-play states')
	# 			plotter.plot_tree_results(sim_result, title)
	# 			plotter.save_figs('../current/models/{}{}_nans.pdf'.format(params[0].training_team, params[0].i+1))
	# 			exit()

	# 		states_per_file.extend(sim_result["states"])
	# 	self_play_states.append(states_per_file[0:param.l_num_points_per_file])

	# plotter.save_figs('../current/models/{}{}_self_play_samples.pdf'.format(params[0].training_team, params[0].i+1))

	plotter.merge_figs(glob.glob('../current/models/temp_**'),\
		'../current/models/{}{}_self_play_samples.pdf'.format(params[0].training_team, params[0].i+1))
	print('self-play sample collection completed.')

	return self_play_states

def instance_self_play_wrapper(arg):
	instance_self_play(*arg)

def instance_self_play(rank, queue, total, param):
	# When using multiprocessing, load cpp_interface per process
	from cpp_interface import play_game

	if rank == 0:
		pbar = tqdm(total=param.l_num_points_per_file*total)

	# print('starting self-play states {}'.format(param.dataset_fn))
	states_per_file = []
	remaining_plots_per_file = 2
	while len(states_per_file) < param.l_num_points_per_file:
		param.state = param.make_initial_condition()
		# sim_result = self_play(param,deterministic=False)
		sim_result = play_game(param,param.policy_dict_a,param.policy_dict_b,deterministic=False)

		# clean data
		# idxs = np.logical_not(np.isnan(sim_result["states"]).any(axis=2).any(axis=1))
		# sim_result["states"] = sim_result["states"][idxs]
		# sim_result["actions"] = sim_result["actions"][idxs]
		# sim_result["times"] = sim_result["times"][idxs]
		# sim_result["rewards"] = sim_result["rewards"][idxs]

		if remaining_plots_per_file > 0:
			title = policy_title(param.policy_dict_a,"a") + " vs " + policy_title(param.policy_dict_b,"b")
			plotter.plot_tree_results(sim_result, title)
			remaining_plots_per_file -= 1
		
		states_per_file.extend(sim_result["states"])

		# update status
		if rank == 0:
			count = len(sim_result["states"])
			try:
				while True:
					count += queue.get_nowait()
			except Empty:
				pass
			pbar.update(count)
		else:
			queue.put_nowait(len(sim_result["states"]))

	# plot figs 
	plotter.save_figs('../current/models/temp_{}.pdf'.format(os.path.basename(param.dataset_fn)))
	
	# write data
	fn = '{}/states_for_{}.pickle'.format(\
		os.path.dirname(param.dataset_fn),os.path.basename(param.dataset_fn)) 
	with open(fn, 'wb') as h:
		pickle.dump(states_per_file, h)	
	# print('completed self-play states {}'.format(param.dataset_fn))


def policy_title(policy_dict,team):
	title = policy_dict["sim_mode"] 
	if policy_dict["sim_mode"] in ["D_MCTS","MCTS","GLAS"]:
		title += " {}".format(policy_dict['path_glas_model_{}'.format(team)])
	return title 

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
			
			if np.isnan(state[robot_idx,:]).any(): # non active robot 
				continue

			o_a, o_b, goal = global_to_local(state,param,robot_idx)
			key = (param.training_team,len(o_a),len(o_b))

			for action, weight in zip(policy_dist[robot_idx][:,0],policy_dist[robot_idx][:,1]):

				if not (np.isnan(action).any() or np.isnan(weight).any()):
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

		o_a,o_b,goal,value,action,weight = dh.read_oa_batch(batched_file,df_param.l_gaussian_on)

		if df_param.l_gaussian_on: 
			data = [
				torch.from_numpy(o_a).float().to(df_param.device),
				torch.from_numpy(o_b).float().to(df_param.device),
				torch.from_numpy(goal).float().to(df_param.device),
				torch.from_numpy(value).float().to(df_param.device).unsqueeze(1),
				torch.from_numpy(action).float().to(df_param.device),
				torch.from_numpy(weight).float().to(df_param.device),
				]

		else: 
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

	if df_param.l_gaussian_on: 
		single_model = GaussianEmptyNet(df_param,df_param.device)
	else:
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

		train_epoch_loss = train(model,optimizer,train_loader,df_param.l_subsample_on,df_param.l_gaussian_on,df_param.l_sync_every,epoch,train_scheduler)
		test_epoch_loss = test(model,test_loader,df_param.l_subsample_on,df_param.l_gaussian_on)

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
		plotter.save_figs("../current/models/{}_losses.pdf".format(os.path.basename(model_fn).split('.')[0]))

		# plotter.open_figs('plots/model.pdf')

		print("visualize training data and model distribution...")
		plotter.plot_training(df_param,batched_files,model_fn)
		plotter.save_figs("../current/models/{}_dist_vis.pdf".format(os.path.basename(model_fn).split('.')[0]))

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

def make_dataset(states,params,df_param,testing=None):
	print('making dataset...')

	for param in params:

		# delta-uniform sampling for curriculum 
		robot_team_composition, skill_a, skill_b, env_l = sample_curriculum(param.curriculum)

		# update
		param.robot_team_composition = robot_team_composition
		param.env_l = env_l
		param.update()

		# imitate expert policy 
		expert_policy_dict = {
			'sim_mode' : 				"MCTS", 
			'path_glas_model_a' : 		None, 	
			'path_glas_model_b' : 		None, 	
			'mcts_tree_size' : 			param.l_num_expert_nodes,
			'mcts_rollout_beta' : 		0.0,
			'mcts_c_param' : 			1.4,
			'mcts_pw_C' : 				1.0,
			'mcts_pw_alpha' : 			0.25,
			'mcts_vf_beta' : 			0.0,
		}

		# my policy 
		param.my_policy_dict = expert_policy_dict.copy()
		if param.i == 0 or param.l_mode in ["IL","DAgger"]:
			param.my_policy_dict["path_glas_model_{}".format(param.training_team)] = None  
			param.my_policy_dict["mcts_rollout_beta"] = 0.0 
		else:
			param.my_policy_dict["path_glas_model_{}".format(param.training_team)] = param.l_model_fn.format(\
				DATADIR=param.path_current_models,\
				TEAM=param.training_team,\
				ITER=param.i)
			param.my_policy_dict["mcts_rollout_beta"] = param.l_mcts_rollout_beta 

		opponents_key = "Skill_B" if param.training_team == "a" else "Skill_A"
		opponents_team = "b" if param.training_team == "a" else "a"
		param.other_policy_dicts = []
		for other_policy_skill in param.curriculum[opponents_key]:
			other_policy_dict = expert_policy_dict.copy()
			if param.i == 0 or param.l_mode in ["IL","DAgger"] or other_policy_skill is None:
				other_policy_dict["path_glas_model_{}".format(opponents_team)] = None  
				other_policy_dict["mcts_rollout_beta"] = 0.0 
			else:
				other_policy_dict["path_glas_model_{}".format(opponents_team)] = param.l_model_fn.format(\
					DATADIR=param.path_current_models,\
					TEAM=opponents_team,\
					ITER=other_policy_skill)
			param.other_policy_dicts.append(other_policy_dict)

		print('evaluate-expert policies...')
		print('param.my_policy_dict: ',param.my_policy_dict)
		print('param.other_policy_dicts: ',param.other_policy_dicts)


		# param.policy_dict["sim_mode"] = "MCTS" 

	if not df_param.l_parallel_on:
		from cpp_interface import evaluate_expert
		if df_param.mice_testing_on:
			for states_per_file, param in zip(states, params): 
				test_evaluate_expert(states_per_file,param,testing,quiet_on=True,progress=None) 				
		else:
			for states_per_file, param in zip(states, params): 
				evaluate_expert(states_per_file, param, quiet_on=False)
	else:
		ncpu = mp.cpu_count()
		print('ncpu: ', ncpu)
		num_workers = min(ncpu-1, len(params))
		with mp.Pool(num_workers) as p:
			manager = mp.Manager()
			queue = manager.Queue()
			total = sum([len(states_per_file) for states_per_file in states])
			if df_param.mice_testing_on:
				args = list(zip(itertools.count(), itertools.repeat(queue), itertools.repeat(total), states, params, itertools.repeat(testing)))
				for _ in p.imap_unordered(test_evaluate_expert_wrapper, args):
					pass
			else:
				args = list(zip(itertools.count(), itertools.repeat(queue), itertools.repeat(total), states, params))
				for _ in p.imap_unordered(evaluate_expert_wrapper, args):
					pass
			# p.starmap(evaluate_expert, states, params)
			# p.starmap(evaluate_expert, list(zip(states, params)))

def make_labelled_dataset(df_param,i):

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
			for file in glob.glob(datadir + "/*"):
				os.remove(file)
		os.makedirs(datadir,exist_ok=True)

	if df_param.clean_models_on:
		modeldir = df_param.path_current_models
		if os.path.exists(modeldir):
			for file in glob.glob(modeldir + "/*"):
				os.remove(file)
		os.makedirs(modeldir,exist_ok=True)	

def sample_curriculum(curriculum):

	mode = "naive"

	if mode == "naive": 
		# 'naive' curriculum learning 
		num_a = curriculum["NumA"][-1]
		num_b = curriculum["NumB"][-1]
		skill_a = curriculum["Skill_A"][-1]
		skill_b = curriculum["Skill_B"][-1]
		env_l = curriculum["EnvironmentLength"][-1]

	elif mode == "uniform":
		num_a = curriculum["NumA"][np.random.randint(len(curriculum["NumA"]))]
		num_b = curriculum["NumB"][np.random.randint(len(curriculum["NumB"]))]
		skill_a = curriculum["Skill_A"][np.random.randint(len(curriculum["Skill_A"]))]
		skill_b = curriculum["Skill_B"][np.random.randint(len(curriculum["Skill_B"]))]
		env_l = curriculum["EnvironmentLength"][np.random.randint(len(curriculum["EnvironmentLength"]))]

	robot_team_composition = {
		'a': {'standard_robot':num_a,'evasive_robot':0},
		'b': {'standard_robot':num_b,'evasive_robot':0}
	}

	return robot_team_composition, skill_a, skill_b, env_l 

def initialCurriculum(df_param):
	curriculum = {
		'Skill_A' : [None],
		'Skill_B' : [None],
		'EnvironmentLength' : [df_param.l_env_l0],
		'NumA' : [1],
		'NumB' : [1],
	}
	return curriculum 

def isTrainingConverged(df_param,i,k):
	return i >= df_param.l_num_iterations + k * df_param.l_num_iterations
	# if df_param.l_mode in ["IL"]:
	# 	return True 
	# elif df_param.l_mode in ["ExIt","MICE","DAgger"]: 
	# 	return i >= df_param.l_num_iterations
	# 	# return True
	# else: 
	# 	print('not recognized: ', df_param.l_mode)
	# 	exit()

def isCurriculumConverged(df_param,curriculum,desired_game):
	return desired_game["EnvironmentLength"] in curriculum["EnvironmentLength"]
	# return True 
	# for key, desired_game_value in desired_game.items():
	# 	if not desired_game_value in curriculum[key]:
	# 		return False
	# return True  

def incrementCurriculum(df_param,curriculum,desired_game):
	
	done = isCurriculumConverged(df_param,curriculum,desired_game)

	if done: 
		return curriculum, done 

	else: 
		# if curriculum["Skill_A"] < desired_game["Skill_A"] : 
		# 	curriculum["Skill_A"].append(len(curriculum["Skill_A"]))
		# if curriculum["Skill_B"] < desired_game["Skill_B"] : 
		# 	curriculum["Skill_B"].append(len(curriculum["Skill_B"]))
		if not desired_game["EnvironmentLength"] in curriculum["EnvironmentLength"]: 
			curriculum["EnvironmentLength"].append(curriculum["EnvironmentLength"][-1] + df_param.l_env_dl)
		return curriculum , done 

if __name__ == '__main__':

	# parameters
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
		'EnvironmentLength' : 1.0,
		'NumA' : 1,
		'NumB' : 1,
	}

	# initial curriculum 
	curriculum = initialCurriculum(df_param)
	print('\n\n -------------- {} curriculum: {} -------------- \n\n'.format(0,curriculum))	

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
						# LEARNING_ITER='**',\
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

			if isTrainingConverged(df_param,i,k):
				curriculum, curriculumDone = incrementCurriculum(df_param,curriculum,desired_game)
				k = k + 1
				print('\n\n -------------- {} curriculum: {} -------------- \n\n'.format(k,curriculum))
				break 

		# if isCurriculumConverged(df_param,curriculum,desired_game):
		if curriculumDone:
			break 

	print('done!')