

import os,sys,glob,shutil
import numpy as np 
import time 
import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from itertools import repeat 
from multiprocessing import cpu_count, Pool
from tqdm import tqdm 

import plotter
import datahandler as dh
from cpp_interface import evaluate_expert
from param import Param 
from learning.discrete_emptynet import DiscreteEmptyNet


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


def uniform_sample(param,n):
	states = [] 
	for _ in range(n):
		param.make_initial_condition() 
		states.append(param.state)
	return states


def get_uniform_samples(params):
	print('getting uniform samples...')
	states = []
	for param in params: 
		states.append(uniform_sample(param,param.l_num_points_per_file))
	print('uniform samples collection completed.')
	return states, params


def get_self_play_samples(gparam,params):
	exit('not implented/tested')
	print('getting self-play samples...')
	states = []
	for param in params: 
		sim_result = rollout(param)
		states.append(sim_result["states"])
	print('self-play sample collection completed.')
	return states, params 
	

def increment(gparam):
	exit('not implemented')


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


def write_labelled_data(df_param,oa_pairs_by_size):

	for (team, num_a, num_b), oa_pairs in oa_pairs_by_size.items():
		batch_num = 0 
		batched_dataset = [] 
		for (o_a, o_b, goal, action) in oa_pairs:
			data = np.concatenate((np.array(o_a).flatten(),np.array(o_b).flatten(),np.array(goal).flatten(),np.array(action).flatten()))
			batched_dataset.append(data)
			if len(batched_dataset) >= df_param.l_batch_size:
				batch_fn = df_param.l_labelled_fn.format(DATADIR=df_param.path_current_data,TEAM=team,NUM_A=num_a,NUM_B=num_b,IDX_TRIAL=batch_num)
				dh.write_oa_batch(batched_dataset,batch_fn) 
				batch_num += 1 
				batched_dataset = [] 
		if len(batched_dataset) > 0:
			batch_fn = df_param.l_labelled_fn.format(DATADIR=df_param.path_current_data,TEAM=team,NUM_A=num_a,NUM_B=num_b,IDX_TRIAL=batch_num)
			dh.write_oa_batch(batched_dataset,batch_fn) 	


def get_dataset_size(df_param,batched_files):
	n_points = 0 
	for batched_file in batched_files:
		# print('batched_file',batched_file)
		o_a,o_b,goal,action = dh.read_oa_batch(batched_file)
		n_points += action.shape[0]
	n_points = np.min((n_points, df_param.l_max_dataset_size))
	return n_points


def make_loaders(df_param,batched_files,n_points):

	train_loader = [] # lst of batches 
	test_loader = [] 
	curr_points, train_dataset_size, test_dataset_size = 0,0,0
	for batched_file in batched_files: 
		o_a,o_b,goal,action = dh.read_oa_batch(batched_file)
		if curr_points < df_param.l_test_train_ratio * n_points: 
			train_loader.append([
				torch.from_numpy(o_a).float().to(df_param.device),
				torch.from_numpy(o_b).float().to(df_param.device),
				torch.from_numpy(goal).float().to(df_param.device),
				torch.from_numpy(action).float().to(df_param.device)])
				# torch.from_numpy(action).type(torch.long).to(df_param.device)])
			train_dataset_size += action.shape[0]

		elif curr_points < n_points:
			test_loader.append([
				torch.from_numpy(o_a).float().to(df_param.device),
				torch.from_numpy(o_b).float().to(df_param.device),
				torch.from_numpy(goal).float().to(df_param.device),
				torch.from_numpy(action).float().to(df_param.device)]) # for others
				# torch.from_numpy(action).type(torch.long).to(df_param.device)]) # for cross entopy
			test_dataset_size += action.shape[0]
		curr_points += action.shape[0]	

	return train_loader,test_loader, train_dataset_size, test_dataset_size


def train_model(df_param,batched_files,training_team,model_fn):

	n_points = get_dataset_size(df_param,batched_files)
	train_loader,test_loader,train_dataset_size,test_dataset_size = make_loaders(df_param,batched_files,n_points)

	print('train dataset size: ', train_dataset_size)
	print('test dataset size: ', test_dataset_size)

	model = DiscreteEmptyNet(df_param,df_param.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=df_param.l_lr, weight_decay=df_param.l_wd)
	
	# train 
	losses = []
	with open(model_fn + ".csv", 'w') as log_file:
		log_file.write("time,epoch,train_loss,test_loss\n")
		start_time = time.time()
		best_test_loss = np.Inf
		scheduler = ReduceLROnPlateau(optimizer, 'min')
		for epoch in range(1,df_param.l_n_epoch+1):
			train_epoch_loss = train(model,optimizer,train_loader)
			test_epoch_loss = test(model,optimizer,test_loader)
			scheduler.step(test_epoch_loss)
			losses.append((train_epoch_loss,test_epoch_loss))
			if epoch%df_param.l_log_interval==0:
				print('epoch: ', epoch)
				print('   Train Epoch Loss: ', train_epoch_loss)
				print('   Test Epoch Loss: ', test_epoch_loss)
				if test_epoch_loss < best_test_loss:
					best_test_loss = test_epoch_loss
					print('      saving @ best test loss:', best_test_loss)
					torch.save(model.state_dict(), model_fn)
					model.to(df_param.device)
			log_file.write("{},{},{},{}\n".format(time.time() - start_time, epoch, train_epoch_loss, test_epoch_loss))
	plotter.plot_loss(losses,training_team)


def load_param(some_dict):
	param = Param()
	param.from_dict(some_dict)
	return param 


def make_dataset(states,params,df_param):
	print('making dataset...')

	if not df_param.l_parallel_on:
		for states_per_file, param in zip(states, params): 
			evaluate_expert(states_per_file, param)
	else:
		ncpu = cpu_count()
		print('ncpu: ', ncpu)
		with Pool(ncpu-1) as p:
			# p.starmap(evaluate_expert, states, params)
			p.starmap(evaluate_expert, list(zip(states, params)))


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

		for trial in range(df_param.l_num_trials_per_iteration): 

			param = Param()
			param.seed = int.from_bytes(os.urandom(4), sys.byteorder)
			param.robot_team_composition = robot_team_composition 
			param.mcts_tree_size = df_param.mcts_tree_size
			param.l_num_points_per_file = df_param.l_num_points_per_file

			if df_param.l_mode == "DAgger" or df_param.l_mode == "IL":
				param.mcts_rollout_beta = 0.0
				param.mcts_mode = "MCTS_RANDOM"
			elif df_param.l_mode == "ExIt" or df_param.l_mode == "Mice":
				param.mcts_rollout_beta = df_param.mcts_rollout_beta
				param.mcts_mode = "MCTS_GLAS"
			else: 
				exit('df_param.mode not recognized')

			param.training_team = training_team
			param.iter_i = iter_i 
			param.glas_model_A = df_param.l_model_fn.format(DATADIR=df_param.path_current_data,TEAM="a",ITER=iter_i)
			param.glas_model_B = df_param.l_model_fn.format(DATADIR=df_param.path_current_data,TEAM="b",ITER=iter_i)
			param.update() 
			param.dataset_fn = df_param.l_raw_fn.format(DATADIR=df_param.path_current_data,TEAM=training_team,\
				NUM_A=param.num_nodes_A,NUM_B=param.num_nodes_B,IDX_TRIAL=trial+start)

			params.append(param)

	return params	


def format_dir(df_param):
	datadir = df_param.path_current_data
	if os.path.exists(datadir):
		for file in glob.glob(datadir + "/*"):
			os.remove(file)
	os.makedirs(datadir,exist_ok=True)


if __name__ == '__main__':

	df_param = Param() 
	format_dir(df_param)
	
	for iter_i in range(df_param.l_num_iterations):
		for training_team in df_param.l_training_teams: 

			print('iter: {}/{}, training team: {}'.format(iter_i,df_param.l_num_iterations,training_team))

			params = get_params(df_param,training_team,iter_i)
			
			if iter_i == 0 or df_param.l_mode == "IL":
				states, params = get_uniform_samples(params)
			else: 
				states, params = get_self_play_samples(params) 
			
			make_dataset(states,params,df_param)
			train_model(df_param,\
				glob.glob(df_param.l_labelled_fn.format(\
					DATADIR=df_param.path_current_data,NUM_A='**',NUM_B='**',IDX_TRIAL='**',TEAM=training_team,ITER='**')), \
				training_team,\
				df_param.l_model_fn.format(\
					DATADIR=df_param.path_current_models,TEAM=training_team,ITER=iter_i))

			if df_param.l_mode == "Mice":
				increment(df_param)
			