
# standard
import os, sys, glob, shutil 
import itertools
import numpy as np 
import torch 
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multiprocessing import cpu_count, Pool 
import concurrent.futures
import tempfile 
import subprocess
from collections import defaultdict

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


def train(model,optimizer,loader):
	
	# loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	loss_func = torch.nn.CrossEntropyLoss()  
	epoch_loss = 0
	for step, (o_a,o_b,goal,action) in enumerate(loader): 
		prediction = model(o_a,o_b,goal,training=True)
		# loss = loss_func(prediction, action) 
		loss = loss_func(prediction, action.flatten()) 
		optimizer.zero_grad()   
		loss.backward()         
		optimizer.step()        
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def test(model,optimizer,loader):
	
	# loss_func = torch.nn.MSELoss()  
	loss_func = torch.nn.CrossEntropyLoss()  
	epoch_loss = 0
	for step, (o_a,o_b,goal,action) in enumerate(loader): 
		prediction = model(o_a,o_b,goal,training=True)     
		loss = loss_func(prediction, action.flatten())
		# loss = loss_func(prediction, action)
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def run_mcts_batch(param, instance_key, datadir): 

	with tempfile.TemporaryDirectory() as tmpdirname:
		input_file = tmpdirname + "/config.yaml" 
		dh.write_mcts_config_file(param, input_file)
		output_file = tmpdirname + "/output.csv"
		print('running instance {}'.format(instance_key))
		subprocess.run("../mcts/cpp/buildRelease/swarmgame -i {} -o {}".format(input_file, output_file), shell=True)
		data = np.loadtxt(output_file, delimiter=',', skiprows=1, ndmin=2, dtype=np.float32)
		sim_result = dh.convert_cpp_data_to_sim_result(data,param)

	print('writing instance {}... '.format(instance_key))
	dh.write_sim_result(sim_result,datadir + instance_key)
	print('completed instance {}'.format(instance_key))

def prepare_raw_data_gen(gparam):

	params, instance_keys  = [], []
	# for (num_nodes_A, num_nodes_B) in zip(gparam.num_nodes_A_lst,gparam.num_nodes_B_lst):
	for robot_teams in gparam.robot_team_composition_cases:
		start = 0
		for trial in range(gparam.num_trials):
			
			# param 
			param = Param()
			param.robot_teams = robot_teams 
			param.seed = int.from_bytes(os.urandom(4), sys.byteorder)
			param.controller_name = gparam.expert_controller
			param.update()
			env = Swarm(param)

			# save 
			while os.path.exists('{}{}.pickle'.format(gparam.demonstration_data_dir,get_instance_fn(param.num_nodes_A,param.num_nodes_B,start+trial))):
				start += 1 	
			instance_key = get_instance_fn(param.num_nodes_A,param.num_nodes_B,trial+start) 
			
			# assign 
			params.append(param)
			instance_keys.append(instance_key)

	return params, instance_keys


def action_to_classification(action,action_list):

	class_action = np.zeros((2))
	
	if action[0] > 0:
		class_action[0] = 1 
	elif action[0] < 0:
		class_action[0] = -1
	if action[1] > 0:
		class_action[1] = 1 
	elif action[1] < 0:
		class_action[1] = -1

	not_found = True
	for k,candidate_action in enumerate(action_list): 
		if np.allclose(candidate_action.flatten(),class_action.flatten()):
			not_found = False
			break 

	if not_found: 
		print('action {} not found in {}!'.format(action,action_list))
		print('u_max',u_max)
		exit()

	return k

def load_param(some_dict):
	param = Param()
	param.from_dict(some_dict)
	return param 


def get_instance_fn(num_nodes_A,num_nodes_B,trial):
	fn = '{}a_{}b_{}trial'.format(num_nodes_A,num_nodes_B,trial)
	return fn 


def get_batch_fn(datadir,team,num_a,num_b,batch_num):
	return '{}labelled_{}team_{}a_{}b_{}trial.npy'.format(datadir,team,num_a,num_b,batch_num)


def get_dbg_observation_fn(datadir,instance,team,num_a,num_b):
	return '{}observations_from_{}_{}team_{}a_{}b.npy'.format(datadir,instance,team,num_a,num_b)


def get_instance_keys(gparam):
	instance_keys = [] 
	for instance_key in glob.glob('{}*.pickle'.format(gparam.demonstration_data_dir)):
		instance_key = os.path.basename(instance_key)
		instance_key = instance_key.split('.pickle')[0]
		instance_keys.append(instance_key)
	return instance_keys	


if __name__ == '__main__':

	gparam = Gparam()

	# run expert and write (state, action) pairs into files 
	if gparam.make_raw_data_on:
		print('making raw data...')

		if gparam.clean_raw_data_on: 
			print('cleaning training data...')
			shutil.rmtree(gparam.demonstration_data_dir)
			os.makedirs(gparam.demonstration_data_dir)
		
		params, instance_keys = prepare_raw_data_gen(gparam) 
		if gparam.serial_on:
			for (param, instance_key) in zip(params, instance_keys):
				run_mcts_batch(param, instance_key, gparam.demonstration_data_dir)
		else:	
			ncpu = cpu_count()
			print('ncpu: ', ncpu)
			with Pool(ncpu-1) as p:
				p.starmap(run_mcts_batch, zip(params,instance_keys,itertools.repeat(gparam.demonstration_data_dir)))

	# load (state,action) files, apply measurement model, and write (observation,action) binary files
	if gparam.make_labelled_data_on: 
		print('make labelled data...')

		# read instances into observation-action directory batched by number neighbors team_a, team_b 
		oa_pairs_by_size = defaultdict(list) 
		oa_pairs_by_file = defaultdict(lambda: defaultdict(list)) 
		instance_keys = get_instance_keys(gparam) 

		for instance_key in instance_keys: 
			# print('\t instance_key:',instance_key)

			sim_result = dh.load_sim_result(gparam.demonstration_data_dir+instance_key+'.pickle')
			param = load_param(sim_result["param"])
			states = sim_result["states"] # nt x nrobots x nstate_per_robot
			actions = sim_result["actions"] 

			for robot in param.robots:
				robot["r_sense"] = gparam.r_sense

			for timestep,(state,action) in enumerate(zip(states,actions)):
				
				for robot_idx in range(state.shape[0]):
					o_a, o_b, goal = relative_state(state,param,robot_idx)
					action_per_robot = np.expand_dims(action[robot_idx,:],axis=1) # (2,1)

					if gparam.discrete_on: 
						action_per_robot = action_to_classification(action_per_robot, gparam.actions)

					if robot_idx < param.num_nodes_A: 
						team = "a"
					else: 
						team = "b" 

					key = (team,len(o_a),len(o_b))
					oa_pairs_by_size[key].append((o_a, o_b, goal, action_per_robot))
					oa_pairs_by_file[instance_key][key].append((o_a, o_b, goal, action_per_robot))

		# make dbg batches and write to file 
		# for instance_key, oa_pairs_by_size_dbg in oa_pairs_by_file.items():
		# 	for (team,num_a,num_b),oa_pairs in oa_pairs_by_size_dbg.items():
		# 		batched_dataset = [] 
		# 		for (o_a, o_b, goal, action) in oa_pairs:
		# 			data = np.concatenate((np.array(o_a).flatten(),np.array(o_b).flatten(),np.array(goal).flatten(),np.array(action).flatten()))
		# 			batched_dataset.append(data)
		# 		batch_fn = get_dbg_observation_fn(gparam.demonstration_data_dir,instance_key,team,num_a,num_b)
		# 		dh.write_oa_batch(batched_dataset,batch_fn) 

		# make actual batches and write to file 
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

	# check data
	if gparam.dbg_vis_on:
		print('vis...')

		num_plots = 10
		instance_keys = get_instance_keys(gparam) 

		# check state action pairs 
		for count, instance_key in enumerate(instance_keys):
			sim_result = dh.load_sim_result(gparam.demonstration_data_dir+instance_key+'.pickle')
			plotter.plot_tree_results(sim_result)

			if count > num_plots:
				break 

		# check observation mapping 
		for count, instance_key in enumerate(instance_keys):
			sim_result = dh.load_sim_result(gparam.demonstration_data_dir+instance_key+'.pickle')

			observations_list = [] 

			batch_fns = glob.glob(get_dbg_observation_fn(gparam.demonstration_data_dir,instance_key,'*','*','*'))
			for batch_fn in batch_fns: 
				observations_list.append(dh.read_dbg_observation_fn(batch_fn,gparam.demonstration_data_dir))

			plotter.plot_dbg_observations(sim_result,observations_list)
			
			if count > num_plots:
				break 

		# todo 

		# check action classification 
		# todo 


	# load (observation,action) binary files, train a model, and write model to file 
	if gparam.train_model_on: 

		for training_team in gparam.training_teams:

			print('training model for team {}...'.format(training_team))
		
			batched_files = glob.glob(get_batch_fn(gparam.demonstration_data_dir,training_team,'*','*','*'))

			n_points = 0 
			for batched_file in batched_files:
				o_a,o_b,goal,action = dh.read_oa_batch(batched_file,gparam.demonstration_data_dir)
				n_points += action.shape[0]
			n_points = np.min((n_points, gparam.il_n_points))
			print('n_points',n_points)

			# get loader 
			train_loader = [] # lst of batches 
			test_loader  = [] 
			curr_points, train_dataset_size, test_dataset_size = 0,0,0
			for batched_file in batched_files: 
				o_a,o_b,goal,action = dh.read_oa_batch(batched_file,gparam.demonstration_data_dir)
				if curr_points < gparam.il_test_train_ratio * n_points: 
					train_loader.append([
						torch.from_numpy(o_a).float().to(gparam.device),
						torch.from_numpy(o_b).float().to(gparam.device),
						torch.from_numpy(goal).float().to(gparam.device),
						# torch.from_numpy(action).float().to(gparam.device)])
						torch.from_numpy(action).type(torch.long).to(gparam.device)])
					train_dataset_size += action.shape[0]

				elif curr_points < n_points:
					test_loader.append([
						torch.from_numpy(o_a).float().to(gparam.device),
						torch.from_numpy(o_b).float().to(gparam.device),
						torch.from_numpy(goal).float().to(gparam.device),
						# torch.from_numpy(action).float().to(gparam.device)])
						torch.from_numpy(action).type(torch.long).to(gparam.device)])
					test_dataset_size += action.shape[0]

				curr_points += action.shape[0]

			print('train dataset size: ', train_dataset_size)
			print('test dataset size: ', test_dataset_size)

			# init model
			if gparam.discrete_on:
				model = DiscreteEmptyNet(gparam,gparam.device)
			else:
				model = EmptyNet(gparam,gparam.device)

			# init optimizer
			optimizer = torch.optim.Adam(model.parameters(), lr=gparam.il_lr, weight_decay=gparam.il_wd)
			
			# train 
			losses = []
			il_train_model_fn = gparam.il_train_model_fn.format(training_team)
			with open(il_train_model_fn + ".csv", 'w') as log_file:
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
							torch.save(model.state_dict(), il_train_model_fn)
							model.to(gparam.device)

					log_file.write("{},{},{},{}\n".format(time.time() - start_time, epoch, train_epoch_loss, test_epoch_loss))

			plotter.plot_loss(losses,training_team)

	if plotter.has_figs():
		plotter.save_figs('plots.pdf')
		plotter.open_figs('plots.pdf')
