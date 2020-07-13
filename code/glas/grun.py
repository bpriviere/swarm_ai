
# standard
import os, sys, glob
import itertools
import numpy as np 
import torch 
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multiprocessing import cpu_count, Pool 

# project
sys.path.append("../")
from env import Swarm
from run import run_sim
from param import Param 
from gparam import Gparam
from learning.emptynet import EmptyNet
from utilities import load_module
import datahandler as dh



def format_dir(param):
	if not os.path.exists(param.demonstration_data_dir):
		os.makedirs(param.demonstration_data_dir,exist_ok=True)


def train(model,optimizer,loader):
	
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (o_a,o_b,action) in enumerate(loader): 
		prediction = model(o_a,o_b)     
		loss = loss_func(prediction, action) 
		optimizer.zero_grad()   
		loss.backward()         
		optimizer.step()        
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def test(model,optimizer,loader):
	
	loss_func = torch.nn.MSELoss()  
	epoch_loss = 0
	for step, (o_a,o_b,action) in enumerate(loader): 
		prediction = model(o_a,o_b)     
		loss = loss_func(prediction, action)
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def prepare_raw_data_gen(gparam):

	params, instance_keys  = [], []
	cases = itertools.product(*(gparam.num_nodes_A_lst,gparam.num_nodes_B_lst))
	for (num_nodes_A, num_nodes_B) in cases:
		for trial in range(gparam.num_trials):
			
			# save 
			instance_key = '{}a_{}b_{}trial'.format( \
				num_nodes_A,num_nodes_B,trial)

			# param 
			sim_param = Param()
			sim_param.num_nodes_A = num_nodes_A 
			sim_param.num_nodes_B = num_nodes_B
			sim_param.quiet_on = True 
			sim_param.update()
			
			# assign 
			params.append(sim_param)
			instance_keys.append(instance_key)

	return params, instance_keys


def run_batch(param, instance_key):

	env = Swarm(param)
	estimator = load_module(param.estimator_name).Estimator(param,env)
	attacker = load_module(param.attacker_name).Attacker(param,env)
	controller = load_module(param.controller_name).Controller(param,env)
	reset = env.get_reset()

	print('running instance {}... '.format(instance_key))
	sim_result = run_sim(param,env,reset,estimator,attacker,controller)
	try: 
		sim_result = run_sim(param,env,reset,estimator,attacker,controller)
	except:
		print('sim failed')
		return 

	state_action_fn = '{}raw_{}'.format( \
		gparam.demonstration_data_dir,instance_key)

	param_fn = '{}param_{}.json'.format( \
		gparam.demonstration_data_dir,instance_key)

	print('writing instance {}... ', instance_key)
	dh.write_state_action_pairs(sim_result,state_action_fn)
	dh.write_parameters(param.to_dict(),param_fn)


if __name__ == '__main__':

	gparam = Gparam()

	format_dir(gparam) 

	# run expert and write (state, action) pairs into files 
	if gparam.make_raw_data_on:

		print('making raw data...')

		# prepare run 
		params, instance_keys  = prepare_raw_data_gen(gparam) 

		# run 
		if gparam.serial_on:
			for (param, instance_key) in zip(params, instance_keys):
				run_batch(param, instance_key)
		else:	
			ncpu = cpu_count()
			print('ncpu: ', ncpu)
			with Pool(ncpu-1) as p:
				p.starmap(run_batch, zip(params,instance_keys))


	# load (state,action) files, apply measurement model, and write (observation,action) binary files
	if gparam.make_labelled_data_on: 

		print('(state,action) -> (observation,action)...')

		batched_observation_action_pairs = dict() # batched by number neighbors team_a, team_b 

		for instance_key in glob.glob('{}*.json'.format(gparam.demonstration_data_dir)):

			print('\t instance_key:',instance_key)

			instance_key = instance_key.split(gparam.demonstration_data_dir)[-1]
			instance_key = instance_key.split('.json')[0]
			instance_key = instance_key.split('param_')[-1]

			# filenames 
			state_action_fn = '{}raw_{}.npy'.format(\
				gparam.demonstration_data_dir,instance_key)

			param_fn = '{}param_{}.json'.format(\
				gparam.demonstration_data_dir,instance_key)

			# parameters
			sim_param_dict = dh.read_parameters(param_fn)
			sim_param = Param()
			sim_param.from_dict(sim_param_dict)

			# state action pairs 
			states,actions = dh.read_state_action_pairs(state_action_fn,sim_param)

			# init environment 
			env = Swarm(sim_param)

			# extract observations (by size)  
			for timestep,(state,action) in enumerate(zip(states,actions)):


				# first update state 
				state_dict = env.state_vec_to_dict(state)
				for node in env.nodes: 
					node.state = state_dict[node]


				# make neighbors
				neighbors_A = dict()
				neighbors_B = dict()
				for node_i in env.nodes:

					neighbors_A[node_i] = []
					neighbors_B[node_i] = [] 

					for node_j in env.nodes: 
						if node_j is not node_i: 
							if node_j.team_A and np.linalg.norm(node_j.state[0:2] - node_i.state[0:2]) < sim_param.r_sense: 
								neighbors_A[node_i].append(node_j) 
							elif node_j.team_B and np.linalg.norm(node_j.state[0:2] - node_i.state[0:2]) < sim_param.r_sense: 
								neighbors_B[node_i].append(node_j)


				# extract observation/action pair 			
				for node_i in env.nodes: 

					action_dim_per_agent = 2
					action_idxs = action_dim_per_agent * node_i.idx + np.arange(2)
					action_i = np.expand_dims(action[action_idxs],axis=1)

					observation_team_a = []
					observation_team_b = []
					for node_j in neighbors_A[node_i]: 
						observation_team_a.append(node_j.state - node_i.state)
					for node_j in neighbors_B[node_i]:
						observation_team_b.append(node_j.state - node_i.state)

					# append datapoint 
					key = (node_i.team_A,len(observation_team_a),len(observation_team_b))
					if key not in batched_observation_action_pairs.keys():
						batched_observation_action_pairs[key] = [(observation_team_a, observation_team_b, action_i)]
					else:
						batched_observation_action_pairs[key].append((observation_team_a, observation_team_b, action_i))

		dh.write_observation_action_pairs(batched_observation_action_pairs,gparam.demonstration_data_dir)


	# load (observation,action) binary files, train a model, and write model to file 
	if gparam.train_model_on: 

		print('training model...')
		
		training_team = "B"

		# get loader 
		loader = [] # lst of batches 
		for batched_file in glob.glob('{}**labelled_{}team**'.format(gparam.demonstration_data_dir,training_team)):
			o_a,o_b,action = dh.read_observation_action_pairs(batched_file,gparam.demonstration_data_dir)
			loader.append([
				torch.from_numpy(o_a).float().to(gparam.device),
				torch.from_numpy(o_b).float().to(gparam.device),
				torch.from_numpy(action).float().to(gparam.device)])
			
		# init model
		model = EmptyNet(gparam,gparam.device)

		# init optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=gparam.il_lr, weight_decay = gparam.il_wd)
		
		# train 
		with open(gparam.il_train_model_fn + ".csv", 'w') as log_file:
			log_file.write("time,epoch,train_loss,test_loss\n")
			start_time = time.time()
			best_test_loss = np.Inf
			scheduler = ReduceLROnPlateau(optimizer, 'min')
			for epoch in range(1,gparam.il_n_epoch+1):
				# train_epoch_loss = train(model,optimizer,loader_train)
				# test_epoch_loss = test(model,loader_test)
				train_epoch_loss = train(model,optimizer,loader)
				test_epoch_loss = test(model,optimizer,loader)
				scheduler.step(test_epoch_loss)
			
				if epoch%gparam.il_log_interval==0:
					print('epoch: ', epoch)
					print('   Train Epoch Loss: ', train_epoch_loss)
					print('   Test Epoch Loss: ', test_epoch_loss)

					if test_epoch_loss < best_test_loss:
						best_test_loss = test_epoch_loss
						print('      saving @ best test loss:', best_test_loss)
						torch.save(model.to('cpu'), gparam.il_train_model_fn)
						# model.save_weights(gparam.il_train_model_fn + ".tar")
						model.to(gparam.device)

				log_file.write("{},{},{},{}\n".format(time.time() - start_time, epoch, train_epoch_loss, test_epoch_loss))

