
# standard
import os, sys, glob
import itertools
import numpy as np 
import torch 
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multiprocessing import cpu_count, Pool 
import concurrent.futures

# project
sys.path.append("../")
from env import Swarm
from run import run_sim
from param import Param 
from gparam import Gparam
from learning.emptynet import EmptyNet
from measurements.relative_state import relative_state
from utilities import load_module
import datahandler as dh



def format_dir(param):
	if not os.path.exists(param.demonstration_data_dir):
		os.makedirs(param.demonstration_data_dir,exist_ok=True)
	if not os.path.exists(param.model_dir):
		os.makedirs(param.model_dir,exist_ok=True)
		

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
	# cases = itertools.product(*(gparam.num_nodes_A_lst,gparam.num_nodes_B_lst))
	for (num_nodes_A, num_nodes_B) in zip(gparam.num_nodes_A_lst,gparam.num_nodes_B_lst):
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

	print('writing instance {}... '.format(instance_key))
	dh.write_state_action_pairs(sim_result,state_action_fn)
	dh.write_parameters(param.to_dict(),param_fn)

	print('completed instance {}'.format(instance_key))


if __name__ == '__main__':

	gparam = Gparam()

	format_dir(gparam) 

	# run expert and write (state, action) pairs into files 
	if gparam.make_raw_data_on:
		print('making raw data...')
		
		params, instance_keys = prepare_raw_data_gen(gparam) 
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
		print('make labelled data...')

		batched_oa_pairs = dict() # batched by number neighbors team_a, team_b 
		for instance_key in glob.glob('{}*.json'.format(gparam.demonstration_data_dir)):

			instance_key = instance_key.split(gparam.demonstration_data_dir)[-1]
			instance_key = instance_key.split('.json')[0]
			instance_key = instance_key.split('param_')[-1]

			print('\t instance_key:',instance_key)

			# filenames 
			state_action_fn = '{}raw_{}.npy'.format(\
				gparam.demonstration_data_dir,instance_key)

			param_fn = '{}param_{}.json'.format(\
				gparam.demonstration_data_dir,instance_key)

			# parameters
			param_dict = dh.read_parameters(param_fn)
			param = Param()
			param.from_dict(param_dict)

			# state action pairs 
			states,actions = dh.read_state_action_pairs(state_action_fn,param)

			env = Swarm(param)
			for timestep,(state,action) in enumerate(zip(states,actions)):

				# first update state 
				state_dict = env.state_vec_to_dict(state)
				for node in env.nodes: 
					node.state = state_dict[node]

				observations = relative_state(env.nodes,param.r_sense)

				# extract observation/action pair 			
				for node_i in env.nodes: 

					action_dim_per_agent = 2
					action_idxs = action_dim_per_agent * node_i.idx + np.arange(2)
					action_i = np.expand_dims(action[action_idxs],axis=1)

					o_a, o_b = observations[node_i]

					# append datapoint 
					key = (node_i.team_A,len(o_a),len(o_b))
					if key not in batched_oa_pairs.keys():
						batched_oa_pairs[key] = [(o_a, o_b, action_i)]
					else:
						batched_oa_pairs[key].append((o_a, o_b, action_i))

		dh.write_observation_action_pairs(batched_oa_pairs,gparam.demonstration_data_dir)


	# load (observation,action) binary files, train a model, and write model to file 
	if gparam.train_model_on: 

		print('training model...')
		
		# get loader 
		train_loader = [] # lst of batches 
		test_loader  = [] 
		n_points 	 = 0 
		for batched_file in glob.glob('{}**labelled_{}team**'.format(gparam.demonstration_data_dir,gparam.training_team)):
			o_a,o_b,action = dh.read_observation_action_pairs(batched_file,gparam.demonstration_data_dir)
			
			if n_points < gparam.il_test_train_ratio * gparam.il_n_points: 
				train_loader.append([
					torch.from_numpy(o_a).float().to(gparam.device),
					torch.from_numpy(o_b).float().to(gparam.device),
					torch.from_numpy(action).float().to(gparam.device)])

			elif n_points < gparam.il_n_points:
				test_loader.append([
					torch.from_numpy(o_a).float().to(gparam.device),
					torch.from_numpy(o_b).float().to(gparam.device),
					torch.from_numpy(action).float().to(gparam.device)])

			else: 
				break 

			n_points += action.shape[0]

		# get sizes
		test_dataset_size = 0 
		train_dataset_size = 0 
		for batch in test_loader: 
			test_dataset_size += batch[2].shape[0]
		for batch in train_loader: 
			train_dataset_size += batch[2].shape[0]	

		print('train dataset size: ', train_dataset_size)
		print('test dataset size: ', test_dataset_size)

		# init model
		model = EmptyNet(gparam,gparam.device)

		# init optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=gparam.il_lr, weight_decay=gparam.il_wd)
		
		# train 
		with open(gparam.il_train_model_fn + ".csv", 'w') as log_file:
			log_file.write("time,epoch,train_loss,test_loss\n")
			start_time = time.time()
			best_test_loss = np.Inf
			scheduler = ReduceLROnPlateau(optimizer, 'min')
			for epoch in range(1,gparam.il_n_epoch+1):
				train_epoch_loss = train(model,optimizer,train_loader)
				test_epoch_loss = test(model,optimizer,test_loader)
				scheduler.step(test_epoch_loss)
			
				if epoch%gparam.il_log_interval==0:
					print('epoch: ', epoch)
					print('   Train Epoch Loss: ', train_epoch_loss)
					print('   Test Epoch Loss: ', test_epoch_loss)

					if test_epoch_loss < best_test_loss:
						best_test_loss = test_epoch_loss
						print('      saving @ best test loss:', best_test_loss)
						torch.save(model.state_dict(), gparam.il_train_model_fn)
						model.to(gparam.device)

				log_file.write("{},{},{},{}\n".format(time.time() - start_time, epoch, train_epoch_loss, test_epoch_loss))

