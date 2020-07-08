
import numpy as np
import os
import shutil
import json
import glob
import pandas as pd
from datetime import datetime,timedelta

from utilities import dbgp

class NumpyEncoder(json.JSONEncoder):

	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self,obj)


def write_sim_result(sim_result,sim_result_dir):
	# input: 
	# 	- dict of sim result  
	# 	- dir , ex: ../current_results/sim_result_2/ 
	# output: 
	# 	- bunch of files in a directory

	# make dir 
	if not os.path.exists(sim_result_dir):
		os.makedirs(sim_result_dir)

	# write 
	for key,item in sim_result.items():

		# parameters of simulation into a dict 
		if 'param' in key:
			param_fn = '{}/param.json'.format(sim_result_dir)
			with open(param_fn, 'w') as fp:
				json.dump(sim_result["param"], fp, cls=NumpyEncoder, indent=2)

		# unpack info into binaries 
		elif 'info' in key:
			for info_key, info_value in sim_result["info"].items():
				with open("{}/{}.npy".format(sim_result_dir,info_key), "wb") as f:
					np.save(f,info_value)	

		# rest into binary files 
		else: 
			with open("{}/{}.npy".format(sim_result_dir,key), "wb") as f:
				np.save(f,sim_result[key])


def load_sim_result(sim_result_dir):
	# input: 
	# 	- sim_result_dir: ex: '../current_results/sim_result_0'
	# output: 
	# 	- sim_result: dict of sim results 

	sim_result = dict()

	# load param 
	param_fn = '{}/param.json'.format(sim_result_dir)
	with open(param_fn, 'r') as j:
		sim_result["param"] = json.loads(j.read())

	# init info dict 
	info_keys = sim_result["param"]["info_keys"]
	sim_result["info"] = dict()

	# read binaries
	for file in glob.glob(sim_result_dir + '/*.npy'):
		base = file.split("/")[-1]
		base = base.split(".npy")[0]
		value = np.load(file,allow_pickle=True)

		if "_name" in base:
			value = str(value)

		if base in info_keys:
			sim_result["info"][base] = value

		else:
			sim_result[base] = value

	return sim_result
	

def write_state_action_pairs(sim_result,fn):

	states = np.array(sim_result["info"]["state_vec"]) # [nt x state_dim x 1]
	actions = sim_result["actions"] # [nt x ni x action_dim_per_agent x 1]
	times = np.expand_dims(sim_result["times"],axis=1) # [nt x 1]

	num_timesteps, num_agents, action_dim_per_agent, _ = actions.shape
	_, state_dim, _ = states.shape

	flat_actions = np.zeros((num_timesteps, num_agents*action_dim_per_agent,1))
	for timestep,action in enumerate(actions): 
		new_idxs = np.arange(action_dim_per_agent)
		for node_idx, action_i in enumerate(action): 
			flat_actions[timestep,new_idxs,:] = action_i
			new_idxs += action_dim_per_agent

	state_action_pairs = np.zeros((num_timesteps, state_dim + num_agents*action_dim_per_agent))
	for timestep,(state,action) in enumerate(zip(states,flat_actions)):
		state_action_pairs[timestep,:] = np.vstack((state,action)).flatten()

	np.save(fn,state_action_pairs)


def write_observation_action_pairs(observation_action_pairs_dict, datadir):

	for (team, num_a, num_b), observation_action_pairs in observation_action_pairs_dict.items():
		batched_dataset = [] 
		for (o_a, o_b, action) in observation_action_pairs:
			data = np.concatenate((np.array(o_a).flatten(),np.array(o_b).flatten(),np.array(action).flatten()))
			batched_dataset.append(data)

		if team: 
			team_name = "A"
		else:
			team_name = "B"

		fn = '{}labelled_{}team_{}a_{}b.npy'.format(datadir,team_name,num_a,num_b)
		np.save(fn,batched_dataset)


def write_parameters(param_dict,fn):

	with open(fn, 'w') as fp:
		json.dump(param_dict, fp, cls=NumpyEncoder, indent=2)


def read_parameters(fn):

	with open(fn, 'r') as j:
		param_dict = json.loads(j.read())
	return param_dict


def read_state_action_pairs(fn,param):

	data = np.load(fn)
	states = data[:,0:param.state_dim]
	actions = data[:,param.state_dim:]
	return states,actions


def read_observation_action_pairs(fn,datadir):

	data = np.load(fn)

	key = fn.split(datadir)[-1]
	key = key.split('_')
	num_a = int(key[2].split('a')[0])
	num_b = int(key[3].split('b')[0])
	action_dim_per_agent = 2
	state_dim_per_agent = 4

	o_a = data[:,0:num_a*state_dim_per_agent]
	o_b = data[:,num_a*state_dim_per_agent:(num_a+num_b)*state_dim_per_agent]
	action = data[:,-action_dim_per_agent:]
	
	return o_a,o_b,action 

