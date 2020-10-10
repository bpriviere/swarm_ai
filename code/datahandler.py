
import numpy as np
import os
import shutil
import json
import glob
import yaml
import pandas as pd
import pickle
from datetime import datetime,timedelta

def write_sim_result(sim_result_dict,fn):
	with open(fn+'.pickle', 'xb') as h:
		pickle.dump(sim_result_dict, h)

def load_sim_result(fn):
	with open(fn, 'rb') as h:
		sim_result = pickle.load(h)
	return sim_result

def write_oa_batch(batched_dataset,batch_fn):
	np.save(batch_fn, batched_dataset)

def read_oa_batch(fn,l_gaussian_on):
	data = np.load(fn)

	key = os.path.basename(fn)
	key = key.split('_')
	num_a = int(key[3].split('a')[-1])
	num_b = int(key[4].split('b')[-1])
	action_dim_per_agent = 2
	state_dim_per_agent = 4

	o_a = data[:,0:num_a*state_dim_per_agent]
	o_b = data[:,num_a*state_dim_per_agent:(num_a+num_b)*state_dim_per_agent]
	goal = data[:,(num_a+num_b)*state_dim_per_agent:(num_a+num_b)*state_dim_per_agent+state_dim_per_agent]

	value = data[:,(num_a+num_b)*state_dim_per_agent+state_dim_per_agent]

	if l_gaussian_on:
		action = data[:,((num_a+num_b)*state_dim_per_agent+state_dim_per_agent+1):-2]
		weight = data[:,-2:]
	else: 
		action = data[:,((num_a+num_b)*state_dim_per_agent+state_dim_per_agent+1):-1]
		weight = data[:,-1]
	
	return o_a,o_b,goal,value,action,weight 
	


def write_parameters(param_dict,fn):
	with open(fn, 'w') as fp:
		json.dump(param_dict, fp, cls=NumpyEncoder, indent=2)


def read_parameters(fn):
	with open(fn, 'r') as j:
		param_dict = json.loads(j.read())
	return param_dict