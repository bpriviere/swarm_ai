
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
	

