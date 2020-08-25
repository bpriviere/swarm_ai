
import mcts
import numpy as np 
import matplotlib.pyplot as plt 
import multiprocessing as mp
import glob
import os 
import shutil 
import time as timer
import tempfile
import subprocess

import sys
sys.path.append("../")
sys.path.append("cpp")
from param import Param 
from env import Swarm
import datahandler as dh
import plotter 
from convertNN import convertNN


def format_dir(param):

	# clean current results dir 
	for old_sim_result_dir in glob.glob(param.current_results_dir):
		shutil.rmtree(old_sim_result_dir)

	# make current results dir 
	os.makedirs(param.current_results_dir,exist_ok=True)


def get_params(df_param):
	params = [] 

	count = 0 
	for case in range(df_param.num_cases): 

		df_param.seed = int.from_bytes(os.urandom(4), sys.byteorder)
		df_param.make_initial_condition()
		initial_condition = df_param.state
		
		for tree_size in df_param.tree_sizes:
		
			for trial in range(df_param.num_trials):
			
				for glas_rollout_on in df_param.glas_rollout_on_cases: 
				# for glas_rollout_on in [True,False]: 
					param = Param()
					param.current_results_dir = '../' + param.current_results_dir
					param.seed = int.from_bytes(os.urandom(4), sys.byteorder)
					param.tree_size = tree_size
					param.tree_sizes = df_param.tree_sizes 
					param.trial = trial 
					param.num_trials = df_param.num_trials
					param.case = case
					param.num_cases = df_param.num_cases
					param.glas_rollout_on = glas_rollout_on
					param.glas_rollout_on_cases = df_param.glas_rollout_on_cases
					param.count = count 
					param.update(initial_condition)
					
					params.append(param)
					count += 1 

	return params 


def get_title(tree_size,glas_on):
	if glas_on: 
		policy = 'GLAS'
	else:
		policy = 'Random'
	return 'Num Nodes: {}, \n Rollout: {} '.format(tree_size, policy)


def write_combined_model_file(param):
	convertNN(param.glas_model_A, param.glas_model_B, param.combined_model_name)

def run_sim(param):

	with tempfile.TemporaryDirectory() as tmpdirname:
		input_file = tmpdirname + "/config.yaml" 
		dh.write_mcts_config_file(param, input_file)
		output_file = tmpdirname + "/output.csv"
		print('running instance...')

		start = timer.time()

		if param.glas_rollout_on:
			model_file = param.combined_model_name
			subprocess.run("../mcts/cpp/buildRelease/swarmgame -i {} -o {} -n {}".format(input_file, output_file, model_file), shell=True)
		else:
			subprocess.run("../mcts/cpp/buildRelease/swarmgame -i {} -o {}".format(input_file, output_file), shell=True)

		elapsed = timer.time() - start
		param.elapsed = elapsed
		data = np.loadtxt(output_file, delimiter=',', skiprows=1, ndmin=2, dtype=np.float32)

	sim_result = dh.convert_cpp_data_to_sim_result(data,param)

	save_fn = param.current_results_dir + '/sim_result_{}'.format(param.count)
	print('writing instance {}... '.format(save_fn))
	dh.write_sim_result(sim_result,save_fn)
	print('completed instance {}'.format(save_fn))


if __name__ == '__main__':

	run_on = True
	parallel_on = True

	df_param = Param()
	df_param.num_trials = 25
	df_param.glas_rollout_on_cases = [True,False] 
	df_param.num_cases = 5
	df_param.tree_sizes = [1000,5000,10000,50000,100000,200000] 
	df_param.current_results_dir = '../' + df_param.current_results_dir	
	df_param.glas_model_A = '../' + df_param.glas_model_A
	df_param.glas_model_B = '../' + df_param.glas_model_B

	if run_on:

		format_dir(df_param)
		write_combined_model_file(df_param)
		params = get_params(df_param)

		if parallel_on: 
			pool = mp.Pool(mp.cpu_count()-1)
			for _ in pool.imap_unordered(run_sim, params):
				pass 
		else: 
			for param in params: 
				run_sim(param)
				break 

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.current_results_dir + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_convergence(sim_results)
	plotter.plot_exp1_results(sim_results)
	
	plotter.save_figs("plots.pdf")
	plotter.open_figs("plots.pdf")
