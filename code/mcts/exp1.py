
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
from param import Param 
from env import Swarm
import datahandler as dh
import plotter 


def format_dir(param):

	# clean current results dir 
	for old_sim_result_dir in glob.glob(param.current_results_dir):
		shutil.rmtree(old_sim_result_dir)

	# make current results dir 
	os.makedirs(param.current_results_dir,exist_ok=True)


def get_params(df_param):
	params = [] 
	for tree_size in df_param.tree_sizes:
		for trial in range(df_param.num_trials):
			
			seed = int.from_bytes(os.urandom(4), sys.byteorder)

			for i in range(2): 
				param = Param()
				param.current_results_dir = '../' + param.current_results_dir
				param.seed = seed
				param.tree_size = tree_size
				param.update(df_param.state)

				if i == 0: 
					param.glas_rollout_on = True
				else:
					param.glas_rollout_on = False

				param.sim_results_fig_title = get_title(tree_size,param.glas_rollout_on) 
				params.append(param)

	return params 


def get_title(tree_size,glas_on):
	if glas_on: 
		policy = 'GLAS'
	else:
		policy = 'Random'
	return 'Num Nodes: {}, \n Rollout: {} '.format(tree_size, policy)


def write_combined_model_file(param):
	cmd = "python cpp/convertNN.py {} {} {}".format(param.glas_model_A, param.glas_model_B, param.combined_model_name)
	print(cmd)
	subprocess.run(cmd, shell=True)


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

	case = len(glob.glob(param.current_results_dir + '/*'))
	save_fn = param.current_results_dir + '/sim_result_{}'.format(case)
	print('writing instance {}... '.format(save_fn))
	dh.write_sim_result(sim_result,save_fn)
	print('completed instance {}'.format(save_fn))


if __name__ == '__main__':

	df_param = Param()
	df_param.num_trials = 10
	df_param.tree_sizes = [1000,5000,10000,50000]
	df_param.current_results_dir = '../' + df_param.current_results_dir	
	df_param.glas_model_A = '../' + df_param.glas_model_A
	df_param.glas_model_B = '../' + df_param.glas_model_B

	format_dir(df_param)
	write_combined_model_file(df_param)
	params = get_params(df_param)

	parallel = True
	if parallel: 
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

	plotter.plot_exp1_results(sim_results)
	plotter.save_figs("plots.pdf")
	plotter.open_figs("plots.pdf")
