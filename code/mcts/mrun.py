

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

	initial_condition = df_param.state 

	params = [] 
	for _ in range(mp.cpu_count()-1):
		param = Param() 
		param.current_results_dir = '../'+param.current_results_dir
		param.update(initial_condition)
		params.append(param)
	return params


def run_sim(param):

	with tempfile.TemporaryDirectory() as tmpdirname:
		input_file = tmpdirname + "/config.yaml" 
		dh.write_mcts_config_file(param, input_file)
		output_file = tmpdirname + "/output.csv"
		print('running instance...')
		subprocess.run("../mcts/cpp/buildRelease/swarmgame -i {} -o {}".format(input_file, output_file), shell=True)
		data = np.loadtxt(output_file, delimiter=',', skiprows=1, dtype=np.float32)

	sim_result = dh.convert_cpp_data_to_sim_result(data,param)

	case = len(glob.glob(param.current_results_dir + '/*'))
	save_fn = param.current_results_dir + '/sim_result_{}'.format(case)
	print('writing instance {}... '.format(save_fn))
	dh.write_sim_result(sim_result,save_fn)
	print('completed instance {}'.format(save_fn))

if __name__ == '__main__':

	df_param = Param()
	df_param.current_results_dir = '../'+df_param.current_results_dir
	format_dir(df_param)

	params = get_params(df_param)

	parallel = True
	if parallel: 
		pool = mp.Pool(mp.cpu_count()-1)
		for _ in pool.imap_unordered(run_sim, params):
		# for _ in pool.imap_unordered(run_sim, [param for _ in range(ncases)]):
			pass 
	else: 
		for param in params: 
			run_sim(param)
			break 

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.current_results_dir + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	for sim_result in sim_results:
		plotter.plot_tree_results(sim_result)

	plotter.save_figs("plots.pdf")
	plotter.open_figs("plots.pdf")