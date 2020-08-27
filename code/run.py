

# standard packages
import os, sys
import glob
import multiprocessing as mp 
import argparse 

# my packages 
from param import Param
from cpp_interface import rollout
import datahandler
import plotter 

def run_sim(param):
	sim_result = rollout(param,param.sim_mode)
	results_fn = os.path.join(param.path_current_results,'sim_result_{}'.format(param.trial))
	datahandler.write_sim_result(sim_result,results_fn)

def get_params(df_param):	
	params = [] 
	for trial in range(df_param.sim_num_trials):
		param = Param()
		param.sim_mode = df_param.sim_mode
		param.path_glas_model_a = df_param.path_glas_model_a
		param.path_glas_model_b = df_param.path_glas_model_b
		param.trial = trial
		param.update()
		params.append(param)
	return params 

def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)
	os.makedirs(df_param.path_current_models,exist_ok=True)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-sim_mode", default=None, required=False)
	parser.add_argument("-path_glas_model_a", default=None, required=False)
	parser.add_argument("-path_glas_model_b", default=None, required=False)
	args = parser.parse_args()

	df_param = Param() 

	if not args.sim_mode is None: 
		df_param.sim_mode = args.sim_mode
	if not args.path_glas_model_a is None: 
		df_param.path_glas_model_a = args.path_glas_model_a
	if not args.path_glas_model_b is None: 
		df_param.path_glas_model_b = args.path_glas_model_b				

	format_dir(df_param)
	params = get_params(df_param)
	
	# run sim 
	print('running sims...')
	if df_param.sim_parallel_on: 
		pool = mp.Pool(min((mp.cpu_count()-1,df_param.sim_num_trials)))
		for _ in pool.imap_unordered(run_sim, params):
			pass 
	else: 
		df_param.trial = 0 
		run_sim(df_param)

	# load sim results 
	print('loading sim results...')
	sim_results = [] # lst of dicts
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(datahandler.load_sim_result(sim_result_dir))

	# plotting 
	print('plotting sim results...')
	for sim_result in sim_results:
		plotter.plot_tree_results(sim_result)

	print('saving and opening figs...')
	plotter.save_figs('plots/run.pdf')
	plotter.open_figs('plots/run.pdf')
