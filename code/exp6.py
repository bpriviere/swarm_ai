
import numpy as np 
import multiprocessing as mp
import shutil, os, glob

from param import Param 
import plotter 
import datahandler as dh 

"""
This experiment plots trees over time 
"""

def do_work(param):
	from cpp_interface import self_play

	print('{}/{}...'.format(param.i_trial,param.total))

	# play game 
	sim_result = self_play(param)

	# write sim results
	dh.write_sim_result(sim_result,param.dataset_fn)

	# plot and video results 
	plotter.plot_exp6(sim_result,param.snapshot_dir) 
	plotter.save_video(param.snapshot_dir + '/',param.save_dir,str(param.i_trial))

def get_params(df_param):
	params = [] 
	count = 0 
	for i_trial in range(df_param.n_trials):
		for policy_i in df_param.policies: 
			param = Param() 
			param.policy_dict['path_glas_model_a'] = "{}/a{}.pt".format(df_param.exp_6_model_dir,policy_i) if policy_i > 0 else None
			param.policy_dict['path_glas_model_b'] = "{}/b{}.pt".format(df_param.exp_6_model_dir,policy_i) if policy_i > 0 else None
			param.policy_dict['path_value_fnc'] = "{}/v{}.pt".format(df_param.exp_6_model_dir,policy_i) if policy_i > 0 else None
			param.plot_tree_on = True 
			param.tree_timestep = 1 
			param.i_trial = count
			param.n = i_trial
			param.dataset_fn = '{}sim_result_{:03d}'.format(\
						df_param.path_current_results,param.i_trial)
			param.save_dir = df_param.save_dir
			param.snapshot_dir = os.path.join(df_param.save_dir,"{}".format(param.i_trial))
			param.update(initial_condition=df_param.state)
			params.append(param)
			count += 1 

	for param in params:
		param.total = len(params)
	return params		

def format_dir(df_param):

	# wipe existing directory with videos/snapshots
	if os.path.exists(df_param.save_dir):
		shutil.rmtree(df_param.save_dir)
	os.makedirs(df_param.save_dir)

	# create subdirs to save 
	for i_trial in range(df_param.n_trials * len(df_param.policies)):
		subdir = os.path.join(df_param.save_dir,"{}".format(i_trial))
		os.makedirs(subdir)

	# wipe current results directory 
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)

def main():

	df_param = Param() 
	df_param.n_trials = 5
	df_param.save_dir = "plots/exp6"
	df_param.exp_6_model_dir = "../saved/r28"
	df_param.policies = [3]
	df_param.state = [
		[0.1*df_param.env_l,0.4*df_param.env_l,0,0],
		[0.1*df_param.env_l,0.6*df_param.env_l,0,0],
		[0.9*df_param.env_l,0.5*df_param.env_l,0,0],
		]

	format_dir(df_param)

	params = get_params(df_param)
	with mp.Pool(mp.cpu_count()-1) as pool:
		for _ in pool.imap_unordered(do_work, params):
			pass
	
	# load sim results
	sim_results = [] 
	files = sorted(glob.glob(df_param.path_current_results + '/*'))
	for sim_result_dir in files:
		sim_results.append(dh.load_sim_result(sim_result_dir))

	# Plot results 
	print("\nGenerating Plots")
	count = 0 
	for sim_result in sim_results:
		_, filename = os.path.split(files[count])

		model_name = os.path.basename(sim_result["param"]["policy_dict"]["path_glas_model_a"])[1] if \
			sim_result["param"]["policy_dict"]["path_glas_model_a"] is not None else "None"

		# Plot results of each run
		sim_result["trees"] = []
		plotter.plot_tree_results(sim_result,title=
			"i={},model={}',trial={}".format(sim_result["param"]["i_trial"],model_name,sim_result["param"]["n"]))
		count += 1 
		# Limit the maximum number of results files to plot
		if count >= 50: 
			break 

	print('saving and opening figs...')
	plotter.save_figs("plots/exp6.pdf")
	plotter.open_figs("plots/exp6.pdf")	

if __name__ == '__main__':
	main()


# print('playing game...')
# sim_result = play_game(param,policy_dict_a,policy_dict_b)

# print('plotting sim_results and making videos...')
# for sim_result in [sim_result]: 
# 	# label = policies_to_fn(\
# 	# 	param.policy_dict_a,param.policy_dict_b)
# 	label='test'
# 	plotter.plot_exp6(sim_result,"plots/exp6/{}/".format(label)) 

# 	# save_video(png_directory,output_dir,output_file)
# 	plotter.save_video("plots/exp6/{}/".format(label),"plots/exp6/",label)

# from param import Param 
# from cpp_interface import play_game
# import plotter 
# import numpy as np 

# """
# This experiment plots trees over time 
# """

# def get_params(df_param):

# 	params = [] 
# 	for i_trial in range(df_param.n_trials):

# 		param = Param() 
# 		param.plot_tree_on = True 
# 		param.tree_timestep = 1 




# def main():

# 	n_trials = 4

# 	param = Param() 
# 	param.plot_tree_on = True 
# 	param.tree_timestep = 1 

# 	# robot types 
# 	param.robot_types = {
# 		'standard_robot' : {
# 			# 'speed_limit': 0.125,
# 			'speed_limit': 0.0625,
# 			'acceleration_limit':0.125,
# 			'tag_radius': 0.0125,
# 			'dynamics':'double_integrator',
# 			'r_sense': 0.5,
# 			'radius': 0.025,
# 		},
# 		'evasive_robot' : {
# 			'speed_limit': 0.125,
# 			'acceleration_limit':0.2,
# 			'tag_radius': 0.025,
# 			'dynamics':'double_integrator',
# 			'r_sense': 0.5,
# 			'radius': 0.025,
# 		}
# 	}

# 	# fix some initial condition 
# 	param.update()
# 	param.state = [
# 		[0.1*param.env_l,0.5*param.env_l,0,0],
# 		[0.9*param.env_l,0.5*param.env_l,0,0],
# 		]
# 	param.goal = np.array([0.8*param.env_l,0.5*param.env_l,0,0])

# 	policy_dict_a = {
# 		'sim_mode' : 				"MCTS", # "MCTS, D_MCTS, RANDOM, PANAGOU, GLAS"
# 		'path_glas_model_a' : 		None, #'../current/models/a4.pt', None
# 		'path_glas_model_b' : 		None, #'../current/models/b4.pt', None
# 		'path_value_fnc' : 			None, #'../current/models/v4.pt', None
# 		'mcts_tree_size' : 			10000, # 10000,
# 		'mcts_c_param' : 			5.0,
# 		'mcts_pw_C' : 				0.5,
# 		'mcts_pw_alpha' : 			0.25,
# 		'mcts_beta1' : 				0.0,
# 		'mcts_beta2' : 				0.5,
# 		'mcts_beta3' : 				0.5,
# 	}
# 	policy_dict_b = policy_dict_a.copy()

# 	print('playing game...')
# 	sim_result = play_game(param,policy_dict_a,policy_dict_b)

# 	print('plotting sim_results and making videos...')
# 	for sim_result in [sim_result]: 
# 		# label = policies_to_fn(\
# 		# 	param.policy_dict_a,param.policy_dict_b)
# 		label='test'
# 		plotter.plot_exp6(sim_result,"plots/exp6/{}/".format(label)) 

# 		# save_video(png_directory,output_dir,output_file)
# 		plotter.save_video("plots/exp6/{}/".format(label),"plots/exp6/",label)


	

# if __name__ == '__main__':
# 	main()