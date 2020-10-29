

from param import Param 
from cpp_interface import play_game
import plotter 

"""
This experiment plots trees over time 
"""



def main():

	n_trials = 4

	param = Param() 
	param.plot_tree_on = True 
	param.tree_timestep = 1 

	policy_dict_a = {
		'sim_mode' : 				"MCTS", # "MCTS, D_MCTS, RANDOM, PANAGOU, GLAS"
		'path_glas_model_a' : 		'../current/models/a4.pt', 
		'path_glas_model_b' : 		'../current/models/b4.pt', 
		'path_value_fnc' : 			'../current/models/v4.pt', 
		'mcts_tree_size' : 			1000,
		'mcts_c_param' : 			1.4,
		'mcts_pw_C' : 				1.0,
		'mcts_pw_alpha' : 			0.25,
		'mcts_beta1' : 				0.0,
		'mcts_beta2' : 				0.5,
		'mcts_beta3' : 				0.5,
	}
	policy_dict_b = policy_dict_a.copy()

	print('playing game...')
	sim_result = play_game(param,policy_dict_a,policy_dict_b)

	print('plotting sim_results and making videos...')
	for sim_result in [sim_result]: 
		# label = policies_to_fn(\
		# 	param.policy_dict_a,param.policy_dict_b)
		label='test'
		plotter.plot_exp6(sim_result,"plots/exp6/{}/".format(label)) 

		# save_video(png_directory,output_dir,output_file)
		plotter.save_video("plots/exp6/{}/".format(label),"plots/exp6/",label)


	

if __name__ == '__main__':
	main()