
import plotter 
import numpy as np 


if __name__ == '__main__':
	# /home/ben/projects/swarm_ai/saved/m5/models/a1_losses.npy

	dirnames = [
		"../saved/m5/models",
		"../saved/m6/models",
	]
	model_numbers = np.arange(1,8)
	loss_fn = "{DIRNAME}/{TEAM}{MODELNUMBER}_losses.npy"

	result = {} 

	for dirname in dirnames: 
		for team in ["a","b"]:
			for model_number in model_numbers: 
				key = (dirname,team,model_number)
				result[key] = np.load(loss_fn.format(
					DIRNAME=dirname, 
					TEAM=team, 
					MODELNUMBER=model_number
					))

	plotter.plot_exp9(result)
	print('saving and opening figs...')
	plotter.save_figs("plots/exp3.pdf")
	plotter.open_figs("plots/exp3.pdf")