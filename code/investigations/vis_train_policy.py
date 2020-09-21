
import sys, os 
import glob
import torch
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append('../')
from param import Param 
from mice import format_data
from learning.continuous_emptynet import ContinuousEmptyNet
import datahandler as dh
import plotter

if __name__ == '__main__':

	# params 
	# - default
	df_param = Param() 
	# - model 
	# path_to_model = '../../saved/test_IL_09192020_models/models/a10.pt' # test_DAgger_09192020_models, test_IL_09192020_models
	path_to_model = '../../current/models/a1.pt' # test_DAgger_09192020_models, test_IL_09192020_models
	# - dataset 
	# path_to_datadir = '../../saved/test_IL_09192020_data/data/' # test_DAgger_09192020_data, test_IL_09192020_data
	path_to_datadir = '../../current/data/' # test_DAgger_09192020_data, test_IL_09192020_data
	# - vis 
	team_1_color = 'blue'
	team_2_color = 'orange'
	goal_color = 'green'
	self_color = 'black'
	nbins = 20
	LIMS = df_param.standard_robot["acceleration_limit"]*np.array([[-1,1],[-1,1]])
	rsense = df_param.standard_robot["r_sense"]
	env_xlim = df_param.env_xlim 
	env_ylim = df_param.env_ylim 
	num_vis = 1
	n_samples = 100
	eps = 0 # only take identical game conditions -> at least #subsamples 

	# load dataset 	
	# fix relative path stupidness 
	batched_fns = glob.glob(df_param.l_labelled_fn.format(\
		DATADIR=path_to_datadir,NUM_A='**',NUM_B='**',IDX_TRIAL='**',TEAM="a",ITER='**'))
	o_as,o_bs,goals,values,actions,weights = dh.read_oa_batch(batched_fns[0])

	# print('o_as',o_as)
	# print('o_bs',o_bs)
	# print('goals',goals)
	# print('values',values)
	# print('actions',actions)
	
	# load models
	model = ContinuousEmptyNet(df_param,"cpu")
	model.load_state_dict(torch.load(path_to_model))

	# pick random observations	
	idxs = np.random.choice(len(o_as),num_vis)

	for i_state in range(num_vis):

		# pick random observation 

		# select candidate observations 
		candidate = (o_as[idxs[i_state]],o_bs[idxs[i_state]],goals[idxs[i_state]])
		print('candidate',candidate)

		# append all identical ones (should be # subsamples)
		conditionals = [] 
		dataset_actions = []
		for o_a,o_b,goal,action in zip(o_as,o_bs,goals,actions):
			if (np.linalg.norm(o_a - candidate[0]) <= eps) and \
				(np.linalg.norm(o_b - candidate[1]) <= eps) and \
				(np.linalg.norm(goal - candidate[2]) <= eps):

				conditionals.append((o_a,o_b,goal))
				dataset_actions.append(action)

		print('conditionals',conditionals)
		print('dataset_actions',dataset_actions)

		# query model 
		model_actions = [] 
		for o_a,o_b,goal in conditionals:
			o_a,o_b,goal = format_data(o_a,o_b,goal)
			for _ in range(n_samples):
				value, policy = model(o_a,o_b,goal)
				model_actions.append(policy.detach().numpy())

		print('model_actions',model_actions)

		# convert for easy plot
		model_actions = np.array(model_actions).squeeze()
		dataset_actions = np.array(dataset_actions)
		
		# vis 
		# fig: game state encoding  
		fig, axs = plt.subplots(nrows=2,ncols=2)

		# 
		axs[0][1].set_axis_off()

		# - self 
		vx = -1*candidate[2][2]
		vy = -1*candidate[2][3]
		axs[0][0].scatter(0,0,color=self_color)
		axs[0][0].arrow(0,0,vx,vy,color=self_color,alpha=0.5)	

		# - goal 
		axs[0][0].scatter(candidate[2][0],candidate[2][1],color=goal_color,alpha=0.5)

		# - neighbors 
		num_a = int(len(candidate[0])/4)
		num_b = int(len(candidate[1])/4)
		for robot_idx in range(num_a):
			axs[0][0].scatter(candidate[0][robot_idx*4],candidate[0][robot_idx*4+1],color=team_1_color)
			axs[0][0].arrow(candidate[0][robot_idx*4],candidate[0][robot_idx*4+1],\
				vx+candidate[0][robot_idx*4+2],vy+candidate[0][robot_idx*4+3],\
				color=team_1_color,alpha=0.5)
		for robot_idx in range(num_b):
			axs[0][0].scatter(candidate[1][robot_idx*4],candidate[1][robot_idx*4+1],color=team_2_color)
			axs[0][0].arrow(candidate[1][robot_idx*4],candidate[1][robot_idx*4+1],\
				vx+candidate[1][robot_idx*4+2],vy+candidate[1][robot_idx*4+3],\
				color=team_2_color,alpha=0.5)

		# - arrange  
		axs[0][0].set_xlim([np.max((-rsense,-env_xlim[1])),np.min((rsense,env_xlim[1]))])
		axs[0][0].set_ylim([np.max((-rsense,-env_ylim[1])),np.min((rsense,env_ylim[1]))])
		# axs[0][0].set_xlim([-rsense,rsense])
		# axs[0][0].set_ylim([-rsense,rsense])
		axs[0][0].set_title('game state: {}'.format(i_state))
		axs[0][0].set_aspect('equal')
		
		# fig: histograms of model/dataset in action space
		# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
		xedges = np.linspace(LIMS[0,0],LIMS[0,1],nbins) 
		yedges = np.linspace(LIMS[1,0],LIMS[1,1],nbins) 

		h_mcts, xedges, yedges = np.histogram2d(dataset_actions[:,0],dataset_actions[:,1],bins=(xedges,yedges),range=LIMS,density=True)
		h_model, xedges, yedges = np.histogram2d(model_actions[:,0],model_actions[:,1],bins=(xedges,yedges),range=LIMS,density=True)

		h_mcts = h_mcts.T 
		h_model = h_model.T 		

		im1 = axs[1][0].imshow(h_mcts,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]])
		im2 = axs[1][1].imshow(h_model,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]])

		# - arrange 
		axs[1][0].set_title('mcts: {}'.format(i_state))
		axs[1][1].set_title('model: {}'.format(i_state))
		axs[1][0].set_xlabel('x-action')
		axs[1][1].set_xlabel('x-action')
		axs[1][0].set_ylabel('y-action')

		fig.tight_layout()
		fig.colorbar(im1, ax=axs[1][0])
		fig.colorbar(im2, ax=axs[1][1])


	plotter.save_figs("../plots/vis_train.pdf")
	plotter.open_figs("../plots/vis_train.pdf")
