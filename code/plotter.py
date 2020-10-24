

import numpy as np 
import math
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import os, subprocess
import matplotlib.patches as mpatches

from matplotlib import cm	
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

import seaborn as sns

import matplotlib.transforms as mtransforms
import cv2
import imutils

import glob
import random 

import datahandler as dh
from param import Param 

# defaults
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 2.5

import matplotlib
matplotlib.use('Agg')

def has_figs():
	if len(plt.get_fignums()) > 0:
		return True
	else:
		return False


def save_figs(filename):
	'''
	Saves all open figures into a pdf
	'''

	# Make sure the directory exists, otherwise create it
	file_dir,  file_name = os.path.split(filename)

	if len(file_dir) >0 and not (os.path.isdir(file_dir)):
		os.makedirs(file_dir)

	# Open the PDF and save each of the figures 
	fn = os.path.join( os.getcwd(), filename)
	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()


def open_figs(filename):
	pdf_path = os.path.join( os.getcwd(), filename)
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])

def merge_figs(pdfs,result_fn):

	from PyPDF2 import PdfFileMerger

	merger = PdfFileMerger()

	# write new one 
	for pdf in pdfs:
	    merger.append(pdf)
	merger.write(result_fn)
	merger.close()

	# delete old files 
	for pdf in pdfs: 
		os.remove(pdf)

def make_fig():
	return plt.subplots()


def calc_idx_collisions(pos_x,pos_y):
	'''
	# Function to detect when collisions occur in a trajectory
	# We really only need to do this once but that's a bit beyond me at the moment
	# Calculates the indicies of captures by detecting large jumps in the array
	'''

	#print("\tChecking for collisions\n")
	idx = np.empty(0,int)

	for ii in range(0,pos_x.size-2):
		dist_between_steps = (pos_x[ii+1]-pos_x[ii])**2 + (pos_y[ii+1]-pos_y[ii])**2
		if (dist_between_steps > 0.1**2):
			#print("\t\tCollision at idx: "+str(ii))
			idx = np.append(idx,ii)

	return idx


def make_gif(sim_result):
	
	import imageio, glob

	# make gif directory 
	gif_dir = '../gif/'
	gif_name = gif_dir + 'movie.gif'
	format_dir(gif_dir) 

	# save images to 
	for timestep,time in enumerate(sim_result["times"]):
		fig,ax = plot_nodes(sim_result,timestep)
		fig.savefig('{}{}.png'.format(gif_dir,timestep))

	images = []
	for filename in sorted(glob.glob(gif_dir + '*')):
		images.append(imageio.imread(filename))

	duration = 0.5 
	imageio.mimsave(gif_name, images, duration = duration)

def plot_panagou(R,R_nom,I,states,param,plot_isochrones=True,plot_nominal=True,plot_intersections=True,plot_states=True):

	num_robots, num_times, num_thetas, state_dim = R.shape

	colors = get_2_colors(num_robots,len(param.team_1_idxs))

	intersection_color = 'yellow'
	goal_color = 'green'
	nominal_color = 'black'
	df_alpha = 0.2

	fig,ax = plt.subplots()
	for i_robot in range(num_robots):

		if plot_isochrones:
			for i_time in range(num_times):
				ax.plot(R[i_robot,i_time,:,0],R[i_robot,i_time,:,1],color=colors[i_robot],marker='o',alpha=df_alpha)
				ax.plot([R[i_robot,i_time,-1,0],R[i_robot,i_time,0,0]],
					[R[i_robot,i_time,-1,1],R[i_robot,i_time,0,1]],color=colors[i_robot],marker='o',alpha=df_alpha)
		
		if plot_nominal:
			ax.plot(R_nom[i_robot,:,0],R_nom[i_robot,:,1],color=nominal_color,marker='o',alpha=df_alpha)

		if plot_intersections:
			for (ii_robot,jj_robot), intersections in I.items():
				if i_robot == ii_robot or i_robot == jj_robot: 
					for (ii_theta,jj_theta,ii_time) in intersections:
						ax.plot(R[ii_robot,ii_time,ii_theta,0],R[ii_robot,ii_time,ii_theta,1],\
							color=intersection_color,marker='s',alpha=df_alpha)
						ax.plot(R[jj_robot,ii_time,jj_theta,0],R[jj_robot,ii_time,jj_theta,1],\
							color=intersection_color,marker='s',alpha=df_alpha)

		if plot_states: 
			for i_robot in range(num_robots):
				ax.plot(states[:,i_robot,0],states[:,i_robot,1],color=colors[i_robot])

	ax.plot(param.goal[0],param.goal[1],color=goal_color,marker='*')
	ax.set_xlim([param.env_xlim[0],param.env_xlim[1]])
	ax.set_ylim([param.env_ylim[0],param.env_ylim[1]])
	ax.grid(True)


def plot_sa_pairs(sampled_sa_pairs,sim_result,team):

	team_1_idxs = sim_result["param"]["team_1_idxs"]
	team_2_idxs = sim_result["param"]["team_2_idxs"]
	action_list = sim_result["param"]["actions"]
	env_xlim = sim_result["param"]["env_xlim"]
	env_ylim = sim_result["param"]["env_ylim"]
	tag_radius = sim_result["param"]["robot_types"]["standard_robot"]["tag_radius"]
	goal = sim_result["param"]["goal"]

	team_1_color = 'blue'
	team_2_color = 'orange'
	action_color = 'black'
	best_action_color = 'red'
	goal_color = 'green'

	if team == "a":
		idxs = team_1_idxs
	else:
		idxs = team_2_idxs

	fig,axs = plt.subplots(nrows=3,ncols=3) 

	for i_ax, (states,actions) in enumerate(sampled_sa_pairs):

		ax = axs[ int(np.floor(i_ax/3)), np.remainder(i_ax,3)]
		ax.set_xlim(env_xlim)
		ax.set_ylim(env_ylim)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')

		ax.add_patch(mpatches.Circle(goal, tag_radius, color=goal_color,alpha=0.5))

		for robot_idx, (state_per_robot, action_per_robot) in enumerate(zip(states,actions)):

			if robot_idx in team_1_idxs:
				color = team_1_color
			elif robot_idx in team_2_idxs:
				color = team_2_color

			ax.scatter(state_per_robot[0],state_per_robot[1],marker='o',color=color)
			ax.arrow(state_per_robot[0],state_per_robot[1],state_per_robot[2],state_per_robot[3],color=color,alpha=0.5)

			if robot_idx in idxs: 

				for direction, p in zip(action_list,action_per_robot):
					if p == np.max(action_per_robot):
						color = best_action_color
					else: 
						color = action_color

					dist = np.linalg.norm(direction,2)
					if dist > 0:
						ax.arrow(state_per_robot[0],state_per_robot[1],direction[0]*p/dist,direction[1]*p/dist,color=color,alpha=0.5)
					elif dist == 0:
						ax.arrow(state_per_robot[0],state_per_robot[1],0,1e-3,color=color,alpha=0.5)


def plot_oa_pairs(sampled_oa_pairs,abs_goal,team,rsense,action_list,env_length):

	team_1_color = 'blue'
	team_2_color = 'orange'
	action_color = 'black'
	best_action_color = 'red'
	goal_color = 'green'

	if team == "a":
		self_color = team_1_color
	elif team == "b":
		self_color = team_2_color

	fig,axs = plt.subplots(nrows=3,ncols=3) 

	for i_ax, (o_a,o_b,goal,actions) in enumerate(sampled_oa_pairs):

		ax = axs[int(np.floor(i_ax/3)), np.remainder(i_ax,3)]
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

		ax.set_xlim([-env_length,env_length])
		ax.set_ylim([-env_length,env_length])

		ax.scatter(0,0,color=self_color)
		ax.scatter(goal[0],goal[1],color=goal_color)
		ax.add_patch(mpatches.Circle((0,0), rsense, color='black',alpha=0.1))

		num_a = int(o_a.shape[0]/4)
		num_b = int(o_b.shape[0]/4)
		for robot_idx in range(num_a):
			ax.scatter(o_a[robot_idx*4],o_a[robot_idx*4+1],color=team_1_color)
			ax.arrow(o_a[robot_idx*4],o_a[robot_idx*4+1],o_a[robot_idx*4+2],o_a[robot_idx*4+3],color=team_1_color,alpha=0.5)
		for robot_idx in range(num_b):
			ax.scatter(o_b[robot_idx*4],o_b[robot_idx*4+1],color=team_2_color)
			ax.arrow(o_b[robot_idx*4],o_b[robot_idx*4+1],o_b[robot_idx*4+2],o_b[robot_idx*4+3],color=team_2_color,alpha=0.5)
		for direction, p in zip(action_list,actions):
			if p == np.max(actions):
				color = best_action_color
			else: 
				color = action_color

			dist = np.linalg.norm(direction,2)
			if dist > 0:
				ax.arrow(0,0,direction[0]*p/dist,direction[1]*p/dist,color=color,alpha=0.5)
			elif dist == 0:
				ax.arrow(0,0,0,1e-3,color=color,alpha=0.5)



def plot_loss(losses,lrs,team):

	losses = np.array(losses)

	fig,ax = plt.subplots()

	ax.plot(losses[:,0],label='train')
	ax.plot(losses[:,1],label='test')
	
	ax.legend()
	ax.set_ylabel('mse')
	ax.set_xlabel('epoch')
	if (losses > 0).all():
		ax.set_yscale('log')
	ax.set_title('Team {}'.format(team))
	ax.grid(True)

	# plot losses
	color = 'red'
	ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
	ax2.plot(np.array(lrs), color=color)
	ax2.set_ylabel('learning rate', color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()


def get_colors(param):

	colors = []

	start, stop = 0.4, 0.7
	cm_subsection = np.linspace(start, stop, param["num_nodes"]) 

	colors_a = [ cm.Blues(x) for x in cm_subsection]
	colors_b = [ cm.Oranges(x) for x in cm_subsection]

	colors = []
	for i in range(param["num_nodes"]):
		if i < param["num_nodes_A"]:
			colors.append(colors_a[i])
		else:
			colors.append(colors_b[i])

	return colors

def get_2_colors(total,split):

	colors = []

	start, stop = 0.4, 0.7
	cm_subsection = np.linspace(start, stop, total) 

	colors_a = [ cm.Blues(x) for x in cm_subsection]
	colors_b = [ cm.Oranges(x) for x in cm_subsection]

	colors = []
	for i in range(total):
		if i < split:
			colors.append(colors_a[i])
		else:
			colors.append(colors_b[i])

	return colors


def get_n_colors(n,cmap=None):
	colors = []
	cm_subsection = np.linspace(0, 1, n)

	if cmap is None:
		cmap = cm.tab20

	colors = [ cmap(x) for x in cm_subsection]
	# colors = [ cm.tab20(x) for x in cm_subsection]
	return colors


def plot_tree_results(sim_result,title=None): 

	def team_1_reward_to_gamma(reward_1):
		# gamma = reward_1 * 2 - 1 
		gamma = reward_1 
		return gamma 

	states = sim_result["states"]
	actions = sim_result["actions"]

	# states = sim_result["states"]
	# actions = sim_result["actions"]	

	nt, nrobots, state_dim = states.shape 

	if "times" in sim_result.keys():
		times = sim_result["times"]
	else: 
		times = range(nt)
	if "rewards" in sim_result.keys():
		rewards = sim_result["rewards"]
	else: 
		rewards = np.nan*np.ones((nt,2))

	team_1_idxs = sim_result["param"]["team_1_idxs"]
	num_nodes = sim_result["param"]["num_nodes"]
	goal = sim_result["param"]["goal"]
	tag_radius = sim_result["param"]["robots"][0]["tag_radius"]
	env_xlim = sim_result["param"]["env_xlim"]	
	env_ylim = sim_result["param"]["env_ylim"]	

	team_1_color = 'blue'
	team_2_color = 'orange'
	goal_color = 'green'

	colors = get_colors(sim_result["param"])

	fig,axs = plt.subplots(nrows=2,ncols=2,constrained_layout=True)

	# state space
	ax = axs[0,0]
	ax.grid(True)
	ax.set_aspect('equal')
	ax.set_title('State Space')
	ax.add_patch(mpatches.Circle(goal, tag_radius, color=goal_color,alpha=0.5))
	for i in range(num_nodes):
		for t in range(states.shape[0]):
			if np.isfinite(states[t,i]).all():
				ax.add_patch(mpatches.Circle(states[t,i,0:2], sim_result["param"]["robots"][i]["tag_radius"], \
					color=colors[i],alpha=0.2,fill=False))
				ax.arrow(states[t,i,0],states[t,i,1],states[t,i,2],states[t,i,3],color=colors[i])
		ax.plot(states[:,i,0],states[:,i,1],linewidth=3,color=colors[i])
		ax.scatter(states[:,i,0],states[:,i,1],marker='o',color=colors[i],alpha=0.75)
	ax.set_xlim([env_xlim[0],env_xlim[1]])
	ax.set_ylim([env_ylim[0],env_ylim[1]])

	# value func
	ax = axs[0,1] 
	ax.grid(True)
	ax.set_title('Value Function')
	ax.plot(times,rewards[:,0],color='black',alpha=0.75,label='truth')
	ax.set_ylim([0,1])

	path_value_fnc = None
	if "path_value_fnc" in sim_result["param"]["policy_dict_a"]: 
		path_value_fnc = sim_result["param"]["policy_dict_a"]["path_value_fnc"]
	elif "path_value_fnc" in sim_result["param"]["policy_dict_b"]:  # weird logic here in case of "MCTS(unbiased) vs MCTS(biased)"
		path_value_fnc = sim_result["param"]["policy_dict_b"]["path_value_fnc"]

	if path_value_fnc is not None: 

		# from learning.continuous_emptynet import ContinuousEmptyNet
		from learning.value_emptynet import ValueEmptyNet
		from learning_interface import format_data_value, global_to_value 
		from param import Param 
		import torch 

		param_obj = Param()
		param_obj.from_dict(sim_result["param"])

		with torch.no_grad():
			model = ValueEmptyNet(param_obj,"cpu")
			model.load_state_dict(torch.load(path_value_fnc))

			mus = [] 
			sigmas = [] 
			n_a = param_obj.num_nodes_A
			n_b = param_obj.num_nodes_B
			for k,(t,n_rg) in enumerate(zip(times,sim_result["n_rgs"])):
				v_a,v_b = global_to_value(param_obj,states[k,:,:])
				v_a,v_b,n_a,n_b,n_rg = format_data_value(v_a,v_b,n_a,n_b,n_rg)
				_,mu,logvar = model(v_a,v_b,n_a,n_b,n_rg,training=True)

				mu = mu.detach().numpy().squeeze()
				sigma = torch.sqrt(torch.exp(logvar)).detach().numpy().squeeze()

				mus.append(mu)
				sigmas.append(sigma)

		mus = np.array(mus)
		sigmas = np.array(sigmas)

		ax.plot(times,mus,color='green',label='learned') 
		ax.fill_between(times,mus-sigmas,mus+sigmas,color='green',alpha=0.5) 
		ax.legend()

	# time varying velocity
	ax = axs[1,0]
	ax.grid(True)
	ax.set_title('Speed Profile')
	for i in range(num_nodes):
		ax.axhline(sim_result["param"]["robots"][i]["speed_limit"],color=colors[i],linestyle='--')
		ax.plot(times,np.linalg.norm(states[:,i,2:],axis=1),color=colors[i])
	ax.set_ylim(bottom=0)

	# time varying acc
	ax = axs[1,1]
	ax.grid(True)
	ax.set_title('Acceleration Profile')
	for i in range(num_nodes):
		ax.axhline(sim_result["param"]["robots"][i]["acceleration_limit"],color=colors[i],linestyle='--')
		ax.plot(times,np.linalg.norm(actions[:,i],axis=1),color=colors[i])
	ax.set_ylim(bottom=0)

	# velociy-space trajectories
	# ax = axs[2,0]
	# ax.grid(True)
	# ax.set_title('Velocity Trajectories')
	# for i in range(num_nodes):
	# 	# ax.axhline(sim_result["param"]["robots"][i]["speed_limit"],color=colors[i],linestyle='--')
	# 	ax.plot(states[:,i,2],states[:,i,3],color=colors[i])
	# 	arrow_dirs_x = states[1:,i,2] - states[0:-1,i,2]
	# 	arrow_dirs_y = states[1:,i,3] - states[0:-1,i,3]
	# 	for k,t in enumerate(times[0:-1]):
	# 		ax.arrow(states[k,i,2],states[k,i,3],arrow_dirs_x[k],arrow_dirs_y[k],color=colors[i])
	# ax.set_xlabel('v_x')
	# ax.set_ylabel('v_y')

	# # acceleration-space trajectories 
	# ax = axs[2,1]
	# ax.grid(True)
	# ax.set_title('Acceleration Trajectories')
	# for i in range(num_nodes):
	# 	# ax.axhline(sim_result["param"]["robots"][i]["speed_limit"],color=colors[i],linestyle='--')
	# 	ax.plot(actions[:,i,0],actions[:,i,1],color=colors[i])
	# 	arrow_dirs_x = actions[1:,i,0] - actions[0:-1,i,0]
	# 	arrow_dirs_y = actions[1:,i,1] - actions[0:-1,i,1]
	# 	for k,t in enumerate(times[0:-1]):
	# 		ax.arrow(actions[k,i,0],actions[k,i,1],arrow_dirs_x[k],arrow_dirs_y[k],color=colors[i])
	# ax.set_xlabel('a_x')
	# ax.set_ylabel('a_y')

	# path_policy_ = None
	# if "path_value_fnc" in sim_result["param"]["policy_dict_a"]: 
	# 	path_value_fnc = sim_result["param"]["policy_dict_a"]["path_value_fnc"]

	# 

	# for team in ["a","b"]:

	# if path_value_fnc is not None: 
	# if path_value_fnc is not None: 

	# 	# from learning.continuous_emptynet import ContinuousEmptyNet
	# 	from learning.value_emptynet import ValueEmptyNet
	# 	from learning_interface import format_data_value, global_to_value 
	# 	from param import Param 
	# 	import torch 

	# 	param_obj = Param()
	# 	param_obj.from_dict(sim_result["param"])

	# 	with torch.no_grad():
	# 		model = ValueEmptyNet(param_obj,"cpu")
	# 		model.load_state_dict(torch.load(path_value_fnc))

	# 		mus = [] 
	# 		sigmas = [] 
	# 		n_a = param_obj.num_nodes_A
	# 		n_b = param_obj.num_nodes_B
	# 		for k,(t,n_rg) in enumerate(zip(times,sim_result["n_rgs"])):
	# 			v_a,v_b = global_to_value(param_obj,states[k,:,:])
	# 			v_a,v_b,n_a,n_b,n_rg = format_data_value(v_a,v_b,n_a,n_b,n_rg)
	# 			_,mu,logvar = model(v_a,v_b,n_a,n_b,n_rg,training=True)

	# 			mu = mu.detach().numpy().squeeze()
	# 			sigma = torch.sqrt(torch.exp(logvar)).detach().numpy().squeeze()

	# 			mus.append(mu)
	# 			sigmas.append(sigma)

	# 			# cppstate = cpp_interface.state_to_cpp_game_state(states[k,:,:],"a",param_obj.team_1_idxs,param_obj.team_2_idxs)
	# 			# cppvalue = valPred.estimate(cppstate, param_obj.goal)

	# 			# print(value, cppvalue)

	# 	mus = np.array(mus)
	# 	sigmas = np.array(sigmas)

	# 	ax.plot(times,mus,color='green',label='learned') 
	# 	ax.fill_between(times,mus-sigmas,mus+sigmas,color='green',alpha=0.5) 
	# 	ax.legend()

	# add figure title 
	if title is not None: 
		fig.suptitle(title)	

	# tree vis 
	if len(sim_result["trees"]) > 0:

		max_trees = 2
		if len(sim_result["trees"]) > max_trees:
			sim_result["trees"] = sim_result["trees"][0:max_trees]

		fig,axs = plt.subplots(nrows=len(sim_result["trees"]),ncols=3,squeeze=False,constrained_layout=True)
		# fig,axs = plt.subplots(nrows=len(sim_result["trees"]),ncols=2,squeeze=False,constrained_layout=True)

		for i_tree, data in enumerate(sim_result["trees"]):

			# [number of nodes x (parentIdx, reward, isBest, \{position, velocity\}_{for all robots})]

			tree_timestep = sim_result["param"]["tree_timestep"] 
			random_factor = 3.0 
			goal = sim_result["param"]["goal"]
			acceleration_lims = sim_result["param"]["robots"][0]["acceleration_limit"]
			tree_team_1_idxs = sim_result["tree_params"][i_tree]["tree_team_1_idxs"]
			tree_time = sim_result["tree_params"][i_tree]["time"]
			tree_robot_idx = sim_result["tree_params"][i_tree]["robot_idx"]
			tree_colors = get_2_colors(2,1)

			# first col, position 
			ax = axs[i_tree,0]
			segments = []
			best_segments = []
			cs = []
			linewidths = []

			num_nodes = (data.shape[1]-3) // 4 

			for row in data:
				parentIdx = int(row[0])
				if parentIdx >= 0:
					for i in range(0, num_nodes):
						segments.append([row[(3+4*i):(5+4*i)], data[parentIdx][(3+4*i):(5+4*i)]])
						if row[2] == 1 and data[parentIdx][2] == 1:
							best_segments.append(segments[-1])
						reward = np.min((1,random_factor*row[1])) if i in tree_team_1_idxs else 1.0-row[1]
						color = tree_colors[0] if i in tree_team_1_idxs else tree_colors[1]

						linewidths.append(reward)
						# cs.append((colors[i][0],colors[i][1],colors[i][2],reward))
						cs.append((color[0],color[1],color[2],reward))

			ax.grid(True)
			ax.axis('equal')
			ax.set_title('STP at i={},t={}'.format(tree_robot_idx,tree_time))
			ln_coll = matplotlib.collections.LineCollection(segments, colors=cs, linewidth=2*linewidths)
			ax.add_collection(ln_coll)
			ln_coll = matplotlib.collections.LineCollection(best_segments, colors='k', zorder=3)
			ax.add_collection(ln_coll)
			ax.add_patch(mpatches.Circle(goal, tag_radius, color=goal_color,alpha=0.5))

			ax.set_xlim([env_xlim[0],env_xlim[1]])
			ax.set_ylim([env_ylim[0],env_ylim[1]])

			# second col, velocity embedding 
			ax = axs[i_tree,1]
			segments = []
			best_segments = []
			cs = []

			for row in data:
				parentIdx = int(row[0])
				if parentIdx >= 0:
					for i in range(0, num_nodes):
						segments.append([row[(5+4*i):(7+4*i)], data[parentIdx][(5+4*i):(7+4*i)]])
						if row[2] == 1 and data[parentIdx][2] == 1:
							best_segments.append(segments[-1])
						reward = np.min((1,random_factor*row[1])) if i in tree_team_1_idxs else 1.0-row[1]
						color = tree_colors[0] if i in tree_team_1_idxs else tree_colors[1]
						linewidths.append(reward)
						# cs.append((colors[i][0],colors[i][1],colors[i][2],reward))
						cs.append((color[0],color[1],color[2],reward))

			ax.grid(True)
			ax.axis('equal')
			ax.set_title('STV at i={},t={}'.format(tree_robot_idx,tree_time))
			ln_coll = matplotlib.collections.LineCollection(segments, colors=cs, linewidth=2*linewidths)
			ax.add_collection(ln_coll)
			ln_coll = matplotlib.collections.LineCollection(best_segments, colors='k', zorder=3)
			ax.add_collection(ln_coll)
			ax.set_xlim([-acceleration_lims,acceleration_lims])
			ax.set_ylim([-acceleration_lims,acceleration_lims])

			# third col, histogram of tree depth
			ax = axs[i_tree,2]
			# ax = axs[i_tree,1]
			num_nodes_by_depth = defaultdict(int)
			for row in data:
				parentIdx = int(row[0])
				depth = 0
				while parentIdx >= 0:
					parentIdx = int(data[parentIdx,0])
					depth += 1
				num_nodes_by_depth[depth] += 1

			ax.set_title('Tree Nodes By Depth')
			ax.bar(list(num_nodes_by_depth.keys()), num_nodes_by_depth.values(), color='g')

			# num_children = np.zeros(data.shape[0])
			# for row in data:
			# 	parentIdx = int(row[0])
			# 	if parentIdx >= 0:
			# 		num_children[parentIdx] += 1
			# axs[0,2].hist(num_children[num_children != 0])

			# fig.tight_layout()

		if title is not None: 
			fig.suptitle(title)

def plot_training_value(df_param,batched_fns,path_to_model):
	import torch 
	# from learning.continuous_emptynet import ContinuousEmptyNet
	# from learning.gaussian_emptynet import GaussianEmptyNet
	from learning.value_emptynet import ValueEmptyNet
	from learning_interface import format_data_value

	def gaussian(x, mu, sigma):
	    return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2.)/2)

	# - vis 
	team_1_color = 'blue'
	team_2_color = 'orange'
	goal_color = 'green'
	self_color = 'black'
	LIMS = df_param.robot_types["standard_robot"]["acceleration_limit"]*np.array([[-1,1],[-1,1]])
	rsense = df_param.robot_types["standard_robot"]["r_sense"]
	env_xlim = df_param.env_xlim 
	env_ylim = df_param.env_ylim 
	nbins = 20
	num_vis = 10
	n_samples = 100
	eps = 0.01  

	v_as,v_bs,n_as,n_bs,n_rgs,values = [],[],[],[],[],[]
	for batched_fn in batched_fns:
		v_a,v_b,n_a,n_b,n_rg,value = dh.read_sv_batch(batched_fn)
		v_as.extend(v_a)
		v_bs.extend(v_b)
		n_as.extend(n_a)
		n_bs.extend(n_b)
		n_rgs.extend(n_rg)
		values.extend(value)

	# load models
	model = ValueEmptyNet(df_param,"cpu")
	model.load_state_dict(torch.load(path_to_model))

	# pick random observations	
	idxs = np.random.choice(len(v_as),num_vis)

	for i_state in range(num_vis):

		# pick random observation 

		# select candidate observations 
		candidate = (v_as[idxs[i_state]],v_bs[idxs[i_state]],n_as[idxs[i_state]],n_bs[idxs[i_state]],n_rgs[idxs[i_state]])
		print('candidate {}/{}: {}'.format(i_state,num_vis,candidate))

		fig, axs = plt.subplots(nrows=1,ncols=2,squeeze=False)

		# append all eps-close ones and record dataset values 
		neighbors = [candidate] 
		dataset_values = [values[idxs[i_state]]]
		for v_a,v_b,n_a,n_b,n_rg,value in zip(v_as,v_bs,n_as,n_bs,n_rgs,values):
			if v_a.shape == candidate[0].shape and \
				v_b.shape == candidate[1].shape and \
				n_a == candidate[2] and \
				n_b == candidate[3] and \
				n_rg == candidate[4]: 

				if (np.linalg.norm(v_a - candidate[0]) <= eps) and \
					(np.linalg.norm(v_b - candidate[1]) <= eps): 

					neighbors.append((v_a,v_b,n_a,n_b,n_rg))
					dataset_values.append(value)

		# query model for all neighbors 
		model_values = []
		for v_a,v_b,n_a,n_b,n_rg in neighbors: 
			v_a,v_b,n_a,n_b,n_rg = format_data_value(v_a,v_b,n_a,n_b,n_rg)
			model_value = model(v_a,v_b,n_a,n_b,n_rg)
			model_values.append(model_value.detach().numpy().squeeze())

		# query model training for candidate 
		x = np.linspace(0,1,50)
		v_a,v_b,n_a,n_b,n_rg = format_data_value(v_a,v_b,n_a,n_b,n_rg)
		_,mu,logvar = model(v_a,v_b,n_a,n_b,n_rg,training=True)
		mu = mu.detach().numpy().squeeze()
		sigma = torch.sqrt(torch.exp(logvar)).detach().numpy().squeeze()
		y = gaussian(x,mu,sigma)

		# value func histogram  
		axs[0][1].set_title('value: n_a = {}, n_b = {}, n_rg = {}'.format(\
			int(candidate[2]),int(candidate[3]),int(candidate[4])))
		axs[0][1].hist(model_values, bins=20, range=[0,1],alpha=0.5, label="NN")
		axs[0][1].hist(dataset_values, bins=20, range=[0,1],alpha=0.5, label="data")
		axs[0][1].plot(x,y,color='green',alpha=0.5)
		axs[0][1].axvline(mu,color='green',alpha=0.5)
		axs[0][1].set_xlim([0,1])
		axs[0][1].set_xlabel('value')
		axs[0][1].set_ylabel('count')
		x0,x1 = axs[0][1].get_xlim()
		y0,y1 = axs[0][1].get_ylim()
		axs[0][1].set_aspect(abs(x1-x0)/abs(y1-y0))
		axs[0][1].legend()
		axs[0][1].grid(True)

		# game state encoding  

		# - goal 
		axs[0][0].scatter(0,0,color=goal_color,alpha=0.5)

		# - neighbors 
		num_a = int(len(candidate[0])/4)
		num_b = int(len(candidate[1])/4)
		goal = np.array([df_param.goal[0],df_param.goal[1],0,0])
		for robot_idx in range(num_a):
			# v_a = s^j - g
			v_a_idxs = np.arange(4) + 4*robot_idx 
			sj = candidate[0][v_a_idxs] + goal 
			axs[0][0].scatter(sj[0],sj[1],color=team_1_color)
			axs[0][0].arrow(sj[0],sj[1],sj[2],sj[3],color=team_1_color,alpha=0.5)
		for robot_idx in range(num_b):
			# v_b = s^j - g
			v_b_idxs = np.arange(4) + 4*robot_idx 
			sj = candidate[1][v_b_idxs] + goal 
			axs[0][0].scatter(sj[0],sj[1],color=team_2_color)
			axs[0][0].arrow(sj[0],sj[1],sj[2],sj[3],color=team_2_color,alpha=0.5)

		# - arrange  
		l = np.max((np.abs(axs[0][0].get_xlim()),np.abs(axs[0][0].get_ylim())))
		axs[0][0].set_xlim([-l,l])
		axs[0][0].set_ylim([-l,l])
		# axs[0][0].set_xlim([np.min((axs[0][0].get_xlim()[0],axs[0][0].get_ylim()[0])),np.max((axs[0][0].get_xlim()[1],axs[0][0].get_ylim()[1]))])
		# axs[0][0].set_ylim([np.min((axs[0][0].get_xlim()[0],axs[0][0].get_ylim()[0])),np.max((axs[0][0].get_xlim()[1],axs[0][0].get_ylim()[1]))])
		# axs[0][0].set_xlim([np.max((-rsense,-env_xlim[1])),np.min((rsense,env_xlim[1]))])
		# axs[0][0].set_ylim([np.max((-rsense,-env_ylim[1])),np.min((rsense,env_ylim[1]))])
		# axs[0][0].set_xlim([-rsense,rsense])
		# axs[0][0].set_ylim([-rsense,rsense])

		# - sensing radius 
		axs[0][0].add_patch(mpatches.Circle((0,0), rsense, color='black',alpha=0.1))
		axs[0][0].set_title('game state: {}'.format(i_state))
		axs[0][0].set_aspect('equal')

		fig.tight_layout()




def plot_training(df_param,batched_fns,path_to_model):
	import torch 
	from learning.continuous_emptynet import ContinuousEmptyNet
	# from learning.gaussian_emptynet import GaussianEmptyNet
	from learning.policy_emptynet import PolicyEmptyNet
	from mice import format_data

	# - vis 
	team_1_color = 'blue'
	team_2_color = 'orange'
	goal_color = 'green'
	self_color = 'black'
	LIMS = df_param.robot_types["standard_robot"]["acceleration_limit"]*np.array([[-1,1],[-1,1]])
	rsense = df_param.robot_types["standard_robot"]["r_sense"]
	env_xlim = df_param.env_xlim 
	env_ylim = df_param.env_ylim 
	nbins = 20
	num_vis = 10
	n_samples = 100
	eps = 0.01  

	o_as,o_bs,goals,actions,weights = [],[],[],[],[]
	for batched_fn in batched_fns:
		o_a,o_b,goal,action,weight = dh.read_oa_batch(batched_fn,df_param.l_gaussian_on)
		o_as.extend(o_a)
		o_bs.extend(o_b)
		goals.extend(goal)
		actions.extend(action)
		weights.extend(weight)

	# load models
	if df_param.l_gaussian_on:
		model = PolicyEmptyNet(df_param,"cpu")
	else:
		model = ContinuousEmptyNet(df_param,"cpu")
	model.load_state_dict(torch.load(path_to_model))

	# pick random observations	
	idxs = np.random.choice(len(o_as),num_vis)

	for i_state in range(num_vis):

		# pick random observation 

		# select candidate observations 
		candidate = (o_as[idxs[i_state]],o_bs[idxs[i_state]],goals[idxs[i_state]])
		print('candidate {}/{}: {}'.format(i_state,num_vis,candidate))

		fig, axs = plt.subplots(nrows=2,ncols=2)

		# append all identical ones (should be # subsamples)
		conditionals = [] 
		dataset_actions = []
		dataset_weights = []
		for o_a,o_b,goal,action,weight in zip(o_as,o_bs,goals,actions,weights):
			if o_a.shape == candidate[0].shape and \
				o_b.shape == candidate[1].shape and \
				goal.shape == candidate[2].shape: 

				if (np.linalg.norm(o_a - candidate[0]) <= eps) and \
					(np.linalg.norm(o_b - candidate[1]) <= eps) and \
					(np.linalg.norm(goal - candidate[2]) <= eps):

					conditionals.append((o_a,o_b,goal))
					dataset_actions.append(action)
					dataset_weights.append(weight)

		if df_param.l_gaussian_on: 
			# dataset_actions = mean 
			# dataset_weights = variance 
			mean = np.array(dataset_actions)
			sd = np.sqrt(np.array(dataset_weights))
			eps2 = np.random.normal(size=mean.shape)
			dataset_actions = mean + sd * eps2
			for m, s in zip(mean, sd):
				axs[1][0].add_patch(Ellipse(m, width=s[0] * 2, height=s[1] * 2, alpha=0.5))
		else: 
			dataset_weights = dataset_weights / sum(dataset_weights)
			choice_idxs = np.random.choice(len(dataset_actions),n_samples,p=dataset_weights)
			weighted_dataset_actions = np.array([dataset_actions[choice_idx] for choice_idx in choice_idxs])
			dataset_actions = weighted_dataset_actions
		
		# print('conditionals',conditionals)
		# print('dataset_actions',dataset_actions)

		# query model 
		model_actions = [] 

		if df_param.mice_testing_on: 
			for _ in range(n_samples):
				o_a,o_b,goal = format_data(candidate[0],candidate[1],candidate[2])

				if df_param.l_gaussian_on:
					with torch.no_grad():
						_, mu, logvar = model(o_a, o_b, goal, True)
					m = mu.numpy()
					s = torch.sqrt(torch.exp(logvar)).numpy()
					axs[1][1].add_patch(Ellipse(m[0], width=s[0,0] * 2, height=s[0,1] * 2, alpha=0.5))
				else: 
					policy = model(o_a,o_b,goal)
					model_actions.append(policy.detach().numpy())
					
		else:
			for o_a,o_b,goal in conditionals:
				o_a,o_b,goal = format_data(o_a,o_b,goal)
				if df_param.l_gaussian_on:
					with torch.no_grad():
						_, mu, logvar = model(o_a, o_b, goal, True)
					m = mu.numpy()
					s = torch.sqrt(torch.exp(logvar)).numpy()
					axs[1][1].add_patch(Ellipse(m[0], width=s[0,0] * 2, height=s[0,1] * 2, alpha=0.5))
				else:
					for _ in range(n_samples):
						policy = model(o_a,o_b,goal)
						model_actions.append(policy.detach().numpy())

		# convert for easy plot
		model_actions = np.array(model_actions).squeeze()
		dataset_actions = np.array(dataset_actions)
		
		# value func histogram  
		# axs[0][1].set_title('value')
		# axs[0][1].hist(model_values, bins=20, range=[0,1],alpha=0.5, label="NN")
		# axs[0][1].hist(dataset_values, bins=20, range=[0,1],alpha=0.5, label="data")
		# axs[0][1].set_xlim([0,1])
		# axs[1][0].set_xlabel('value')
		# axs[1][0].set_ylabel('count')
		# axs[0][1].legend()

		# game state encoding  
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
		l = np.max((np.abs(axs[0][0].get_xlim()),np.abs(axs[0][0].get_ylim())))
		axs[0][0].set_xlim([-l,l])
		axs[0][0].set_ylim([-l,l])
		# axs[0][0].set_xlim([np.min((axs[0][0].get_xlim()[0],axs[0][0].get_ylim()[0])),np.max((axs[0][0].get_xlim()[1],axs[0][0].get_ylim()[1]))])
		# axs[0][0].set_ylim([np.min((axs[0][0].get_xlim()[0],axs[0][0].get_ylim()[0])),np.max((axs[0][0].get_xlim()[1],axs[0][0].get_ylim()[1]))])
		# axs[0][0].set_xlim([np.max((-rsense,-env_xlim[1])),np.min((rsense,env_xlim[1]))])
		# axs[0][0].set_ylim([np.max((-rsense,-env_ylim[1])),np.min((rsense,env_ylim[1]))])
		# axs[0][0].set_xlim([-rsense,rsense])
		# axs[0][0].set_ylim([-rsense,rsense])

		# - sensing radius 
		axs[0][0].add_patch(mpatches.Circle((0,0), rsense, color='black',alpha=0.1))
		
		axs[0][0].set_title('game state: {}'.format(i_state))
		axs[0][0].set_aspect('equal')

		# - arrange 
		axs[1][0].set_title('mcts: {}'.format(i_state))
		axs[1][1].set_title('model: {}'.format(i_state))
		axs[1][0].set_xlabel('x-action')
		axs[1][1].set_xlabel('x-action')
		axs[1][0].set_ylabel('y-action')

		if not df_param.l_gaussian_on:
			
			# fig: histograms of model/dataset in action space
			# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
			xedges = np.linspace(LIMS[0,0],LIMS[0,1],nbins) 
			yedges = np.linspace(LIMS[1,0],LIMS[1,1],nbins) 

			h_mcts, xedges, yedges = np.histogram2d(dataset_actions[:,0],dataset_actions[:,1],bins=(xedges,yedges),range=LIMS) 
			h_model, xedges, yedges = np.histogram2d(model_actions[:,0],model_actions[:,1],bins=(xedges,yedges),range=LIMS) 

			h_mcts = h_mcts.T / np.sum(np.sum(h_mcts))
			h_model = h_model.T / np.sum(np.sum(h_model))

			# im1 = axs[1][0].imshow(h_mcts,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]],vmin=0,vmax=1)
			# im2 = axs[1][1].imshow(h_model,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]],vmin=0,vmax=1)
			
			im1 = axs[1][0].imshow(h_mcts,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]])
			im2 = axs[1][1].imshow(h_model,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]])

			fig.colorbar(im1, ax=axs[1][0])
			fig.colorbar(im2, ax=axs[1][1])

		else:
			axs[1][0].set_xlim(LIMS[0])
			axs[1][0].set_ylim(LIMS[1])
			axs[1][0].set_aspect('equal')
			axs[1][1].set_xlim(LIMS[0])
			axs[1][1].set_ylim(LIMS[1])
			axs[1][1].set_aspect('equal')

		fig.tight_layout()

def plot_exp1_results(all_sim_results):

	# group by tree size, rollout policy and case number 
	results = defaultdict(list)
	tree_sizes = set()
	cases = set()
	for sim_result in all_sim_results:

		tree_size = sim_result["param"]["tree_size"]
		policy = sim_result["param"]["glas_rollout_on"]
		case = sim_result["param"]["case"]

		tree_sizes.add(tree_size)
		cases.add(case)

		key = (tree_size,policy,case)
		results[key].append(sim_result)
		
	tree_sizes = np.array(list(tree_sizes))
	tree_sizes = np.sort(tree_sizes)

	num_trials = np.min((10,sim_result["param"]["num_trials"]))
	num_trials_per_fig = 4 

	for case in cases: 

		for glas_rollout_on in sim_result["param"]["glas_rollout_on_cases"]:

			policy = 'GLAS' if glas_rollout_on else 'Random'
			suptitle = 'Rollout: {}, Case: {}'.format(policy,case)

			for i_trial in range(num_trials):

				if i_trial % num_trials_per_fig == 0:
					fig, axs = plt.subplots(nrows=np.min((num_trials_per_fig,num_trials - i_trial)),ncols=tree_sizes.shape[0],sharex='col', sharey='row')
					fig.suptitle(suptitle)
				
				for i_tree,tree_size in enumerate(tree_sizes):
			
					key = (tree_size,glas_rollout_on,case)
					sim_result = results[key][i_trial]

					times = sim_result["times"]
					states = sim_result["states"]
					team_1_idxs = sim_result["param"]["team_1_idxs"]
					num_nodes = sim_result["param"]["num_nodes"]
					goal = sim_result["param"]["goal"]
					tag_radius = sim_result["param"]["robots"][0]["tag_radius"]
					env_xlim = sim_result["param"]["env_xlim"]	
					env_ylim = sim_result["param"]["env_ylim"]	

					team_1_color = 'blue'
					team_2_color = 'orange'
					goal_color = 'green'

					colors = get_colors(sim_result["param"])

					ax = axs[i_trial % num_trials_per_fig,i_tree]

					# state space
					ax.grid(True)
					ax.set_aspect('equal')
					ax.add_patch(mpatches.Circle(goal, tag_radius, color=goal_color,alpha=0.5))
					for i in range(num_nodes):
						for t in range(states.shape[0]):
							ax.add_patch(mpatches.Circle(states[t,i,0:2], sim_result["param"]["robots"][i]["tag_radius"], \
								color=colors[i],alpha=0.2,fill=False))
						ax.plot(states[:,i,0],states[:,i,1],linewidth=1,color=colors[i])
						ax.scatter(states[:,i,0],states[:,i,1],s=10,marker='o',color=colors[i])
					ax.set_xlim([env_xlim[0],env_xlim[1]])
					ax.set_ylim([env_ylim[0],env_ylim[1]])

					ax.set_xticklabels([""])
					ax.set_yticklabels([""])

					if i_trial % num_trials_per_fig == 0: 
						ax.set_title('{}K'.format(tree_size/1000))

					if i_tree == 0: 
						ax.set_ylabel('Trial {}'.format(i_trial))

			fig.tight_layout()

def plot_exp4_results(all_sim_results):

	def policy_to_label(policy_dict):
		label = policy_dict["sim_mode"]
		label += policy_dict["team"]
		if "path_glas_model_a" in policy_dict.keys() and policy_dict["path_glas_model_a"] is not None: 
			label += ' ' + os.path.basename(policy_dict["path_glas_model_a"]).split('.')[0]
		if "path_glas_model_b" in policy_dict.keys() and policy_dict["path_glas_model_b"] is not None: 
			label += ' ' + os.path.basename(policy_dict["path_glas_model_b"]).split('.')[0]
		if "path_value_fnc" in policy_dict.keys() and policy_dict["path_value_fnc"] is not None: 
			label += ' ' + os.path.basename(policy_dict["path_value_fnc"]).split('.')[0]
		if policy_dict["sim_mode"] in ["MCTS","D_MCTS"]:
			label += ' |n|:{}'.format(policy_dict["mcts_tree_size"]).split('.')[0]
		return label

	nbins = 10

	# key = (trial, team, prediction_type, exp4_sim_mode)
	# value = (forallrobots, image)
	exp4_sim_modes = set()
	value_ims = defaultdict(list)
	policy_ims = defaultdict(list)
	sim_results_by_key = defaultdict(list)
	for sim_result in all_sim_results: 
		i_case = sim_result["param"]["i_case"]
		prediction_type = sim_result["param"]["exp4_prediction_type"]
		exp4_sim_mode = policy_to_label(sim_result["param"]["policy_dict"])
		team = sim_result["param"]["team"]
	
		key = (i_case, team, prediction_type, exp4_sim_mode)
		exp4_sim_modes.add(exp4_sim_mode)
	
		value_ims[key].append(sim_result["value_ims"])
		policy_ims[key].append(sim_result["policy_ims"])
		sim_results_by_key[key].append(sim_result)


	# print('sim_results_by_key',sim_results_by_key)
	# for (trial, team, prediction_type, exp4_sim_mode), data_value in value_ims.items():
	# 	print('trial',trial)
	# 	print('team',team)
	# exit()


	plotted = []
	for (i_case, team, prediction_type, exp4_sim_mode), data_value in value_ims.items():

		key = (i_case, team, prediction_type, exp4_sim_mode)

		if (i_case,team) in plotted:
			continue
		else:
			plotted.append((i_case,team))

		sim_result = sim_results_by_key[key][0]

		X = sim_result["X"]
		Y = sim_result["Y"]
		nominal_state = sim_result["nominal_state"]
		num_robots = sim_result["param"]["num_nodes"]
		goal = sim_result["param"]["goal"]
		team = sim_result["param"]["team"]
		env_xlim = sim_result["param"]["env_xlim"]
		env_ylim = sim_result["param"]["env_ylim"]
		team_1_idxs = sim_result["param"]["team_1_idxs"]
		team_2_idxs = sim_result["param"]["team_2_idxs"]
		team_idxs = team_1_idxs if team =="a" else team_2_idxs
		exp4_prediction_types = sim_result["param"]["exp4_prediction_types"]

		colors = get_colors(sim_result["param"])

		for robot_idx in range(num_robots):

			if not robot_idx in team_idxs: 
				continue
			
			fig,axs = plt.subplots(nrows=len(exp4_prediction_types)+1,ncols=len(exp4_sim_modes)//2,squeeze=False) 

			# fig title 
			team = "b"
			if robot_idx in team_1_idxs:
				team = "a"
			title = 'Initial Condition {}, Placing Robot From Team {}'.format(i_case,team)
			fig.suptitle(title)	

			curr_pt = -1
			for prediction_type in exp4_prediction_types: 
				curr_pt += 1
				curr_sm = -1
				for exp4_sim_mode in exp4_sim_modes:

					key = (i_case, team, prediction_type, exp4_sim_mode)

					if not key in value_ims.keys() or np.isnan(np.array(value_ims[key])[:,robot_idx,:,:]).any():
						continue

					curr_sm += 1

					ax = axs[curr_pt,curr_sm]

					data = np.mean(np.array(value_ims[key]),axis=0)

					# plot heatmap 
					# levels = MaxNLocator(nbins=nbins).tick_values(0,1)
					# levels = MaxNLocator(nbins=nbins).tick_values(value_ims[key][robot_idx,:,:].min(),value_ims[key][robot_idx,:,:].max())
					# im = ax.contourf(X,Y,value_ims[key][robot_idx,:,:],levels=levels)
					
					# im = ax.imshow(value_ims[key][robot_idx,:,:].T,origin='lower',extent=(X[0], X[-1], Y[0], Y[-1]))
					# im = ax.imshow(value_ims[key][robot_idx,:,:].T,origin='lower',extent=(X[0], X[-1], Y[0], Y[-1]),vmin=0,vmax=1)
					im = ax.imshow(data[robot_idx,:,:].T,origin='lower',extent=(X[0], X[-1], Y[0], Y[-1]),vmin=0,vmax=1)

					# plot state 
					ax.scatter(goal[0],goal[1],color='green')
					for robot_idx_j, robot_state_j in enumerate(nominal_state):
						if robot_idx_j == robot_idx: 
							continue 
						ax.scatter(robot_state_j[0],robot_state_j[1],color=colors[robot_idx_j])

					# labels 
					if curr_pt == 0:
						# ax.set_xlabel(prediction_type)
						ax.set_xlabel(exp4_sim_mode)
						ax.xaxis.set_label_position('top')
					if curr_sm == 0:
						# ax.set_ylabel(exp4_sim_mode)
						ax.set_ylabel(prediction_type)

					# arrange 
					ax.set_xticks(X)
					ax.set_yticks(Y)
					ax.set_xticklabels([])
					ax.set_yticklabels([])
					ax.grid(True,linestyle='-',linewidth=1,alpha=0.2,color='black')


			curr_sm = 0 
			for exp4_sim_mode in exp4_sim_modes: 

				key = (i_case, team, prediction_type, exp4_sim_mode)

				# plot policy quiver for each exp4simmode 
				# if not key in policy_ims.keys() or np.isnan(np.array(policy_ims[key])[:,robot_idx,:,:]).any():
				if not key in policy_ims.keys(): # or np.isnan(np.array(policy_ims[key])[:,robot_idx,:,:]).any():
					continue

				ax = axs[len(exp4_prediction_types),curr_sm]
				data = np.mean(np.array(policy_ims[key])[:,robot_idx,:,:],axis=0)
				C = np.linalg.norm(data,axis=2)
				ax.quiver(np.array(X),np.array(Y),data[:,:,0],data[:,:,1],width=0.01)
				ax.imshow(C.T,origin='lower',extent=(X[0], X[-1], Y[0], Y[-1]))

				if curr_sm == 0:
					ax.set_ylabel("Policy")
				curr_sm += 1

				# arrange 
				ax.set_xticks(X)
				ax.set_yticks(Y)
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				# ax.grid(True,linestyle='-',linewidth=1,alpha=0.2,color='black')

			



# def plot_exp4_results(all_sim_results):

# 	def get_initial_condition(param,X,Y):
# 		pos = np.zeros(2)
# 		pos[0] = param["state"][0][0]
# 		pos[1] = param["state"][0][1]
# 		i_x = np.where(pos[0] == X)[0][0]
# 		i_y = np.where(pos[1] == Y)[0][0]
# 		return pos, i_x, i_y 

# 	X = all_sim_results[0]["param"]["X"]
# 	Y = all_sim_results[0]["param"]["Y"]
# 	num_trials = all_sim_results[0]["param"]["num_trials"]
# 	exp4_sim_modes = all_sim_results[0]["param"]["sim_modes"]
# 	xlim = [all_sim_results[0]["param"]["env_xlim"][0],all_sim_results[0]["param"]["env_xlim"][1]]
# 	ylim = [all_sim_results[0]["param"]["env_ylim"][0],all_sim_results[0]["param"]["env_ylim"][1]]

# 	results = dict() 
# 	for exp4_sim_mode in exp4_sim_modes: 
# 		results[exp4_sim_mode] = np.zeros((X.shape[0],Y.shape[0],num_trials))

# 	for sim_result in all_sim_results:
# 		pos,i_x,i_y = get_initial_condition(sim_result["param"],X,Y)
# 		exp4_sim_mode = sim_result["param"]["exp4_sim_mode"]
# 		results[exp4_sim_mode][i_x,i_y,sim_result["param"]["i_trial"]] = sim_result["rewards"][-1,0]

# 	# values = np.zeros((X.shape[0],Y.shape[0],num_trials)) 
# 	# for sim_result in all_sim_results:
# 	# 	pos,i_x,i_y = get_initial_condition(sim_result["param"],X,Y)
# 	# 	values[i_x,i_y,sim_result["param"]["i_trial"]] = sim_result["rewards"][-1,0]

# 	# print(values.shape)
# 	# print(np.mean(values,axis=2).shape)
# 	# exit()


# 	colors = get_n_colors(len(sim_result["param"]["robots"]))

# 	for sim_mode, values in results.items():
# 		fig,ax = plt.subplots()
# 		im = ax.imshow(np.mean(values,axis=2).T,origin='lower',extent=(xlim[0],xlim[1],ylim[0],ylim[1]),vmin=0,vmax=1)
# 		fig.colorbar(im)
# 		ax.scatter(sim_result["param"]["goal"][0],sim_result["param"]["goal"][1],color='green',marker='o',label='goal')
# 		for robot_idx in range(sim_result["param"]["num_nodes"]):
# 			if not robot_idx == 0:
# 				ax.scatter(sim_result["states"][0,robot_idx,0],sim_result["states"][0,robot_idx,1],marker='o',color=colors[robot_idx],label=str(robot_idx))
# 				ax.arrow(sim_result["states"][0,robot_idx,0], sim_result["states"][0,robot_idx,1], \
# 					sim_result["states"][0,robot_idx,2], sim_result["states"][0,robot_idx,3], color=colors[robot_idx])
# 		ax.set_xlim([xlim[0],xlim[1]])
# 		ax.set_ylim([ylim[0],ylim[1]])

# 		dx = xlim[1] - X[-1]
# 		ax.set_xticks(X - dx)
# 		ax.set_yticks(Y - dx)
# 		ax.grid(True)
# 		ax.set_aspect('equal')
# 		ax.legend(loc='upper left')
# 		ax.set_title(sim_mode)


def plot_exp2_results(all_sim_results):

	training_teams = all_sim_results[0]["param"]["training_teams"]
	modes = all_sim_results[0]["param"]["modes"]
	tree_sizes = all_sim_results[0]["param"]["mcts_tree_sizes"]
	num_trials = all_sim_results[0]["param"]["sim_num_trials"]
	team_comps = all_sim_results[0]["param"]["robot_team_compositions"]
	mcts_beta2s = all_sim_results[0]["param"]["mcts_beta2s"]

	# put into easy-to-use dict! 
	results = dict()

	for sim_result in all_sim_results:

		team_comp = sim_result["param"]["robot_team_composition"]
		training_team = sim_result["param"]["training_team"]
		trial = sim_result["param"]["sim_trial"]
		tree_size = sim_result["param"]["policy_dict"]["mcts_tree_size"]
		mode = sim_result["param"]["policy_dict"]["sim_mode"]
		beta = sim_result["param"]["policy_dict"]["mcts_beta2"]

		num_nodes_A, num_nodes_B = 0,0
		for robot_type, robot_number in team_comp["a"].items():
			num_nodes_A += robot_number 
		for robot_Type, robot_number in team_comp["b"].items():
			num_nodes_B += robot_number 

		key = (num_nodes_A,num_nodes_B,tree_size,training_team,mode,beta,trial)
		results[key] = sim_result

	num_ims_per_ic = 0
	for mode in modes: 
		if mode == "GLAS":
			betas = [0]
		elif mode == "MCTS":
			betas = mcts_beta2s
		for beta in betas:  
			num_ims_per_ic += 1

	# make figs! 
	for team_comp in team_comps: 

		num_nodes_A, num_nodes_B = 0,0
		for robot_type, robot_number in team_comp["a"].items():
			num_nodes_A += robot_number 
		for robot_Type, robot_number in team_comp["b"].items():
			num_nodes_B += robot_number 

		for i_trial in range(num_trials):

			# plot initial condition
			fig,ax = plt.subplots()
			key = (num_nodes_A,num_nodes_B,tree_sizes[0],training_team,modes[0],mcts_beta2s[0],i_trial)
			fig.suptitle('Trial {}'.format(results[key]["param"]["curr_ic"]))
			colors = get_colors(results[key]["param"])
			ax.scatter(results[key]["param"]["goal"][0],results[key]["param"]["goal"][1],color='green',marker='o',label='goal')
			for robot_idx in range(results[key]["param"]["num_nodes"]):
				ax.scatter(results[key]["states"][0,robot_idx,0],results[key]["states"][0,robot_idx,1],marker='o',color=colors[robot_idx],label=str(robot_idx))
				ax.arrow(results[key]["states"][0,robot_idx,0], results[key]["states"][0,robot_idx,1], \
					results[key]["states"][0,robot_idx,2], results[key]["states"][0,robot_idx,3], color=colors[robot_idx])
			ax.set_xlim([results[key]["param"]["env_xlim"][0],results[key]["param"]["env_xlim"][1]])
			ax.set_ylim([results[key]["param"]["env_ylim"][0],results[key]["param"]["env_ylim"][1]])
			ax.grid(True)
			ax.set_aspect('equal')
			ax.legend(loc='upper left')

			for training_team in training_teams: 
				
				if training_team == "a":
					robot_idxs = np.arange(num_nodes_A)
				elif training_team == "b":
					robot_idxs = np.arange(num_nodes_B) + num_nodes_A 

				for robot_idx in robot_idxs: 
	
					# plot policy distribution
					fig, axs = plt.subplots(nrows=1,ncols=num_ims_per_ic)
					fig.suptitle('Trial {} Robot {}'.format(results[key]["param"]["curr_ic"],robot_idx))
					count_ims_per_ic = 0

					for i_mode, mode in enumerate(modes): 

						if mode == "GLAS":
							betas = [0]
						elif mode == "MCTS":
							betas = mcts_beta2s
												
						for beta in betas: 

							if mode == "MCTS":
								title = "MCTS \n b = {}".format(beta)
							elif mode == "GLAS":
								title = "GLAS"

							im = np.nan*np.ones((len(tree_sizes),9))

							if num_ims_per_ic > 1:
								ax = axs[count_ims_per_ic]
								count_ims_per_ic += 1
							else:
								ax = axs 

							for i_tree, tree_size in enumerate(tree_sizes): 

								key = (num_nodes_A,num_nodes_B,tree_size,training_team,mode,beta,i_trial)

								im[i_tree,:] = results[key]["actions"][0,robot_idx,:] 
								imobj = ax.imshow(im.T,vmin=0,vmax=1.0,cmap=cm.coolwarm)

								ax.set_xticks([])
								ax.set_yticks([])

							if i_mode == 0:
								ax.set_ylabel('Action Distribution')
								ax.set_yticks(np.arange(len(results[key]["param"]["actions"]))) # throws future warning 
								ax.set_yticklabels(results[key]["param"]["actions"])

							if i_mode == len(modes)-1 :
								fig.subplots_adjust(right=0.8)
								cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
								fig.colorbar(imobj, cax=cbar_ax) 

							ax.set_title(title)
							ax.set_xticks(np.arange(len(tree_sizes)))
							ax.set_xticklabels(np.array(tree_sizes,dtype=int)/1000,rotation=45)

def policy_to_label(policy):
	# keys = ["mcts_tree_size"]
	# keys = ["sim_mode","path","mcts_tree_size"]
	# keys = ["sim_mode","path","mcts_beta2"] 
	# keys = ["sim_mode","mcts_beta2"] 
	keys = ["sim_mode","mcts_beta2","mcts_tree_size","path"]
	# keys = ["path"]
	label = '' 
	for key, value in policy.items():
		if "path" in key and np.any(['path' in a for a in keys]):
			if value is None: 
				label += 'None'
			else:
				label += '{} '.format(os.path.basename(value).split(".")[0])
		elif key == "mcts_beta2" and key in keys:
			if policy["sim_mode"] == "MCTS":
				label += ', b: {}'.format(value)
		elif key == "mcts_tree_size" and key in keys:
			label += ', |n|: {}'.format(value)
		elif key == "mcts_c_param" and key in keys:
			label += ', c: {}'.format(value)			
		elif key == "sim_mode" and key in keys:
			label += '{} '.format(value)
		elif key in keys:  
			label += ', {}'.format(value)
	
	return label

def plot_exp5_results(all_sim_results):

	# read results into dict 
	rw_results = defaultdict(list) # game reward
	rg_results = defaultdict(list) # reached goal reward 
	model_names_a = set()
	model_names_b = set()
	for sim_result in all_sim_results: 
		# key = (test_team, tree size, model)
		test_team = sim_result["param"]["test_team"]
		if test_team == "a":
			tree_size = sim_result["param"]["policy_dict_a"]["mcts_tree_size"]
			model_name = sim_result["param"]["policy_dict_a"]["path_glas_model_a"]
			model_names_a.add(model_name)
		elif test_team == "b":
			tree_size = sim_result["param"]["policy_dict_b"]["mcts_tree_size"]
			model_name = sim_result["param"]["policy_dict_b"]["path_glas_model_b"]
			model_names_b.add(model_name)

		key = (test_team, tree_size, model_name)
		rw_results[key].append(sim_result["rewards"][-1,0])
		rg_results[key].append(sim_result["reached_goal"])

	# 
	tree_sizes = all_sim_results[0]["param"]["tree_sizes"]
	colors = [get_n_colors(len(model_names_a),cmap=cm.Set1),get_n_colors(len(model_names_b),cmap=cm.Set1)]

	# plots: 
	# 	x-axis: tree size 
	# 	y-axis: value for each bias
	fig,axs = plt.subplots(ncols=2,sharey=True,squeeze=False)
	for i_ax, (test_team, model_names, title) in enumerate(zip(["a","b"],[model_names_a,model_names_b],["Attacking","Defending"])):
		ax = axs[0,i_ax]

		for i_model, model_name in enumerate(model_names):

			mean_data = []
			std_data = []
			for tree_size in tree_sizes:
				key = (test_team, tree_size, model_name)
				mean_data.append(np.mean(np.array(rg_results[key])))
				# std_data.append(np.std(np.array(rg_results[key])))

			label = 'None' if model_name is None else os.path.basename(model_name)
			ax.plot(tree_sizes,mean_data,color=colors[i_ax][i_model],label=label)
			# ax.errorbar(tree_sizes,mean_data,yerr=std_data,color=colors[i_ax][i_model],alpha=0.5,linewidth=1)

		ax.set_xlabel('Tree Size')
		ax.set_xscale('log')
		if i_ax == 0:
			ax.set_ylabel('Reached Goal Reward')
			ax.legend()

		ax.set_title(title)
		ax.set_ylim([0,1])
		ax.grid(True)

		ax.set_xticks(tree_sizes)
		ax.set_xticklabels(tree_sizes,rotation=45)
		# ax.set_xticklabels(rotation=45)

	fig.suptitle('Reached Goal Reward vs Tree Size')
	# fig.tight_layout()

def plot_exp3_results(all_sim_results):

	rw_results = defaultdict(list) # game reward
	rg_results = defaultdict(list) # reached goal reward 
	for sim_result in all_sim_results:
		key = (\
			policy_to_label(sim_result["param"]["policy_dict_a"]),
			policy_to_label(sim_result["param"]["policy_dict_b"]))
		# results[key].append(sim_result["rewards"][-1,0])
		# results[key].append(sim_result["reached_goal"])
		rw_results[key].append(sim_result["rewards"][-1,0])
		rg_results[key].append(sim_result["reached_goal"])

	attackerPolicies = all_sim_results[0]["param"]["attackerPolicyDicts"]
	defenderPolicies = all_sim_results[0]["param"]["defenderPolicyDicts"]

	mean_rw_result = np.zeros((len(attackerPolicies),len(defenderPolicies)))
	std_rw_result = np.zeros((len(attackerPolicies),len(defenderPolicies)))
	mean_rg_result = np.zeros((len(attackerPolicies),len(defenderPolicies)))
	std_rg_result = np.zeros((len(attackerPolicies),len(defenderPolicies)))
	for a_idx, policy_dict_a in enumerate(attackerPolicies):
		for b_idx, policy_dict_b in enumerate(defenderPolicies):
			# key = (sim_mode_a, path_glas_model_a,sim_mode_b,path_glas_model_b)
			key = (\
				policy_to_label(policy_dict_a),
				policy_to_label(policy_dict_b))			
			mean_rw_result[a_idx,b_idx] = np.mean(rw_results[key])
			std_rw_result[a_idx,b_idx] = np.std(rw_results[key])

			mean_rg_result[a_idx,b_idx] = np.mean(rg_results[key])
			std_rg_result[a_idx,b_idx] = np.std(rg_results[key])

	xticklabels = []
	for policy_dict_a in attackerPolicies:
		xticklabels.append(policy_to_label(policy_dict_a))

	yticklabels = []
	for policy_dict_b in defenderPolicies:
		yticklabels.append(policy_to_label(policy_dict_b))

	for mean_result, title in zip([mean_rw_result, mean_rg_result],["Game Reward","Reached Goal"]): 

		fig,ax = plt.subplots()
		# im = ax.imshow(mean_result,origin='lower',vmin=0,vmax=1,cmap=cm.seaborn)
		ax = sns.heatmap(mean_result.T,vmin=0,vmax=1,annot=True)
		# fig.colorbar(im)
		# ax.set_xticks(range(len(attackerPolicies)))
		# ax.set_yticks(range(len(defenderPolicies)))
		ax.set_xticklabels(xticklabels,rotation=45)
		ax.set_yticklabels(yticklabels,rotation=45)
		ax.tick_params(axis='both',labelsize=5)
		ax.set_xlabel('attackers')
		ax.set_ylabel('defenders')
		fig.suptitle(title)
		fig.tight_layout()
		# ax.legend()
		# ax.grid(True)


def plot_test_model(df_param,stats):

	LIMS = df_param.robot_types["standard_robot"]["acceleration_limit"]*np.array([[-1,1],[-1,1]])
	nbins = 20

	for alpha,stats_per_condition in stats.items():

		fig,axs = plt.subplots(ncols=2)

		xedges = np.linspace(LIMS[0,0],LIMS[0,1],nbins) 
		yedges = np.linspace(LIMS[1,0],LIMS[1,1],nbins) 

		h_mcts, xedges, yedges = np.histogram2d(stats_per_condition["test"][:,0],stats_per_condition["test"][:,1],bins=(xedges,yedges),range=LIMS,density=True)
		h_model, xedges, yedges = np.histogram2d(stats_per_condition["learned"][:,0],stats_per_condition["learned"][:,1],bins=(xedges,yedges),range=LIMS,density=True)

		h_mcts = h_mcts.T 
		h_model = h_model.T 

		axs[0].imshow(h_mcts,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]])
		axs[1].imshow(h_model,origin='lower',interpolation='nearest',extent=[LIMS[0,0],LIMS[0,1],LIMS[1,0],LIMS[1,1]])

		# - arrange 
		axs[0].set_title('mcts: {}'.format(alpha))
		axs[1].set_title('model: {}'.format(alpha))
		axs[0].set_xlabel('x-action')
		axs[1].set_xlabel('x-action')
		axs[0].set_ylabel('y-action')

		# fig,ax = plt.subplots()
		# ax.hist2d(stats_per_condition["learned"][:,0],stats_per_condition["learned"][:,1])
		# ax.set_title('learned')
		# ax.set_aspect('equal')

		# fig,ax = plt.subplots()
		# ax.hist2d(stats_per_condition["test"][:,0],stats_per_condition["test"][:,1])
		# ax.set_title('test')
		# ax.set_aspect('equal')
		

def plot_convergence(all_sim_results):

	def extract_reward(sim_result,longest_rollout):
		nt = np.shape(sim_result["rewards"])[0]
		reward = (np.sum(sim_result["rewards"][:,0]) + sim_result["rewards"][-1,0]*(longest_rollout-nt))/longest_rollout
		return reward 

	# group by tree size, rollout policy and case number 
	results = defaultdict(list)
	longest_rollout = defaultdict(int)
	tree_sizes = set()
	cases = set()
	for sim_result in all_sim_results:

		tree_size = sim_result["param"]["tree_size"]
		policy = sim_result["param"]["glas_rollout_on"]
		case = sim_result["param"]["case"]

		tree_sizes.add(tree_size)
		cases.add(case)

		key = (tree_size,policy,case)
		results[key].append(sim_result)
		
		if longest_rollout[case] < sim_result["times"].shape[0]:
			longest_rollout[case] = sim_result["times"].shape[0]

	tree_sizes = np.array(list(tree_sizes))
	tree_sizes = np.sort(tree_sizes)
	cases = np.array(list(cases))
	cases = np.sort(cases)	

	for case in cases: 

		to_plots = defaultdict(list)
		for key,sim_results in results.items():

			if key[2] == case: 

				reward_stats = [] 
				for sim_result in sim_results: 
					reward_stats.append(extract_reward(sim_result,longest_rollout[case]))

				tree_size = key[0]
				policy = key[1]
				case = key[2]
				reward_stats = np.array(reward_stats)
				reward_mean = np.mean(reward_stats)
				reward_std = np.std(reward_stats)

				to_plots[policy,case].append((tree_size,reward_mean,reward_std))

				print('key',key)
				print('reward_stats',reward_stats)
				print('reward_mean',reward_mean)
				print('reward_std',reward_std)
				print('')

		fig,ax = plt.subplots()
		ax.set_title('Case {}'.format(str(case)))
		ax.set_xscale('log')
		ax.grid(True)
		ax.set_ylim([0,1])
		ax.set_ylabel('Reward')
		ax.set_xlabel('Tree Size (K)')

		for (glas_on,case),to_plot in to_plots.items():

			if glas_on: 
				label = 'GLAS'
				color = 'blue'
			else:
				label = 'Random'
				color = 'orange'

			to_plot = np.array(to_plot)
			to_plot = to_plot[to_plot[:,0].argsort()]

			ax.plot(to_plot[:,0],to_plot[:,1],marker='o',label=label,color=color)
			ax.fill_between(to_plot[:,0], to_plot[:,1]-to_plot[:,2], to_plot[:,1]+to_plot[:,2],color=color,alpha=0.5)
			ax.set_xticks(to_plot[:,0])
			ax.set_xticklabels(to_plot[:,0]/1000)

		ax.legend()


def rotate_image(image, angle):
	''' 
	Function for rotating images for plotting
	'''
	# grab the dimensions of the image and then determine the
	# center
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))


def plot_image(x, y, heading, im, ax=None):
	'''
	Adds an image to the current plot at the given points
	'''

	ax = ax or plt.gca()
	rotation_in_degrees = heading*57.7

	# Rotate image
	im = rotate_image(im, rotation_in_degrees)
	
	# Scale and offset image
	im = OffsetImage(im, zoom=15/ax.figure.dpi)

	im.image.axes = ax

	ab = AnnotationBbox(im, (x,y), frameon=False, pad=0.0)
	ax.add_artist(ab)
	#a = 1
	#b = 2
	# (maybe)
	# https://stackoverflow.com/questions/59401957/rotating-images-on-a-matplotlib-plot


def calc_heading(vx,vy):
	'''
	Calculate the heading of the vehicle given the (x,y) co-ordinates
	'''

	# Initialise heading vector
	heading = np.zeros((vx.size,1))

	# Loop through and find the headings
	for ii in range(1,vx.size):
		heading[ii] = math.atan2(vx[ii-1],vy[ii-1])
	
	# Make the initial headings look correct
	heading[1] = heading[2]
	heading[0] = heading[1]

	# Correct headings when a vehicle dies

	
	return heading


def plot_animation(sim_result,args):
	'''
	Function for plotting animations for swarm AI
	'''

	## Setup
	# Extract data from pickle
	times = sim_result["times"]
	states = sim_result["states"]
	actions = sim_result["actions"]
	rewards = sim_result["rewards"]

	num_nodes = sim_result["param"]["num_nodes"]		# Number of agents
	team_1_idxs = sim_result["param"]["team_1_idxs"]	# Identites of attackers
	tag_radius = sim_result["param"]["robots"][0]["tag_radius"] # Tag area of defenders

	goal = sim_result["param"]["goal"] 			# Goal area
	env_xlim = sim_result["param"]["env_xlim"]	# Environment Size (X)
	env_ylim = sim_result["param"]["env_ylim"]	# Environment Size (Y)

	team_1_color = 'blue'
	team_2_color = 'orange'
	goal_color = 'green'

	im_team1 = plt.imread('./resources/teamA.png')
	im_team2 = plt.imread('./resources/teamB.png')

	colors = get_colors(sim_result["param"])

	# Calculate the headings of each agent
	headings = []
	for jj in range(num_nodes):
		agent_vx = states[:,jj,2]
		agent_vy = states[:,jj,3]

		if jj == 0:
			headings = calc_heading(agent_vx,agent_vy)
		else:
			headings = np.concatenate((headings,calc_heading(agent_vx,agent_vy)),axis=1)

	# Inputs

	# Text box setup
	textBox0 = dict(boxstyle='round', facecolor='darkgray', alpha=0.5)

	# Work out file names and directories
	input_dir,  input_file  = os.path.split(args.file)
	input_file, input_ext   = os.path.splitext(os.path.basename(args.file))

	output_dir,  output_file = os.path.split(args.outputMP4)
	output_file, output_ext  = os.path.splitext(os.path.basename(args.outputMP4))

	if output_dir:
		png_directory = output_dir+"/"+output_file+"_png/"
	else:
		png_directory = output_file+"_png/"

	if not os.path.isdir(png_directory):
		os.makedirs(png_directory)

	# Output directory for the images
	print("Generating images...")

	# Plot each frame
	for ii in range(0, times.size):
		## Plot setup
		fig, ax = plt.subplots()
		fig.set_size_inches(12.80,7.20) # Output movie size will be this * 100

		# Grid
		ax.grid(True)
		ax.set_axisbelow(True)
		ax.set_title("t = "+"{:03.2f}".format(times[ii])+" [ s ]")

		# Fix axes to game arena
		ax.set_xlim([env_xlim[0],env_xlim[1]])
		ax.set_ylim([env_ylim[0],env_ylim[1]])
		ax.set_aspect('equal','box')

		# Axis lables
		ax.set_xlabel(args.file)

		## Plot Elements
		# Plot goal
		ax.add_patch(mpatches.Circle(goal, tag_radius, color=goal_color,alpha=0.5))

		# Plot players
		for jj in range(num_nodes):
			# Tag circle (defenders only)
			if not (np.any(team_1_idxs) == jj):
				ax.add_patch(mpatches.Circle(states[ii,jj,0:2], sim_result["param"]["robots"][jj]["tag_radius"], \
					color=colors[jj],alpha=0.3,fill=False,linestyle='--'))   
			# Path                        						
			ax.plot(   states[0:ii+1,jj,0], states[0:ii+1,jj,1], linewidth=3, color=colors[jj], zorder=1)
			# Previous Positions Marker    
			ax.scatter(states[0:ii+1,jj,0], states[0:ii+1,jj,1], marker='o' , color=colors[jj], zorder=1)
			# Current Position
			if (np.isin(team_1_idxs,[jj]).any()):
				plot_image(states[ii,jj,0], states[ii,jj,1], headings[ii,jj], im_team1)
			else :
				plot_image(states[ii,jj,0], states[ii,jj,1], headings[ii,jj], im_team2)

		# Rewards of Each Team
		ax.text(0.10, 0.95, "Reward\n"+"{:.2f}".format(rewards[ii,0]), transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=textBox0,zorder=50)
		ax.text(0.90, 0.95, "Reward\n"+"{:.2f}".format(rewards[ii,1]), transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=textBox0,zorder=50)

		# Debug Text
		#ax.text(0.10, 0.05, "Heading\n"+"{:.2f}".format(headings[ii,0]*57.7), transform=ax.transAxes, fontsize=6, verticalalignment='top', horizontalalignment='center', bbox=textBox0,zorder=50)
		#ax.text(0.90, 0.05, "Heading\n"+"{:.2f}".format(headings[ii,1]*57.7), transform=ax.transAxes, fontsize=6, verticalalignment='top', horizontalalignment='center', bbox=textBox0,zorder=50)

		# Save png image
		fig.savefig(png_directory+"{:03.0f}".format(ii)+".png", dpi=100)
		
		# Save a couple of extra frames if this is the last image
		if ii == times.size-1:
			for jj in range(ii,ii+10):
				fig.savefig(png_directory+"{:03.0f}".format(jj)+".png", dpi=100)

		# Close figure
		plt.close()

	# Combine images to form the movie
	print("Creating MP4")
	cmd = "ffmpeg -y -r 15 -i "+png_directory+"%03d.png -c:v libx264 -vf \"fps=25,format=yuv420p\" "+output_dir+"/"+output_file+".mp4"
	os.system(cmd)


def sanitise_filenames(filename):
	# Puts paths and things in where required to stop things writing to /

	# If the filename string is empty, then we didn't request this file
	if not (filename):
		return filename

	# Split the file name up
	file_dir,  file_name = os.path.split(filename)

	# Fill in extra elements
	if not (file_dir):
		file_dir = os.getcwd()

	# Make new filename
	filename = os.path.join(file_dir, file_name)

	return filename

def get_team_from_raw_fn(raw_fn):
	raw_fn = os.path.basename(raw_fn)
	raw_fn = raw_fn.split("train")
	team = raw_fn[0][-1]
	return team 	

def get_team_from_batch_fn(batch_fn):
	batch_fn = os.path.basename(batch_fn)
	batch_fn = batch_fn.split("train")
	team = batch_fn[0][-1]
	return team 


if __name__ == '__main__':

	import argparse
	import datahandler

	parser = argparse.ArgumentParser()
	parser.add_argument("-plot_type", default=None, required=False)
	parser.add_argument("-file", default=None, required=False)
	args = parser.parse_args() 

	if args.plot_type == "plot_sa_pairs" and not args.file is None:
		num_points_per_file = 9 
		sim_result = dh.load_sim_result(args.file)
		state_action_pairs = list(zip(sim_result["states"],sim_result["actions"]))
		sampled_sa_pairs = random.sample(state_action_pairs,num_points_per_file)
		training_team = get_team_from_raw_fn(args.file)
		plot_sa_pairs(sampled_sa_pairs,sim_result,training_team)

	if args.plot_type == "plot_oa_pairs" and not args.file is None: 
		num_points_per_file = 9 

		param = Param()
		rsense = param.robot_types["standard_robot"]["r_sense"]
		env_length = param.env_xlim[1] - param.env_xlim[0]
		abs_goal = param.goal 
		action_list = param.actions
		training_team = get_team_from_batch_fn(args.file)

		o_a,o_b,goal,actions = dh.read_oa_batch(args.file)
		oa_pairs = list(zip(o_a,o_b,goal,actions))
		sampled_oa_pairs = random.sample(oa_pairs,num_points_per_file)
		plot_oa_pairs(sampled_oa_pairs,abs_goal,training_team,rsense,action_list,env_length)

	if args.plot_type == "plot_sim_result" and not args.file is None: 	

		sim_result = dh.load_sim_result(args.file)
		plot_tree_results(sim_result)

	save_figs('temp_plot.pdf')
	open_figs('temp_plot.pdf')

	# parser = argparse.ArgumentParser()
	# parser.add_argument("file", help="pickle file to visualize")
	# parser.add_argument("--outputPDF", help="output pdf file")
	# parser.add_argument("--outputMP4", help="output video file")

	# # parser.add_argument("--animate", action='store_true', help="animate using meshlab")
	# args = parser.parse_args()

	# # Detect if input file is a directory or a pickle file
	# input_file, input_ext = os.path.splitext(os.path.basename(args.file))
	# if ("pickle" in input_ext):
	# 	print("Generating for a file")

	# 	# Assign argument as per normal
	# 	files = [args.file]
	# 	PDFs = [args.outputPDF]
	# 	MP4s = [args.outputMP4]

	# else:
	# 	# Search directory for matching files
	# 	print("Generating for a directory")
	# 	files = glob.glob(args.file+'**/*.pickle', recursive = True)

	# 	PDFs = []
	# 	MP4s = []

	# 	# Generate save names
	# 	for ii in range(0,len(files)):
	# 		# PDF files
	# 		output_dir,  output_file = os.path.split(args.outputPDF)
	# 		output_file, output_ext  = os.path.splitext(os.path.basename(args.outputPDF))

	# 		PDFs.append(os.path.join(output_dir, "{:03.0f}".format(ii+1)+'-'+output_file+'.pdf'))

	# 		# MP4 files
	# 		output_dir,  output_file = os.path.split(args.outputMP4)
	# 		output_file, output_ext  = os.path.splitext(os.path.basename(args.outputMP4))

	# 		MP4s.append(os.path.join(output_dir, "{:03.0f}".format(ii+1)+'-'+output_file+'.mp4'))

	# # Loop through each of the files in files
	# for ii in range(0,len(files)):
	# 	print("{:3.0f}".format(ii+1),"/"+"{:3.0f}".format(len(files))+" - Generating plots for "+files[ii])
		
	# 	args.file      = sanitise_filenames(files[ii])
	# 	args.outputPDF = sanitise_filenames(PDFs[ii])
	# 	args.outputMP4 = sanitise_filenames(MP4s[ii])

	# 	# Load results
	# 	sim_result = datahandler.load_sim_result(args.file)

	# 	if args.outputPDF:
	# 		plot_tree_results(sim_result)

	# 		save_figs(args.outputPDF)
	# 		# Only open PDF if we're looking at one file
	# 		if len(files) == 1:
	# 			open_figs(args.outputPDF)

	# 	if args.outputMP4:
	# 		plot_animation(sim_result,args)
	
	# # Join the movies together if running in batch mode
	# #	This piece of code will work but needs to be run from the correct directory...
	# #	Haven't worked this out yet...
	# '''
	# if (len(files) > 1):
	# 	print("Combining MP4s")
	# 	# Get a list of the movies generated
	# 	cmd = "for f in *.mp4 ; do echo file \'$f\' >> list.txt;"
	# 	os.system(cmd)
	# 	# Combine them ussing ffmpeg
	# 	cmd = "ffmpeg -f concat -safe 0 -i list.txt -c copy swarm-AI.mp4"
	# 	os.system(cmd)
	# '''
	# #  for f in *.mp4 ; do echo file \'$f\' >> list.txt; done && ffmpeg -f concat -safe 0 -i list.txt -c copy swarm-AI.mp4

	
	# print("\n\nDone!\n")