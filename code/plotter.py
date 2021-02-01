

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

def plot_panagou(states,param):
	num_robots = len(param.team_1_idxs) + len(param.team_2_idxs)
	colors = get_2_colors(num_robots,len(param.team_1_idxs))

	goal_color = 'green'
	df_alpha = 0.2

	fig,ax = plt.subplots()

	# Plot the path of each robot
	for i_robot in range(num_robots):
		ax.plot(states[: ,i_robot,0],states[: ,i_robot,1],color=colors[i_robot],linewidth=1,marker='o',markersize=1)
		ax.plot(states[-1,i_robot,0],states[-1,i_robot,1],color=colors[i_robot],linewidth=1,marker='o',markersize=3)

		# Add the tag radius of attackers
		if i_robot in param.team_2_idxs :
			ax.add_patch(mpatches.Circle(states[-1,i_robot,0:2], param.robots[i_robot]["tag_radius"], \
				color=colors[i_robot],alpha=0.2,fill=False))

		# Add a robot number to the starting point of each robot
		if i_robot in param.team_2_idxs :
			textstr = "DEF\n%d" % i_robot
		else :
			textstr = "ATT\n%d" % i_robot

		ax.text(states[0 ,i_robot,0], states[0 ,i_robot,1], textstr, fontsize=6, verticalalignment='center', horizontalalignment='center')

	# Plot the goal
	ax.plot(param.goal[0],param.goal[1],color=goal_color,marker='*')

	# Set plot range
	ax.set_xlim([param.env_xlim[0],param.env_xlim[1]])
	ax.set_ylim([param.env_ylim[0],param.env_ylim[1]])
	ax.grid(True)
	ax.axis('equal')


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

def plot_exp8_results(all_sim_results):
	results = defaultdict(list)
	tree_sizes = set()
	model_names = set() 
	for sim_result in all_sim_results: 
		tree_size = sim_result["param"]["policy_dict"]["mcts_tree_size"]
		model_name = sim_result["param"]["policy_dict"]["path_glas_model_a"]
		key = (tree_size,model_name)
		results[key].append(sim_result["value"])
		tree_sizes.add(tree_size)
		model_names.add(model_name)

		print('key, value: {},{}'.format(key,sim_result["value"]))

	tree_sizes = sorted(list(tree_sizes))
	colors = ["blue","orange"]

	print('tree_sizes',tree_sizes)
	print('model_names',model_names)

	fig,ax = plt.subplots()
	for i_model, model_name in enumerate(model_names): 

		mean_data = []
		std_data = [] 
		for tree_size in tree_sizes: 
			mean_data.append(np.mean(results[(tree_size,model_name)]))
			std_data.append(np.std(results[(tree_size,model_name)]))

		mean_data = np.array(mean_data)
		std_data = np.array(std_data)

		label = model_name 
		if label is None: 
			label = "None"

		ax.plot(tree_sizes,mean_data,color=colors[i_model],label=label)
		ax.fill_between(tree_sizes, mean_data-std_data, mean_data+std_data,color=colors[i_model],alpha=0.5)

	ax.legend(loc='best')
	ax.set_xscale('log')

	return fig,ax 

def plot_value_dataset_distributions(loader):
	# loader = [(v_a,v_b,n_a,n_b,n_rg,target_value)]
	# v_a = {s^j - g} 
	# v_b = {s^j - g}
	# 	- e.g. 3d dubins : s = (x,y,z,phi,psi,v)

	state_dim_dubins = 6

	state_dim = state_dim_dubins

	print("formatting data...")
	for i, (v_a,v_b,n_a,n_b,n_rg,target_value) in enumerate(loader):

		print("{}/{}".format(i,len(loader)))

		v_a = v_a.cpu().detach().numpy()  
		v_b = v_b.cpu().detach().numpy()  
		n_a = n_a.cpu().detach().numpy()  
		n_b = n_b.cpu().detach().numpy()  
		n_rg = n_rg.cpu().detach().numpy()  
		target_value = target_value.cpu().detach().numpy()  

		# data_i = np.zeros((v_a.shape[0],2*state_dim+4))
		# data_i[:,0:state_dim] = v_a 
		# data_i[:,state_dim:2*state_dim] = v_b
		# data_i[:,2*state_dim] = n_a
		# data_i[:,2*state_dim+1] = n_b
		# data_i[:,2*state_dim+2] = n_rg
		# data_i[:,2*state_dim+3] = target_value

		data_i = [[] for _ in range(2*state_dim + 4)] 
		for j in range(v_a.shape[1]):
			data_i[np.mod(j,state_dim)].extend(v_a[:,j])
		for j in range(v_b.shape[1]):
			data_i[state_dim + np.mod(j,state_dim)].extend(v_b[:,j])

		data_i[2*state_dim].extend(n_a)
		data_i[2*state_dim+1].extend(n_b)
		data_i[2*state_dim+2].extend(n_rg)
		data_i[2*state_dim+3].extend(target_value)

		if i == 0:
			data = data_i 
		else: 
			# data = np.vstack((data,data_i))
			for j,_ in enumerate(data): 
				data[j].extend(data_i[j])

		# break

	labels = ["xa","ya","za","phia","psia","va","xb","yb","zb","phib","psib","vb","numa","numb","numrg","target_value"]

	print('plotting histogram...')
	nrows = 4 
	ncols = 4
	fig,axs = plt.subplots(nrows=nrows,ncols=ncols)
	for idx in range(len(data)):
		print("{}/{}".format(idx,len(data)))
		i_row = int(np.floor(idx/ncols))
		i_col = np.mod(idx,ncols)

		axs[i_row,i_col].hist(np.array(data[idx][:])) 
		axs[i_row,i_col].set_title(labels[idx])

	fig.tight_layout()


def plot_policy_dataset_distributions(loader):
	# loader = [(o_a,o_b,goal,action,weight)]
	# o_a = {s^j - s^i} 
	# o_b = {s^j - s^i}
	# goal = {g - s^i}
	# 	- e.g. 3d dubins : s = (x,y,z,phi,psi,v), a: (phidot, psidot, vdot)


	state_dim_dubins = 6
	action_dim_dubins = 3 

	state_dim = state_dim_dubins
	action_dim = action_dim_dubins

	print("formatting data...")
	for i, (o_a,o_b,goal,action,weight) in enumerate(loader):

		print("{}/{}".format(i,len(loader)))

		o_a = o_a.cpu().detach().numpy()  
		o_b = o_b.cpu().detach().numpy()  
		goal = goal.cpu().detach().numpy()  
		action = action.cpu().detach().numpy()  
		weight = weight.cpu().detach().numpy()  

		data_i = [[] for _ in range(3*state_dim + 2*action_dim)] 
		for j in range(o_a.shape[1]):
			data_i[np.mod(j,state_dim)].extend(o_a[:,j])
		for j in range(o_b.shape[1]):
			data_i[state_dim + np.mod(j,state_dim)].extend(o_b[:,j])
		for j in range(goal.shape[1]):
			data_i[2*state_dim + np.mod(j,state_dim)].extend(goal[:,j])

		data_i[3*state_dim:3*state_dim + action_dim].extend(action)
		data_i[3*state_dim + action_dim:3*state_dim + 2*action_dim].extend(weight)

		if i == 0:
			data = data_i 
		else: 
			# data = np.vstack((data,data_i))
			for j,_ in enumerate(data): 
				data[j].extend(data_i[j])

		# break

	labels = ["xa","ya","za","phia","psia","va",\
		"xb","yb","zb","phib","psib","vb",\
		"xg","yg","zg","phig","psig","vg",\
		"phidot", "psidot", "vdot", \
		"phidotw", "psidotw", "vdotw",]

	print('plotting histogram...')
	nrows = 4 
	ncols = 6
	fig,axs = plt.subplots(nrows=nrows,ncols=ncols)
	for idx in range(len(data)):
		print("{}/{}".format(idx,len(data)))
		i_row = int(np.floor(idx/ncols))
		i_col = np.mod(idx,ncols)

		axs[i_row,i_col].hist(np.array(data[idx][:])) 
		axs[i_row,i_col].set_title(labels[idx])

	for i_row in range(nrows):
		for i_col in range(ncols):
			ax = axs[i_row,i_col]
			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(4)

	fig.tight_layout()	



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

def plot_3d_dubins_result(sim_result,title):

	states = sim_result["states"]
	actions = sim_result["actions"]

	nt, nrobots, state_dim = states.shape 

	x_lim = sim_result["param"]["env_xlim"]
	y_lim = sim_result["param"]["env_ylim"]
	z_lim = sim_result["param"]["env_ylim"] # assume zlim and ylim are same 

	colors = get_colors(sim_result["param"])

	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlim(x_lim)
	ax.set_ylim(y_lim)
	ax.set_zlim(z_lim)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	
	# goal region 
	goal_z = (y_lim[1]-y_lim[0])/2
	tag_radius = sim_result["param"]["robots"][0]["tag_radius"]
	goal = np.array([sim_result["param"]["goal"][0],sim_result["param"]["goal"][1],goal_z])

	# Make data
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = tag_radius * np.outer(np.cos(u), np.sin(v)) + goal[0]
	y = tag_radius * np.outer(np.sin(u), np.sin(v)) + goal[1]
	z = tag_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + goal[2]

	# Plot the surface
	ax.plot_surface(x, y, z, color='green',alpha=0.6)
	ax.plot([0.0,goal[0]],[goal[1],goal[1]],[goal[2],goal[2]], color='green',linewidth=1,linestyle="--")
	ax.plot([goal[0],goal[0]],[y_lim[-1],goal[1]],[goal[2],goal[2]], color='green',linewidth=1,linestyle="--")
	ax.plot([goal[0],goal[0]],[goal[1],goal[1]],[0.0,goal[2]], color='green',linewidth=1,linestyle="--")

	for i_robot in range(nrobots):

		# trajectory 
		ax.plot(states[:,i_robot,0],states[:,i_robot,1],states[:,i_robot,2],color=colors[i_robot])

		# projections 
		ax.plot(np.zeros(states[:,i_robot,0].shape),states[:,i_robot,1],states[:,i_robot,2],color=colors[i_robot],linewidth=1,linestyle="--")
		ax.plot(states[:,i_robot,0],y_lim[-1]*np.ones(states[:,i_robot,1].shape),states[:,i_robot,2],color=colors[i_robot],linewidth=1,linestyle="--")
		ax.plot(states[:,i_robot,0],states[:,i_robot,1],np.zeros(states[:,i_robot,2].shape),color=colors[i_robot],linewidth=1,linestyle="--")

		ax.plot([0.0, states[0,i_robot,0]],[states[0,i_robot,1],states[0,i_robot,1]],[states[0,i_robot,2],states[0,i_robot,2]],color=colors[i_robot],linewidth=1,linestyle="--")
		ax.plot([states[0,i_robot,0],states[0,i_robot,0]],[y_lim[-1],states[0,i_robot,1]],[states[0,i_robot,2],states[0,i_robot,2]],color=colors[i_robot],linewidth=1,linestyle="--")
		ax.plot([states[0,i_robot,0],states[0,i_robot,0]],[states[0,i_robot,1],states[0,i_robot,1]],[0.0,states[0,i_robot,2]],color=colors[i_robot],linewidth=1,linestyle="--")

		# start 
		ax.plot([states[0,i_robot,0]],[states[0,i_robot,1]],states[0,i_robot,2],color=colors[i_robot],marker='s',markersize=10)

		# end 
		# Put special markers on attacker robot events
		if (sim_result["param"]["robots"][i_robot]["team"] == 'a') :
			# Find the last valid states
			idx_unkn = np.where(np.isnan(states[:,i_robot,0]) == True)
			idx_dead = np.where(np.isneginf(states[:,i_robot,0]) == True)
			idx_goal = np.where(np.isposinf(states[:,i_robot,0]) == True)

			# Plot events
			if (len(idx_unkn[0])) :
				# Robot is inactive
				idx = max(0,min(idx_unkn[0])-1)
				ax.plot([states[idx,i_robot,0]],[states[idx,i_robot,1]],states[idx,i_robot,2],linewidth=1,color=colors[i_robot],marker="|",markersize=10)
			if (len(idx_dead[0])) :
				# Robot is dead
				idx = max(0,min(idx_dead[0])-1)
				ax.plot([states[idx,i_robot,0]],[states[idx,i_robot,1]],states[idx,i_robot,2],linewidth=1,color=colors[i_robot],marker="x",markersize=10)
			if (len(idx_goal[0])) :
				# Robot is at the goal
				idx = max(0,min(idx_goal[0])-1)
				ax.plot([states[idx,i_robot,0]],[states[idx,i_robot,1]],states[idx,i_robot,2],linewidth=1,color=colors[i_robot],marker="o",markersize=10)
		
		# tag radius 
		# ax.add_patch(mpatches.Circle(states[-1,i,0:2], sim_result["param"]["robots"][i]["tag_radius"],color=colors[i],alpha=0.2,fill=False))

	set_axes_equal(ax)

	# 
	fig,axs = plt.subplots(nrows=2,ncols=max((sim_result["param"]["dynamics"]["state_dim"],sim_result["param"]["dynamics"]["control_dim"])),squeeze=False)

	times = sim_result["times"]

	for i_state, label in enumerate(sim_result["param"]["dynamics"]["state_labels"]):
		for i_robot in range(sim_result["param"]["num_nodes"]):
			axs[0,i_state].plot(times,states[:,i_robot,i_state],color=colors[i_robot])
		axs[0,i_state].set_title(label) 
		axs[0,i_state].grid(True)

	axs[0,0].set_ylim(x_lim)
	axs[0,1].set_ylim(y_lim)
	axs[0,2].set_ylim(z_lim)

	for i_control, label in enumerate(sim_result["param"]["dynamics"]["control_labels"]):
		for i_robot in range(sim_result["param"]["num_nodes"]):		
			axs[1,i_control].plot(times,actions[:,i_robot,i_control],color=colors[i_robot])
		axs[1,i_control].set_title(label) 
		axs[1,i_control].grid(True)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_tree_results(sim_result,title=None): 

	if sim_result["param"]["dynamics"]["name"] == "dubins_3d":
		return plot_3d_dubins_result(sim_result,title=title)

	def team_1_reward_to_gamma(reward_1):
		# gamma = reward_1 * 2 - 1 
		gamma = reward_1 
		return gamma 

	states = sim_result["states"]
	actions = sim_result["actions"]

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

	# first figure: state space and value function 

	fig,axs = plt.subplots(nrows=1,ncols=2,constrained_layout=True,squeeze=False)

	# state space
	ax = axs[0,0]
	ax.grid(True)
	ax.set_aspect('equal')
	ax.set_title('State Space')
	ax.add_patch(mpatches.Circle(goal, tag_radius, color=goal_color,alpha=0.5))
	for i in range(num_nodes):
		# Robot position (each time step)
		ax.plot(states[:,i,0],states[:,i,1],linewidth=1,color=colors[i],marker="o",markersize=0.75)
		# Tag radius (last time step)
		ax.add_patch(mpatches.Circle(states[-1,i,0:2], sim_result["param"]["robots"][i]["tag_radius"],color=colors[i],alpha=0.2,fill=False))
		# Put special markers on attacker robot events
		if (sim_result["param"]["robots"][i]["team"] == 'a') :
			# Find the last valid states
			idx_unkn = np.where(np.isnan(states[:,i,0]) == True)
			idx_dead = np.where(np.isneginf(states[:,i,0]) == True)
			idx_goal = np.where(np.isposinf(states[:,i,0]) == True)

			# Plot events
			if (len(idx_unkn[0])) :
				# Robot is unknown
				idx = max(0,min(idx_unkn[0])-1)
				ax.plot(states[idx,i,0],states[idx,i,1],linewidth=1,color=colors[i],marker="|",markersize=3)
			if (len(idx_dead[0])) :
				# Robot is dead
				idx = max(0,min(idx_dead[0])-1)
				ax.plot(states[idx,i,0],states[idx,i,1],linewidth=1,color=colors[i],marker="x",markersize=3)
			if (len(idx_goal[0])) :
				# Robot is at the goal
				idx = max(0,min(idx_goal[0])-1)
				ax.plot(states[idx,i,0],states[idx,i,1],linewidth=1,color=colors[i],marker="o",markersize=3)
		
	ax.set_xlim([env_xlim[0],env_xlim[1]])
	ax.set_ylim([env_ylim[0],env_ylim[1]])

	# value func
	ax = axs[0,1] 
	ax.grid(True)
	ax.set_title('Value Function')
	ax.plot(times,rewards[:,0],color='black',alpha=0.75)
	ax.set_ylim([0,1])

	# value model plotting
	for i in range(num_nodes):
		path_value_fnc = None
		if i in team_1_idxs and "path_value_fnc" in sim_result["param"]["policy_dict_a"]:
			path_value_fnc = sim_result["param"]["policy_dict_a"]["path_value_fnc"]	 
		elif i not in team_1_idxs and "path_value_fnc" in sim_result["param"]["policy_dict_b"]:
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

			ax.plot(times,mus,color=colors[i],alpha=0.5) 
			ax.fill_between(times,mus-sigmas,mus+sigmas,color=colors[i],alpha=0.25) 

	ax.set_xlim([0,times[-1]])
	ax.set_ylim([0,1])
	ax.set_aspect(times[-1])


	# second figure: state and control trajectories
	dynamics = sim_result["param"]["dynamics"] 
	fig,axs = plt.subplots(nrows=2,ncols=max((dynamics["state_dim"],dynamics["control_dim"])),squeeze=False)

	for i_state, label in enumerate(dynamics["state_labels"]):
		for i_robot in range(num_nodes):
			axs[0,i_state].plot(times,states[:,i_robot,i_state],color=colors[i_robot])
		axs[0,i_state].set_title(label) 
		axs[0,i_state].grid(True)

	for i_control, label in enumerate(dynamics["control_labels"]):
		for i_robot in range(num_nodes):		
			axs[1,i_control].plot(times,actions[:,i_robot,i_control],color=colors[i_robot])
		axs[1,i_control].set_title(label) 
		axs[1,i_control].grid(True)

	path_glas_model_a = None
	if "path_glas_model_a" in sim_result["param"]["policy_dict_a"]: 
		path_glas_model_a = sim_result["param"]["policy_dict_a"]["path_glas_model_a"]
	path_glas_model_b = None
	if "path_glas_model_b" in sim_result["param"]["policy_dict_b"]: 
		path_glas_model_b = sim_result["param"]["policy_dict_b"]["path_glas_model_b"]

	if path_glas_model_a is not None: 

		from learning.policy_emptynet import PolicyEmptyNet
		from learning_interface import format_data, global_to_local
		from param import Param 
		import torch 

		param_obj = Param()
		param_obj.from_dict(sim_result["param"])

		with torch.no_grad():
			model = PolicyEmptyNet(param_obj,"cpu")
			model.load_state_dict(torch.load(path_glas_model_a))

			for robot_idx in param_obj.team_1_idxs:

				mus = [] 
				sigmas = [] 
				ts = [] 

				for k,t in enumerate(times):

					if not np.isfinite(states[k,robot_idx,:]).all(): # non active robot 
						break

					o_a,o_b,relative_goal = global_to_local(states[k,:,:],param_obj,robot_idx)
					o_a,o_b,relative_goal = format_data(o_a,o_b,relative_goal)
					_,mu,logvar = model(o_a,o_b,relative_goal,training=True)

					mu = mu.detach().numpy().squeeze()
					sigma = torch.sqrt(torch.exp(logvar)).detach().numpy().squeeze()

					mus.append(mu)
					sigmas.append(sigma)
					ts.append(t)

				ts = np.array(ts)
				mus = np.array(mus)
				sigmas = np.array(sigmas)

				for i_control in range(dynamics["control_dim"]):
					axs[1][i_control].plot(ts,mus[:,i_control],color=colors[robot_idx],linestyle='--') 
					axs[1][i_control].fill_between(ts,mus[:,i_control]-sigmas[:,i_control],mus[:,i_control]+sigmas[:,i_control],color=colors[robot_idx],alpha=0.5) 

	if path_glas_model_b is not None: 

		from learning.policy_emptynet import PolicyEmptyNet
		from learning_interface import format_data, global_to_local
		from param import Param 
		import torch 

		param_obj = Param()
		param_obj.from_dict(sim_result["param"])

		with torch.no_grad():
			model = PolicyEmptyNet(param_obj,"cpu")
			model.load_state_dict(torch.load(path_glas_model_b))


			for robot_idx in param_obj.team_2_idxs:
				
				mus = [] 
				sigmas = [] 
				ts = []

				for k,t in enumerate(times):
					if not np.isfinite(states[k,robot_idx,:]).all(): # non active robot 
						break

					o_a,o_b,relative_goal = global_to_local(states[k,:,:],param_obj,robot_idx)
					o_a,o_b,relative_goal = format_data(o_a,o_b,relative_goal)
					_,mu,logvar = model(o_a,o_b,relative_goal,training=True)

					mu = mu.detach().numpy().squeeze()
					sigma = torch.sqrt(torch.exp(logvar)).detach().numpy().squeeze()

					mus.append(mu)
					sigmas.append(sigma)
					ts.append(t)

				ts = np.array(ts)
				mus = np.array(mus)
				sigmas = np.array(sigmas)

				for i_control in range(dynamics["control_dim"]):
					axs[1][i_control].plot(ts,mus[:,i_control],color=colors[robot_idx],linestyle='--') 
					axs[1][i_control].fill_between(ts,mus[:,i_control]-sigmas[:,i_control],mus[:,i_control]+sigmas[:,i_control],color=colors[robot_idx],alpha=0.5) 

	# add figure title 
	if title is not None: 
		fig.suptitle(title)	

	# tree vis 
	if len(sim_result["trees"]) > 0:

		max_trees = 5
		if len(sim_result["trees"]) > max_trees:
			sim_result["trees"] = sim_result["trees"][0:max_trees]


		for i_tree, data in enumerate(sim_result["trees"]):
			
			fig,axs = plt.subplots(nrows=1,ncols=3,squeeze=False,constrained_layout=True)
			fig2,axs2 = plt.subplots(nrows=1,ncols=num_nodes,squeeze=False,constrained_layout=True)
			# fig,axs = plt.subplots(nrows=len(sim_result["trees"]),ncols=2,squeeze=False,constrained_layout=True)

			for i_node in range(num_nodes):
				plot_tree(axs2[0,i_node],data,i_node)
				axs2[0,i_node].set_title("Robot {} at Tree Index {}".format(i_node,i_tree))

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
			ax = axs[0,0]
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
			ax = axs[0,1]
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
			ax = axs[0,2]
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

	if len(sim_result['root_rewards_over_time']) > 0:

		fig,axs = plt.subplots(nrows=len(sim_result["root_rewards_over_time"]),ncols=1,squeeze=False,constrained_layout=True)

		for i_tree, data in enumerate(sim_result["root_rewards_over_time"]):
			tree_time = sim_result["tree_params"][i_tree]["time"]
			tree_robot_idx = sim_result["tree_params"][i_tree]["robot_idx"]

			ax = axs[i_tree,0]
			ax.plot(data)
			ax.set_title('Reward over time at i={},t={}'.format(tree_robot_idx,tree_time))

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
		# print('candidate {}/{}: {}'.format(i_state,num_vis,candidate))

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

def plot_exp9(result):

	exp9_result = defaultdict(list)
	colors = dict()
	colors_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	curr_color = 0 

	for (dirname, team, model_number), losses in result.items():
		key = (dirname, team)
		losses = np.array(losses) # niters x 2 
		value = losses[:,0]
		exp9_result[key].extend(list(value))
		if dirname not in colors.keys():
			colors[dirname] = colors_list[curr_color]
			curr_color += 1 

	fig,axs = plt.subplots(ncols=2,squeeze=False)
	for (dirname, team), losses in exp9_result.items():
		if team == "a":
			idx = 0
		else:
			idx = 1 

		axs[0,idx].plot(losses,label=dirname,color=colors[dirname])

	axs[0,idx].legend(loc='upper right')






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
	dynamics_name = df_param.robot_types["standard_robot"]["dynamics"]
	env_xlim = df_param.env_xlim 
	env_ylim = df_param.env_ylim 
	nbins = 20
	num_vis = 10
	n_samples = 100
	eps = 0.01  

	if dynamics_name == "single_integrator":
		state_dim = 2 
	elif dynamics_name == "double_integrator":
		state_dim = 4 
	elif dynamics_name == "dubins_2d":
		state_dim = 4 	
	elif dynamics_name == "dubins_3d":
		state_dim = 6
	else: 
		exit('plot training dynamics not implemented')

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
		# print('candidate {}/{}: {}'.format(i_state,num_vis,candidate))

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
		
		axs[0][0].scatter(0,0,color=self_color)

		# - goal 
		axs[0][0].scatter(candidate[2][0],candidate[2][1],color=goal_color,alpha=0.5)

		# - neighbors 
		num_a = int(len(candidate[0])/state_dim)
		num_b = int(len(candidate[1])/state_dim)
		for robot_idx in range(num_a):
			axs[0][0].scatter(candidate[0][robot_idx*state_dim],candidate[0][robot_idx*state_dim+1],color=team_1_color)
		for robot_idx in range(num_b):
			axs[0][0].scatter(candidate[1][robot_idx*state_dim],candidate[1][robot_idx*state_dim+1],color=team_2_color)

		# velocities 
		if dynamics_name in ["double_integrator"]:
			vx = -1*candidate[2][2]
			vy = -1*candidate[2][3]

			axs[0][0].arrow(0,0,vx,vy,color=self_color,alpha=0.5)	

			for robot_idx in range(num_a):
				axs[0][0].arrow(candidate[0][robot_idx*state_dim],candidate[0][robot_idx*state_dim+1],\
					vx+candidate[0][robot_idx*state_dim+2],vy+candidate[0][robot_idx*state_dim+3],\
					color=team_1_color,alpha=0.5)
			for robot_idx in range(num_b):
				axs[0][0].arrow(candidate[1][robot_idx*state_dim],candidate[1][robot_idx*state_dim+1],\
					vx+candidate[1][robot_idx*state_dim+2],vy+candidate[1][robot_idx*state_dim+3],\
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

def plot_tree(ax, tree, robot_idx_j, zoom_on=False, env_xlim=None, env_ylim=None):
	# color by rewards 
	rewards = tree[:,1]
	size = plt.rcParams['lines.markersize'] ** 2

	# 
	position_idxs = 3 + 4*robot_idx_j + np.arange(2) 

	segments = []
	linewidths = [] 
	best_segments = [] 
	segment_colors = [] 
	node_colors = [] 
	poses = []
	cmap = cm.viridis
	for i_row,row in enumerate(tree):
		parentIdx = int(row[0])

		if np.isfinite(np.sum(row[position_idxs])) \
			and np.isfinite(np.sum(tree[parentIdx][position_idxs])) \
			and not np.isnan(np.sum(row[position_idxs])).any() \
			and not np.isnan(np.sum(tree[parentIdx][position_idxs])).any() :

			node_colors.append(cmap(rewards[i_row]))
			poses.append(row[position_idxs])

			if parentIdx >= 0:
				segments.append([row[position_idxs], tree[parentIdx][position_idxs]])
				if row[2] == 1 and tree[parentIdx][2] == 1:
					best_segments.append(segments[-1])
				segment_colors.append(cmap(rewards[i_row]))

	# ln_coll = matplotlib.collections.LineCollection(segments, linewidth=0.2, colors=segment_colors)
	# ax.add_collection(ln_coll)
	# ln_coll = matplotlib.collections.LineCollection(best_segments, colors='k', zorder=3, linewidth=1.0)
	# ax.add_collection(ln_coll)

	ln_coll = matplotlib.collections.LineCollection(segments, linewidth=0.2, colors='k', alpha=0.2)
	ax.add_collection(ln_coll)
	# ln_coll = matplotlib.collections.LineCollection(best_segments, colors='k', zorder=3, linewidth=1.0)
	# ax.add_collection(ln_coll)	

	# plot nodes 
	poses = np.array(poses)
	if poses.shape[0] > 0:
		# ax.scatter(poses[:,0],poses[:,1],c=node_colors,s=0.1*size)
		ax.scatter(poses[0,0],poses[0,1],c='k',s=0.1*size)

	if zoom_on: 
		buffer_val = 0.2 
		ax_lim_done = False
		if poses.shape[0] > 1: 
			xlims = [np.min(poses[:,0]), np.max(poses[:,0])]
			ylims = [np.min(poses[:,1]), np.max(poses[:,1])]
			xlims[0] = xlims[0] - buffer_val*(xlims[1]-xlims[0])
			xlims[1] = xlims[1] + buffer_val*(xlims[1]-xlims[0])
			ylims[0] = ylims[0] - buffer_val*(ylims[1]-ylims[0])
			ylims[1] = ylims[1] + buffer_val*(ylims[1]-ylims[0])

			if ylims[1] - ylims[0] > 0 and xlims[1] - xlims[0] > 0:
				ax_lim_done = True
				delta = (xlims[1] - xlims[0]) - (ylims[1] - ylims[0])
				if delta > 0:
					ylims[0] = ylims[0] - delta/2
					ylims[1] = ylims[1] + delta/2
				elif delta < 0: 
					xlims[0] = xlims[0] + delta/2
					xlims[1] = xlims[1] - delta/2				
				ax.set_xlim(xlims)
				ax.set_ylim(ylims)

		if not ax_lim_done: 
			ax.set_xlim([env_xlim[0],env_xlim[1]])
			ax.set_ylim([env_ylim[0],env_ylim[1]])
	
		# arrange 
		ax.set_aspect('equal')
		ax.grid(True)


def plot_exp6(sim_result,dirname):

	# takes one sim result that has many trees (for each timestep) and plots each of the trees
	# if MCTS sim mode the trees for each robot are identical 
	# if D_MCTS sim mode, each robot has a tree for each other robot 	

	env_xlim = sim_result["param"]["env_xlim"]
	env_ylim = sim_result["param"]["env_ylim"]

	def plot_state_space(ax,goal,sim_result,goal_color,robot_idxs,colors,states,env_xlim,env_ylim,zoom_on=False):
		ax.add_patch(mpatches.Circle(goal, sim_result["param"]["robots"][0]["tag_radius"], color=goal_color,alpha=0.5))
		for robot_idx_l in robot_idxs:
			if robot_idx_l in sim_result["param"]["team_1_idxs"]: 
				color = colors[0]
			elif robot_idx_l in sim_result["param"]["team_2_idxs"]: 
				color = colors[1]
			ax.plot(states[time_idxs,robot_idx_l,0],states[time_idxs,robot_idx_l,1],\
				linewidth=2.5,color=color,alpha=0.2,marker="o",markersize=2.5)
			ax.add_patch(mpatches.Circle(states[time_idxs[-1],robot_idx_l,0:2], sim_result["param"]["robots"][robot_idx_l]["tag_radius"],\
				color=color,alpha=0.2,fill=False))	
		if zoom_on: 
			ax.set_xlim([env_xlim[0],env_xlim[1]])
			ax.set_ylim([env_ylim[0],env_ylim[1]])
			ax.set_aspect('equal')
			ax.grid(True)


	# parameters 
	tree_timestep = sim_result["param"]["tree_timestep"] 
	goal = sim_result["param"]["goal"]
	env_xlim = sim_result["param"]["env_xlim"]
	env_ylim = sim_result["param"]["env_ylim"]
	sim_mode = sim_result["param"]["policy_dict"]["sim_mode"]
	n_robots = sim_result["param"]["num_nodes"]
	robot_idxs = list(range(n_robots))
	states = sim_result["states"]
	times = sim_result["times"]

	ncols = n_robots 
	if sim_mode == "MCTS":
		nrows = 1 
		plot_robot_idxs = [0]
	elif sim_mode == "D_MCTS":
		nrows = n_robots 
		plot_robot_idxs = robot_idxs 

	# other
	goal_color='green'
	size = plt.rcParams['lines.markersize'] ** 2
	colors = ['blue','red'] 

	# group by time and robot idx 
	tree_times = set()
	tree_results = dict()
	tree_params = dict()
	sim_params = dict()
	for i_tree, (tree,tree_param) in enumerate(zip(sim_result["trees"],sim_result["tree_params"])):
		tree_time = tree_param["time"]
		tree_robot_idx = tree_param["robot_idx"]
		key = (tree_time,tree_robot_idx)
		tree_times.add(tree_time)
		tree_results[key] = tree
		tree_params[key] = tree_param
	tree_times = sorted(list(tree_times))

	# plotting! 
	for i_tree_time, tree_time in enumerate(tree_times): 
		print('{}/{}'.format(i_tree_time,len(tree_times)))

		time_idxs = range(np.where(times == tree_time)[0][0]+1)

		fig,axs = plt.subplots(ncols=ncols+1 ,nrows=nrows,squeeze=False)
		fig.suptitle('Tree At t={}'.format(tree_time))
		fig.tight_layout()

		for robot_idx_i in plot_robot_idxs: 

			key = (tree_time,robot_idx_i) 

			# macro fig, plot state space and trees in zoom out mode 
			ax = axs[robot_idx_i,0]
			plot_state_space(ax,goal,sim_result,goal_color,robot_idxs,colors,states,env_xlim,env_ylim,zoom_on=True)
			
			# this means that robot_idx_i became inactive prior to the end of the simulation 
			if key not in tree_results.keys():
				continue
			tree = tree_results[key]
			visible_robot_idxs = []
			visible_robot_idxs.extend(tree_params[key]["tree_team_1_idxs"])
			visible_robot_idxs.extend(tree_params[key]["tree_team_2_idxs"])

			for robot_idx_j in visible_robot_idxs:
				plot_tree(ax, tree, robot_idx_j, zoom_on=False, env_xlim=env_xlim, env_ylim=env_ylim)

			# micro fig, zoom in on trees to see structure 
			for robot_idx_j in visible_robot_idxs: 

				ax = axs[robot_idx_i,robot_idx_j+1]
				
				# plot state space 
				plot_state_space(ax,goal,sim_result,goal_color,robot_idxs,colors,states,env_xlim,env_ylim)
				plot_tree(ax, tree, robot_idx_j, zoom_on=True, env_xlim=env_xlim, env_ylim=env_ylim)

		# save
		fig.savefig(os.path.join(dirname,"{:03.0f}.png".format(i_tree_time)), dpi=100)
		plt.close()

def save_video(png_directory,output_dir,output_file):
	# Combine images to form the movie
	print("Creating MP4...")
	# cmd = "ffmpeg -y -r 60 -i "+png_directory+"%03d.png -c:v libx264 -vf \"fps=60,format=yuv420p\" "+output_dir+"/"+output_file+".mp4"
	cmd = "ffmpeg -y -r 1.0 -i "+png_directory+"%03d.png "+output_dir+"/"+output_file+".mp4"
	os.system(cmd)


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

# def plot_exp4_results(all_sim_results):


# 	def policy_to_label(policy_dict):
# 		label = policy_dict["sim_mode"]
# 		# label += policy_dict["team"]
# 		if "path_glas_model_a" in policy_dict.keys() and policy_dict["path_glas_model_a"] is not None: 
# 			label += ' ' + os.path.basename(policy_dict["path_glas_model_a"]).split('.')[0]
# 		if "path_glas_model_b" in policy_dict.keys() and policy_dict["path_glas_model_b"] is not None: 
# 			label += ' ' + os.path.basename(policy_dict["path_glas_model_b"]).split('.')[0]
# 		if "path_value_fnc" in policy_dict.keys() and policy_dict["path_value_fnc"] is not None: 
# 			label += ' ' + os.path.basename(policy_dict["path_value_fnc"]).split('.')[0]
# 		if policy_dict["sim_mode"] in ["MCTS","D_MCTS"]:
# 			label += ' |n|:{}'.format(policy_dict["mcts_tree_size"]).split('.')[0]
# 		return label

# 	# key = (i_case, team, exp4_sim_mode)
# 	# value = (forallrobots, image)
# 	# sim_im = defaultdict(list)
# 	# predict_im = defaultdict(list)
# 	# policy_ims = defaultdict(list)
	
# 	sim_im = dict()
# 	predict_im = dict()
# 	policy_ims = dict()	
# 	dss = dict()
# 	params = dict()
# 	nominal_states = dict()

# 	for sim_result in all_sim_results: 
		
# 		i_case = sim_result["param"]["i_case"]
# 		team = sim_result["param"]["team"]
# 		policy_dict = sim_result["param"]["policy_dict"]
	
# 		key = (i_case, team, policy_to_label(policy_dict))

# 		if sim_result["param"]["exp4_prediction_type"] == "VALUE": 

# 			if key not in predict_im.keys():
# 				predict_im[key] = sim_result["value_ims"]
# 				policy_ims[key] = sim_result["policy_ims"]
# 			else:
# 				predict_im[key] += sim_result["value_ims"]
# 				policy_ims[key] += sim_result["policy_ims"]
			
# 		elif sim_result["param"]["exp4_prediction_type"] == "SIM": 

# 			if key not in sim_im.keys():
# 				sim_im[key] = sim_result["value_ims"]
# 			else:
# 				sim_im[key] += sim_result["value_ims"]

# 		else: 
# 			print('prediction mode: {} not recognized'.format(sim_result["param"]["exp4_prediction_type"]))
# 			exit()

# 		if i_case not in params.keys():
# 			params[i_case] = sim_result["param"]
# 			dss[i_case] = (sim_result["X"],sim_result["Y"])
# 			nominal_states[i_case] = sim_result["nominal_state"]

# 	# some global variables 
# 	attackerPolicyDicts = all_sim_results[0]["param"]["attackerPolicyDicts"]
# 	defenderPolicyDicts = all_sim_results[0]["param"]["defenderPolicyDicts"]
# 	n_cases = all_sim_results[0]["param"]["n_case"]
# 	n_trials = all_sim_results[0]["param"]["exp4_num_trials"]

# 	for i_case in range(n_cases): 

# 		for team in ["a","b"]:

# 			param = params[i_case]
# 			goal = param["goal"]
# 			nominal_state = nominal_states[i_case]
# 			X,Y = dss[i_case]

# 			robot_idxs = param["team_1_idxs"] if team == "a" else param["team_2_idxs"]
# 			policy_dicts = attackerPolicyDicts if team == "a" else defenderPolicyDicts

# 			colors = get_colors(param)

# 			for robot_idx in robot_idxs: 

# 				fig,axs = plt.subplots(nrows=len(policy_dicts), ncols=3,squeeze=False)

# 				fig.suptitle('Case: {} Value and Policy for Placing Robot: {} Team: {}'.format(i_case,robot_idx,team))

# 				for i_policy_dict, policy_dict in enumerate(policy_dicts): 

# 					key = (i_case, team, policy_to_label(policy_dict))

# 					# plot prediction
# 					ax = axs[i_policy_dict,0]
# 					# data = np.mean(np.array(predict_im[key]),axis=0)
# 					data = np.array(predict_im[key]) / n_trials
# 					im = ax.imshow(data[robot_idx,:,:].T,origin='lower',\
# 						extent=(X[0], X[-1], Y[0], Y[-1]))
# 						# extent=(X[0], X[-1], Y[0], Y[-1]),vmin=0,vmax=1)

# 					# plot simulated value
# 					ax = axs[i_policy_dict,1]
# 					# data = np.mean(np.array(sim_im[key]),axis=0)
# 					data = np.array(sim_im[key]) / n_trials
# 					im = ax.imshow(data[robot_idx,:,:].T,origin='lower',\
# 						extent=(X[0], X[-1], Y[0], Y[-1]))
# 						# extent=(X[0], X[-1], Y[0], Y[-1]),vmin=0,vmax=1)

# 					# plot policy 
# 					ax = axs[i_policy_dict,2]
# 					# data = np.mean(np.array(policy_ims[key])[:,robot_idx,:,:],axis=0)
# 					data = np.array(policy_ims[key])[robot_idx,:,:] / n_trials
# 					data = np.transpose(data,axes=(1,0,2))
# 					C = np.linalg.norm(data,axis=2)
# 					ax.quiver(np.array(X),np.array(Y),data[:,:,0],data[:,:,1],width=0.01)
# 					ax.imshow(C,origin='lower',extent=(X[0], X[-1], Y[0], Y[-1]))

# 				# plot state and arrange 
# 				for i_x, mode in enumerate(["Predict","Sim","Policy"]):
# 					for i_y, policy_dict in enumerate(policy_dicts):
# 						ax = axs[i_y,i_x]

# 						# plot state on top of axis 
# 						ax.scatter(goal[0],goal[1],color='green')
# 						for robot_idx_j, robot_state_j in enumerate(nominal_state):
# 							if robot_idx_j == robot_idx: 
# 								continue 
# 							ax.scatter(robot_state_j[0],robot_state_j[1],color=colors[robot_idx_j])

# 						# arrange 
# 						ax.set_xticks(X)
# 						ax.set_yticks(Y)
# 						ax.grid(True,linestyle='-',linewidth=1,alpha=0.2,color='black')
# 						ax.set_xticklabels([])
# 						ax.set_yticklabels([])

# 						if i_x == 0:
# 							ax.set_ylabel(policy_to_label(policy_dict))
# 						if i_y == 0: 
# 							ax.set_xlabel(mode)
# 							ax.xaxis.set_label_position('top')

def plot_exp4_results(all_sim_results):


	def policy_to_label(policy_dict):
		label = policy_dict["sim_mode"]
		# label += policy_dict["team"]
		if "path_glas_model_a" in policy_dict.keys() and policy_dict["path_glas_model_a"] is not None: 
			label += ' ' + os.path.basename(policy_dict["path_glas_model_a"]).split('.')[0]
		if "path_glas_model_b" in policy_dict.keys() and policy_dict["path_glas_model_b"] is not None: 
			label += ' ' + os.path.basename(policy_dict["path_glas_model_b"]).split('.')[0]
		if "path_value_fnc" in policy_dict.keys() and policy_dict["path_value_fnc"] is not None: 
			label += ' ' + os.path.basename(policy_dict["path_value_fnc"]).split('.')[0]
		if policy_dict["sim_mode"] in ["MCTS","D_MCTS"]:
			label += ' |n|:{}'.format(policy_dict["mcts_tree_size"]).split('.')[0]
		return label

	# key = (i_case, team, exp4_sim_mode)
	# value = (forallrobots, image)
	sim_im = defaultdict(list)
	predict_im = defaultdict(list)
	policy_ims = defaultdict(list)
	dss = dict()
	params = dict()
	nominal_states = dict()

	for sim_result in all_sim_results: 
		
		i_case = sim_result["param"]["i_case"]
		team = sim_result["param"]["team"]
		policy_dict = sim_result["param"]["policy_dict"]
	
		key = (i_case, team, policy_to_label(policy_dict))

		if sim_result["param"]["exp4_prediction_type"] == "VALUE": 
			predict_im[key].append(sim_result["value_ims"])
			policy_ims[key].append(sim_result["policy_ims"])
		elif sim_result["param"]["exp4_prediction_type"] == "SIM": 
			sim_im[key].append(sim_result["value_ims"])
		else: 
			print('prediction mode: {} not recognized'.format(sim_result["param"]["exp4_prediction_type"]))
			exit()

		if i_case not in params.keys():
			params[i_case] = sim_result["param"]
			dss[i_case] = (sim_result["X"],sim_result["Y"])
			nominal_states[i_case] = sim_result["nominal_state"]

	# some global variables 
	attackerPolicyDicts = all_sim_results[0]["param"]["attackerPolicyDicts"]
	defenderPolicyDicts = all_sim_results[0]["param"]["defenderPolicyDicts"]
	n_cases = all_sim_results[0]["param"]["n_case"]

	for i_case in range(n_cases): 

		for team in ["a","b"]:

			param = params[i_case]
			goal = param["goal"]
			nominal_state = nominal_states[i_case]
			X,Y = dss[i_case]

			robot_idxs = param["team_1_idxs"] if team == "a" else param["team_2_idxs"]
			policy_dicts = attackerPolicyDicts if team == "a" else defenderPolicyDicts

			colors = get_colors(param)

			for robot_idx in robot_idxs: 

				fig,axs = plt.subplots(nrows=len(policy_dicts), ncols=3,squeeze=False)

				fig.suptitle('Case: {} Value and Policy for Placing Robot: {} Team: {}'.format(i_case,robot_idx,team))

				for i_policy_dict, policy_dict in enumerate(policy_dicts): 

					key = (i_case, team, policy_to_label(policy_dict))

					# plot prediction
					ax = axs[i_policy_dict,0]
					data = np.mean(np.array(predict_im[key]),axis=0)
					im = ax.imshow(data[robot_idx,:,:].T,origin='lower',\
						extent=(X[0], X[-1], Y[0], Y[-1]))
						# extent=(X[0], X[-1], Y[0], Y[-1]),vmin=0,vmax=1)

					# plot simulated value
					ax = axs[i_policy_dict,1]
					data = np.mean(np.array(sim_im[key]),axis=0)
					im = ax.imshow(data[robot_idx,:,:].T,origin='lower',\
						extent=(X[0], X[-1], Y[0], Y[-1]))
						# extent=(X[0], X[-1], Y[0], Y[-1]),vmin=0,vmax=1)

					# plot policy 
					ax = axs[i_policy_dict,2]
					data = np.mean(np.array(policy_ims[key])[:,robot_idx,:,:],axis=0)
					data = np.transpose(data,axes=(1,0,2))
					C = np.linalg.norm(data,axis=2)
					ax.quiver(np.array(X),np.array(Y),data[:,:,0],data[:,:,1],width=0.01)
					ax.imshow(C,origin='lower',extent=(X[0], X[-1], Y[0], Y[-1]))

				# plot state and arrange 
				for i_x, mode in enumerate(["Predict","Sim","Policy"]):
					for i_y, policy_dict in enumerate(policy_dicts):
						ax = axs[i_y,i_x]

						# plot state on top of axis 
						ax.scatter(goal[0],goal[1],color='green')
						for robot_idx_j, robot_state_j in enumerate(nominal_state):
							if robot_idx_j == robot_idx: 
								continue 
							ax.scatter(robot_state_j[0],robot_state_j[1],color=colors[robot_idx_j])

						# arrange 
						ax.set_xticks(X)
						ax.set_yticks(Y)
						ax.grid(True,linestyle='-',linewidth=1,alpha=0.2,color='black')
						ax.set_xticklabels([])
						ax.set_yticklabels([])

						if i_x == 0:
							ax.set_ylabel(policy_to_label(policy_dict))
						if i_y == 0: 
							ax.set_xlabel(mode)
							ax.xaxis.set_label_position('top')

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

	dirname = "None"
	for key, value in policy.items():
		if "path" in key and value is not None: 
			# print('value',value)
			# print('os.path.dirname(value)',os.path.dirname(value))
			dirname = os.path.dirname(value)
			break 

	# add dir where it comes from 
	label = dirname + " " + label
	
	return label

def plot_exp7_results(all_sim_results):

	# read results into dict 
	rw_results = defaultdict(list) # game reward
	rg_results = defaultdict(list) # reached goal reward 
	model_names_a = set()
	model_names_b = set()
	c_params = set()
	for sim_result in all_sim_results: 
		# key = (test_team, tree size, model, c_param)
		test_team = sim_result["param"]["test_team"]
		if test_team == "a":
			tree_size = sim_result["param"]["policy_dict_a"]["mcts_tree_size"]
			model_name = sim_result["param"]["policy_dict_a"]["path_glas_model_a"]
			c_param = sim_result["param"]["policy_dict_a"]["mcts_c_param"]
			model_names_a.add(model_name)
		elif test_team == "b":
			tree_size = sim_result["param"]["policy_dict_b"]["mcts_tree_size"]
			model_name = sim_result["param"]["policy_dict_b"]["path_glas_model_b"]
			c_param = sim_result["param"]["policy_dict_b"]["mcts_c_param"]
			model_names_b.add(model_name)

		c_params.add(c_param)
		key = (test_team, tree_size, model_name, c_param)
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

			for c_param in c_params: 

				mean_data = []
				std_data = []
				for tree_size in tree_sizes:

					key = (test_team, tree_size, model_name, c_param)
					mean_data.append(np.mean(np.array(rg_results[key])))

				label = 'None' if model_name is None else os.path.basename(model_name)
				label += ' c = {}'.format(c_param)

				ax.plot(tree_sizes,mean_data,color=colors[i_ax][i_model],label=label)

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
		if mean_result.shape[0] > 8:
			ax = sns.heatmap(mean_result.T,vmin=0,vmax=1,annot=True,annot_kws={"size":4})
		else:
			ax = sns.heatmap(mean_result.T,vmin=0,vmax=1,annot=True)
		# fig.colorbar(im)
		# ax.set_xticks(range(len(attackerPolicies)))
		# ax.set_yticks(range(len(defenderPolicies)))
		ax.set_xticklabels(xticklabels,rotation=45, ha='right')
		ax.set_yticklabels(yticklabels,rotation=45, ha='right')
		ax.tick_params(axis='both',labelsize=5)
		ax.set_xlabel('attackers')
		ax.set_ylabel('defenders')
		fig.suptitle(title)
		fig.tight_layout()
		# ax.legend()
		# ax.grid(True)

	# make plots with same data easier to understand: 
	# two plots (attacker and defender) 
	# x-axis learning iteraiton (or flat line)
	# y-axis performance : mean + std in 
	fig,axs = plt.subplots(nrows=1,ncols=2,squeeze=False)
	# get colors 
	colors = get_n_colors(len(attackerPolicies))

	for i in range(2):

		to_plot = []
		other_to_plots = dict()

		if i==0:
			self_policies = attackerPolicies
			adv_policies = defenderPolicies
		elif i==1:
			self_policies = defenderPolicies
			adv_policies = attackerPolicies

		for idx, policy_dict in enumerate(self_policies):
			
			if i == 0:
				J_mean = np.sum(mean_rw_result[idx,:])/len(self_policies)
				J_std = np.sqrt(np.sum(np.square(std_rw_result[idx,:])))/len(self_policies)
			else:
				J_mean = np.sum(mean_rw_result[:,idx])/len(self_policies)
				J_std = np.sqrt(np.sum(np.square(std_rw_result[:,idx])))/len(self_policies)
			
			if policy_dict["sim_mode"] == "D_MCTS":

				# get learning index 
				if policy_dict["path_glas_model_a"] is not None: 
					learning_idx = os.path.basename(policy_dict["path_glas_model_a"]).split(".")[0][1:]
				else:
					learning_idx = 0 

				to_plot.append((learning_idx,\
					J_mean,\
					J_std,\
					idx))

			elif policy_dict["sim_mode"] == "MCTS":

				if policy_dict["path_glas_model_a"] is not None: 
					key = "Biased MCTS"
				else: 
					key = "Unbiased MCTS"

				other_to_plots[key] = (\
					J_mean,\
					J_std,
					idx)

			elif policy_dict["sim_mode"] == "PANAGOU":

				key = "PANGAOU"

				other_to_plots[key] = (\
					J_mean,\
					J_std,
					idx)

		# sort 
		sorted(to_plot, key=lambda x: x[0])
		to_plot = np.array(to_plot,dtype=np.float32)
		color_dmcts_idx = int(to_plot[0,3])
		axs[0,i].plot(to_plot[:,0],to_plot[:,1],color=colors[color_dmcts_idx],label="DMCTS")
		axs[0,i].fill_between(to_plot[:,0], to_plot[:,1]-to_plot[:,2], to_plot[:,1]+to_plot[:,2],color=colors[color_dmcts_idx],alpha=0.5)
		for key,value in other_to_plots.items():
			axs[0,i].plot([to_plot[0,0],to_plot[-1,0]],[value[0],value[0]],color=colors[value[2]],label=key)
			axs[0,i].fill_between(to_plot[:,0], value[0]-value[1], value[0]+value[1],color=colors[value[2]],alpha=0.5)

		axs[0,i].set_ylim([0,1])
		axs[0,i].set_xticks(to_plot[:,0])
		axs[0,i].grid(True)
		axs[0,i].set_xlabel("Learning Iteration")
		axs[0,i].set_ylabel("Performance")
		axs[0,i].set_aspect("equal")
	
		if i == 1:
			axs[0,i].legend(loc='best')

	fig.tight_layout()



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