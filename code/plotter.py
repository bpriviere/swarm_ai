

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import os, subprocess
import matplotlib.patches as mpatches

from matplotlib import cm	
from matplotlib.backends.backend_pdf import PdfPages 

from utilities import dbgp

# defaults
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 2.5

def has_figs():
	if len(plt.get_fignums()) > 0:
		return True
	else:
		return False


def save_figs(filename):
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


def make_fig():
	return plt.subplots()


def plot_nodes(sim_result, timestep, fig=None, ax=None):

	times = sim_result["times"]
	node_states = sim_result["info"]["node_state"] # [num timesteps, num nodes, state_dim_per_agent, 1]
	reset_xlim_A = sim_result["param"]["reset_xlim_A"]
	reset_ylim_A = sim_result["param"]["reset_ylim_A"]
	reset_xlim_B = sim_result["param"]["reset_xlim_B"]
	reset_ylim_B = sim_result["param"]["reset_ylim_B"]
	goal_line_x = sim_result["param"]["goal_line_x"]
	colors = get_node_colors(sim_result, timestep)

	if fig is None or ax is None:
		fig,ax = plt.subplots()
	
	# plot nodes
	for node_idx in range(node_states.shape[1]):
		node_state = node_states[timestep,node_idx,:,:]		
		ax.scatter(node_state[0],node_state[1],100,color=colors[node_idx],zorder=10)

	# plot trajectories 
	for node_idx in range(node_states.shape[1]):
		node_trajectory_x = node_states[:,node_idx,0,:]	
		node_trajectory_y = node_states[:,node_idx,1,:]	
		ax.plot(node_trajectory_x,node_trajectory_y,color='black',linestyle='--',alpha=0.25)	

	# plot initialization 
	reset_a = patches.Rectangle((reset_xlim_A[0],reset_ylim_A[0]),\
		reset_xlim_A[1]-reset_xlim_A[0],reset_ylim_A[1]-reset_ylim_A[0],color=colors[0],alpha=0.1)
	reset_b = patches.Rectangle((reset_xlim_B[0],reset_ylim_B[0]),\
		reset_xlim_B[1]-reset_xlim_B[0],reset_ylim_B[1]-reset_ylim_B[0],color=colors[-1],alpha=0.1)
	ax.add_patch(reset_a)
	ax.add_patch(reset_b)

	# plot goal line 
	ax.axvline(goal_line_x,color='green',alpha=0.5,linestyle='--')

	# ax.plot(np.nan,np.nan,color=colors[0],label='Team A')
	# ax.plot(np.nan,np.nan,color=colors[-1],label='Team B')
	# ax.legend(loc='upper right')
	ax.set_xlim(sim_result["param"]["env_xlim"])
	ax.set_ylim(sim_result["param"]["env_ylim"])
	ax.grid(True)
	ax.set_aspect('equal')
	ax.set_xlabel('pos [m]')
	ax.set_ylabel('pos [m]')
	ax.set_title('State Space At Time {}'.format(times[timestep]))

	return fig,ax


def plot_state_estimate(sim_result):

	node_state_means = sim_result["info"]["node_state_mean"] # [num timesteps, num nodes, state dim, 1]
	node_state_covariance = sim_result["info"]["node_state_covariance"] # [num timesteps, num nodes, state dim, state dim]
	states = sim_result["info"]["state_vec"] # [nt, state_dim, 1]
	times = sim_result["times"]
	colors = get_node_colors(sim_result, 0)

	fig,ax = plt.subplots()

	# plot mse and covariance  
	for node_idx in range(node_state_means.shape[1]):

		mse = np.linalg.norm(node_state_means[:,node_idx,:,:] - states, axis=1)
		trace_covariance = np.linalg.norm(node_state_covariance[:,node_idx,:,:], ord = 'fro', axis=(1,2))

		ax.plot(times, mse, color=colors[node_idx], alpha=0.5)
		ax.plot(times, trace_covariance, color=colors[node_idx], alpha=0.5, linestyle = '--')

	# ax.plot(np.nan,np.nan,color=colors[0],label='Team A')
	# ax.plot(np.nan,np.nan,color=colors[-1],label='Team B')
	ax.legend(loc='upper right')
	ax.set_xlabel('time')
	ax.set_ylabel('error')
	ax.set_yscale('log')
	ax.grid(True)

	return fig,ax


def plot_control_effort(sim_result):

	times = sim_result["times"]
	actions = sim_result["actions"] # [nt, ni, control_dim x 1]
	colors = get_node_colors(sim_result, 0)

	fig,ax = plt.subplots() 
	for node_idx in range(actions.shape[1]):
		effort = np.linalg.norm(actions[:,node_idx,:,0],axis=1)
		ax.plot(times, effort, color=colors[node_idx], alpha=0.5)

	ax.set_xlabel('time')
	ax.set_ylabel('effort')
	ax.grid(True)	

	return fig,ax


def plot_speeds(sim_result):

	times = sim_result["times"]
	node_states = sim_result["info"]["node_state"] # [num timesteps, num nodes, state_dim_per_agent, 1]
	colors = get_node_colors(sim_result, 0)

	fig,ax = plt.subplots() 
	for node_idx in range(node_states.shape[1]):
		speed = np.linalg.norm(node_states[:,node_idx,2:,0],axis=1)
		ax.plot(times, speed, color=colors[node_idx], alpha=0.5)

	ax.set_xlabel('time')
	ax.set_ylabel('speed')
	ax.grid(True)	

	return fig,ax


def get_node_colors(sim_result, timestep=0):

	# from https://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
	from matplotlib import cm

	node_idx = sim_result["info"]["node_idx"] # [nt x ni] 
	node_team_A = sim_result["info"]["node_team_A"] # [nt x ni] 
	node_team_B = sim_result["info"]["node_team_B"] # [nt x ni] 

	n_agent = len(node_idx[timestep])

	# some param 
	start = 0.55
	stop = 0.56
	number_of_lines= n_agent
	cm_subsection = np.linspace(start, stop, number_of_lines) 

	colors_A = [ cm.Blues(x) for x in cm_subsection]
	colors_B = [ cm.Reds(x) for x in cm_subsection]

	colors = []
	for i in node_idx[0]:
		if node_team_A[timestep][i]:
			colors.append(colors_A[i])
		elif node_team_B[timestep][i]:
			colors.append(colors_B[i])
		else:
			print('theta value not understood')
			exit()

	return colors


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


def plot_sa_pairs(states,actions,param,instance):

	states = np.asarray(states) # nt x state_dim 
	actions = np.asarray(actions) # nt x action dim 

	fig,ax = plt.subplots()

	team_1_color = 'blue'
	team_2_color = 'orange'
	goal_color = 'green'

	ax.add_patch(mpatches.Circle(param.goal, param.tag_radius, color=goal_color,alpha=0.5))

	for node_idx in range(param.num_nodes):

		pos_x_idx = 4*node_idx + 0 
		pos_y_idx = 4*node_idx + 1 
		action_idx = 2*node_idx + np.arange(2)

		if node_idx in param.team_1_idxs:
			color = team_1_color
		elif node_idx in param.team_2_idxs:
			color = team_2_color

		ax.plot(states[:,pos_x_idx],states[:,pos_y_idx],linewidth=3,color=color)
		ax.scatter(states[:,pos_x_idx],states[:,pos_y_idx],color=color)

	ax.set_xlim(param.env_xlim)
	ax.set_ylim(param.env_ylim)
	ax.grid(True)
	ax.set_aspect('equal')
	ax.set_xlabel('pos [m]')
	ax.set_ylabel('pos [m]')
	ax.set_title('instance {}'.format(instance))


def plot_loss(losses,team):

	losses = np.array(losses)

	fig,ax = plt.subplots()

	ax.plot(losses[:,0],label='train')
	ax.plot(losses[:,1],label='test')
	
	ax.legend()
	ax.set_ylabel('mse')
	ax.set_xlabel('epoch')
	ax.set_yscale('log')
	ax.set_title('Team {}'.format(team))
	ax.grid(True)
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


def plot_tree_results(sim_result): 

	times = sim_result["times"]
	states = sim_result["states"]
	actions = sim_result["actions"]
	rewards = sim_result["rewards"]
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

	fig,axs = plt.subplots(nrows=2,ncols=2) 

	# state space
	ax = axs[0,0]
	ax.grid(True)
	ax.set_aspect('equal')
	ax.set_title('State Space')
	ax.add_patch(mpatches.Circle(goal, tag_radius, color=goal_color,alpha=0.5))
	for i in range(num_nodes):
		for t in range(states.shape[0]):
			ax.add_patch(mpatches.Circle(states[t,i,0:2], sim_result["param"]["robots"][i]["tag_radius"], \
				color=colors[i],alpha=0.2,fill=False))
		ax.plot(states[:,i,0],states[:,i,1],linewidth=3,color=colors[i])
		ax.scatter(states[:,i,0],states[:,i,1],marker='o',color=colors[i])
	ax.set_xlim([env_xlim[0],env_xlim[1]])
	ax.set_ylim([env_ylim[0],env_ylim[1]])

	# value func
	ax = axs[0,1] 
	ax.grid(True)
	ax.set_title('Value Function')
	ax.plot(times,rewards[:,0],color=team_1_color,label='attackers')
	ax.plot(times,rewards[:,1],color=team_2_color,label='defenders')
	ax.legend()

	# time varying velocity
	ax = axs[1,0]
	ax.grid(True)
	ax.set_title('Speed Profile')
	for i in range(num_nodes):
		ax.axhline(sim_result["param"]["robots"][i]["speed_limit"],color=colors[i],linestyle='--')
		ax.plot(times,np.linalg.norm(states[:,i,2:],axis=1),color=colors[i])

	# time varying acc
	ax = axs[1,1]
	ax.grid(True)
	ax.set_title('Acceleration Profile')
	for i in range(num_nodes):
		ax.axhline(sim_result["param"]["robots"][i]["acceleration_limit"],color=colors[i],linestyle='--')
		ax.plot(times,np.linalg.norm(actions[:,i],axis=1),color=colors[i])

	fig.tight_layout()

if __name__ == '__main__':
	import argparse
	import datahandler

	parser = argparse.ArgumentParser()
	parser.add_argument("file", help="pickle file to visualize")
	parser.add_argument("--outputPDF", help="output pdf file")
	parser.add_argument("--outputMP4", help="output video file")

	# parser.add_argument("--animate", action='store_true', help="animate using meshlab")
	args = parser.parse_args()

	sim_result = datahandler.load_sim_result(args.file)

	if args.outputPDF:
		plot_tree_results(sim_result)

		save_figs(args.outputPDF)
		open_figs(args.outputPDF)

	if args.outputMP4:
		# Matt's magic
		pass