

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages 

from utilities import dbgp

# defaults
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 2.5


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

# Function to detect when collisions occur in a trajectory
# We really only need to do this once but that's a bit beyond me at the moment
def calc_idx_collisions(pos_x,pos_y):
	# Calculates the indicies of captures by detecting large jumps in the array
	#print("\tChecking for collisions\n")
	idx = np.empty(0,int)

	for ii in range(0,pos_x.size-2):
		dist_between_steps = (pos_x[ii+1]-pos_x[ii])**2 + (pos_y[ii+1]-pos_y[ii])**2
		if (dist_between_steps > 0.1**2):
			#print("\t\tCollision at idx: "+str(ii))
			idx = np.append(idx,ii)

	return idx

# Function to plot the current timestep
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
	
	# Loop through each agent for plotting
	for node_idx in range(node_states.shape[1]):
		#print("idx = ",timestep,", Agent: ",node_idx)

		# Extract trajectories
		node_trajectory_x = sim_result["states"][:,node_idx,0,:]
		node_trajectory_y = sim_result["states"][:,node_idx,1,:]

		# Calculate and remove paths when collisions reset
		idx = calc_idx_collisions(node_trajectory_x,node_trajectory_y)

		# Extract node data
		node_state = node_states[timestep,node_idx,:,:]	

		# Plot node ("o" if normal, "x" if captured)
		if np.any(timestep == idx-1):
			ax.scatter(node_state[0],node_state[1],200,color=colors[node_idx],zorder=10,marker="x")
		else : 
			ax.scatter(node_state[0],node_state[1],100,color=colors[node_idx],zorder=10,marker="o")

		# Plot trajectories
		idx_start = np.hstack((0, idx+1))
		idx_stop  = np.hstack((idx,node_trajectory_x.size))
		for ii in range(0, idx_start.size):
			plot_x = node_trajectory_x[idx_start[ii]:idx_stop[ii]]
			plot_y = node_trajectory_y[idx_start[ii]:idx_stop[ii]]
			ax.plot( plot_x,plot_y,color='black',linestyle='--',alpha=0.25)

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
	ax.set_title('State Space At Time {:.2f}'.format(times[timestep]))

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

	from env import Swarm 

	env = Swarm(param)

	colors = ['red','blue']

	for timestep,(state,action) in enumerate(zip(states,actions)):

		fig,ax = plt.subplots()

		# first update state 
		state_dict = env.state_vec_to_dict(state)
		for node in env.nodes: 
			node.state = state_dict[node]
			ax.scatter(node.state[0],node.state[1],100,color=colors[node.team_A],zorder=10)

		ax.axvline(param.goal_line_x,color='green',alpha=0.5,linestyle='--')

		ax.set_xlim(param.env_xlim)
		ax.set_ylim(param.env_ylim)
		ax.grid(True)
		ax.set_aspect('equal')
		ax.set_xlabel('pos [m]')
		ax.set_ylabel('pos [m]')
		ax.set_title('{} at time {}'.format(instance,timestep))


def plot_loss(losses):

	losses = np.array(losses)

	fig,ax = plt.subplots()

	ax.plot(losses[:,0],label='train')
	ax.plot(losses[:,1],label='test')
	
	ax.legend()
	ax.set_ylabel('mse')
	ax.set_xlabel('epoch')
	ax.set_yscale('log')
	ax.grid(True)
	fig.tight_layout()