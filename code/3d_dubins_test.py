
import numpy as np 

class DubinDynamics3D:

	def __init__(self):

		# state : [p_x,p_y,p_z,phi,psi,v]
		# control : [phidot,psidot,vdot]

		self.state_dim = 6 
		self.control_dim = 2 


	def step(self,state,control,timestep):

		num_robots, state_dim = state.shape

		next_state = np.zeros((num_robots,state_dim))

		for i in range(num_robots):
			
			dynamics_i = np.zeros((state_dim))

			phi = state[i,3]
			psi = state[i,4]
			phidot = control[i,0]
			psidot = control[i,1]

			v = np.array((
				(self.speed*np.cos(phi)*np.sin(psi)),
				(self.speed*np.cos(phi)*np.cos(psi)),
				(self.speed*np.sin(phi)),
				))
			w = np.array((
				(psidot),
				(phidot*np.sin(phi)),
				(-phidot*np.cos(psi)),
				))

			dynamics_i[0:3] = v 
			dynamics_i[3] = phidot
			dynamics_i[4] = psidot
			# dynamics_i[3:5] = np.cross(v,w)

			# integrate 
			next_state[i,:] = state[i,:] + dynamics_i * timestep 

		return next_state 

def sample_control(control_space,num_robots):

	# control space is hypercube: control_dim x 2 with lim for each dim

	import random 

	control_dim = control_space.shape[0]
	control = np.zeros((control_dim,num_robots))

	for i_control in range(control_dim):
		for i_robot in range(num_robots):
			control[i_control,i_robot] = control_space[i_control,0] + random.random()*\
				(control_space[i_control,1] - control_space[i_control,0]) 

	return control 


if __name__ == '__main__':
	
	# test 
	from param import Param 
	import plotter 

	param = Param()
	dynamics = DubinDynamics3D()

	num_robots = len(param.robots)
	max_freq = 2*np.pi / 5 # one period in 5 seconds 
	control_space = max_freq*np.array((
		(-1,1),(-1,1)
		))

	curr_state = np.array(param.state)
	states,controls,times = [curr_state],[],[]
	for time in np.arange(0,1,param.sim_dt): 

		control = sample_control(control_space,num_robots)
		next_state = dynamics.step(curr_state,control,param.sim_dt)

		curr_state = next_state 
		states.append(curr_state)
		controls.append(control)
		times.append(time)

	sim_result = {
		"param" : param.to_dict(),
		"times" : np.array(times), 
		"states" : np.array(states), 
		"actions" : np.array(controls),
	}

	plotter.plot_tree_results(sim_result,title="testing dubins 3d")

	print('saving and opening figs...')
	plotter.save_figs('plots/3d_dubins_test.pdf')
	plotter.open_figs('plots/3d_dubins_test.pdf')