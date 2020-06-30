
import numpy as np 

class Param:

	def __init__(self):

		self.dynamics_A_names = [
			'system/identity.py',
		]

		self.measurements_A_name = [
			'system/global.py',
		]

		self.dynamics_B_names = [
			'system/identity.py',
		]

		self.measurements_B_name = [
			'system/global.py',
		]		

		self.attacker_names = [
			'attacker/empty.py',
		]

		self.estimator_A_names = [
			'estimator/empty.py',
			# 'estimator/kalman.py'
		]			

		self.controller_A_names = [
			'controller/empty.py',
			# 'controller/team_A_lqg.py',
		]

		self.estimator_B_names = [
			'estimator/empty.py',
			# 'estimator/kalman.py'
		]			

		self.controller_B_names = [
			'controller/empty.py',
			# 'controller/team_A_lqg.py',
		]		

		# sim param 
		self.n_trials = 1
		self.sim_t0 = 0
		self.sim_tf = 5
		self.sim_dt = 0.5
		
		# topology
		self.r_sense = 0.6
		self.r_comm = 0.6
		
		# environment
		self.env_xlim = [0,1]
		self.env_ylim = [0,1]
		self.reset_xlim_A = [0,0.2]
		self.reset_ylim_A = [0,1]
		self.reset_xlim_B = [0.8,1]
		self.reset_ylim_B = [0,1]		
		self.goal_line_x = 0.6
		
		# nodes 
		self.num_nodes_A = 2
		self.num_nodes_B = 1
		
		# estimator parameters
		self.initial_state_covariance = 3e1 # defines initial condition of estimators
		
		# dynamics parameters
		self.process_noise_covariance = 1.0 
		
		# measurement parameters
		self.measurement_noise_covariance = 1.0 

		# policy 
		self.rhc_horizon = 5
		self.lambda_u = 0.1
		self.speed_limit_a = 0.1
		self.speed_limit_b = 0.1
		self.acceleration_limit_a = 0.05
		self.acceleration_limit_b = 0.05
		self.danger_radius = 0.2

		# path stuff
		self.current_results_dir = '../current_results'
		
		# plotting 
		self.plot_fn = 'plots.pdf'
		
		# save stuff 
		self.info_keys = [
			'state',
			'node_idx',
			'node_state_estimate',
			'node_state_covariance',
			'node_state',
			'node_team_A',
			'node_team_B',
		]

		self.update()


	def to_dict(self):
		return self.__dict__


	def update(self):
		self.num_nodes = self.num_nodes_A + self.num_nodes_B
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)			