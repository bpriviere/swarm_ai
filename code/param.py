
import numpy as np 

class Param:

	def __init__(self):

		# modules 
		self.dynamics_name 		= 'dynamics/double_integrator.py' 	# 
		self.measurements_name 	= 'measurements/global.py' 			# global, (local)
		self.estimator_name 	= 'estimator/kalman.py'	 			# empty,(kalman),exact...
		self.attacker_name 		= 'attacker/empty.py' 				# empty, ...
		self.controller_name 	= 'controller/joint_mpc.py'		 	# empty, glas, joint_mpc, ...

		# flags
		self.gif_on 	= False
		self.quiet_on 	= False

		# sim param 
		self.n_trials = 1
		self.sim_t0 = 0
		self.sim_tf = 20
		self.sim_dt = 0.25
		
		# topology
		self.r_sense = 1.6
		self.r_comm = 1.6
		
		# environment
		large_env = False 
		if large_env:
			self.env_xlim = [0,1]
			self.env_ylim = [0,1]
			self.reset_xlim_A = [0,0.2]
			self.reset_ylim_A = [0,1]
			self.reset_xlim_B = [0.8,1]
			self.reset_ylim_B = [0,1]
			self.goal_line_x = 0.6
			self.goal = np.array([0.5,0.5])
		else: 
			l = 0.5
			self.env_xlim = [0,l]
			self.env_ylim = [0,l]
			self.reset_xlim_A = [0.1*l,0.1*l]
			self.reset_ylim_A = [0.5*l,0.5*l]
			self.reset_xlim_B = [0.9*l,0.9*l]
			self.reset_ylim_B = [0.5*l,0.5*l]
			self.goal_line_x = 0.6
			self.goal = np.array([0.5*l,0.75*l])

		# mcts parameters 
		self.tree_size = 500
		self.rollout_horizon = 100
		self.c_param = 1.4
		self.gamma = 0.9
		
		# nodes 
		self.num_nodes_A = 1
		self.num_nodes_B = 1
		
		# estimator parameters
		self.initial_state_covariance = 1e-10 # defines initial condition of estimators
		
		# dynamics parameters
		self.process_noise_covariance = 1e-10
		
		# measurement parameters
		self.measurement_noise_covariance = 1e-10

		# policy 
		self.rhc_horizon = 5
		self.lambda_u = 0.01
		self.speed_limit_a = 0.5 # 0.05 
		self.speed_limit_b = 0*1.0 # 0.1 
		self.acceleration_limit_a = 0.10
		self.acceleration_limit_b = 0.20
		self.danger_radius = 0.1
		self.tag_radius = 0.025

		# path stuff
		self.current_results_dir = '../current_results'
		self.glas_model = '../models/il_save.pt'
		
		# plotting 
		self.plot_fn = 'plots.pdf'
		
		# save stuff 
		self.info_keys = [
			'state_vec',
			'node_idx',
			'node_state_mean',
			'node_state_covariance',
			'node_state',
			'node_team_A',
			'node_team_B',
		]

		self.update()


	def to_dict(self):
		return self.__dict__


	def from_dict(self,some_dict):
		for key,value in some_dict.items():
			setattr(self,key,value)


	def update(self):
		self.num_nodes = self.num_nodes_A + self.num_nodes_B
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)

		self.team_1_idxs = []
		self.team_2_idxs = []
		for i in range(self.num_nodes):
			if i < self.num_nodes_A: 
				self.team_1_idxs.append(i) 
			else:
				self.team_2_idxs.append(i) 
