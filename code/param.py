
import numpy as np 
import itertools, copy

class Param:

	def __init__(self):

		# modules 
		self.dynamics_name 		= 'dynamics/double_integrator.py' 	# 
		self.measurements_name 	= 'measurements/global.py' 			# global, (local)
		self.estimator_name 	= 'estimator/kalman.py'	 			# empty,(kalman),exact...
		self.attacker_name 		= 'attacker/empty.py' 				# empty, ...
		self.controller_name 	= 'controller/glas.py'			 	# empty, glas, joint_mpc, mcts, ...

		# flags
		self.gif_on 	= False
		self.quiet_on 	= False

		# sim param 
		self.n_trials = 1
		self.sim_t0 = 0
		self.sim_tf = 20
		self.sim_dt = 0.25

		# robot types 
		self.standard_robot = {
			'speed_limit': 0.125,
			'acceleration_limit':0.125,
			'tag_radius': 0.025,
			'dynamics':'double_integrator',
			'r_comm': 0.4,
			'r_sense': 0.4,
		}

		self.evasive_robot = {
			'speed_limit': 0.125,
			'acceleration_limit':0.2,
			'tag_radius': 0.025,
			'dynamics':'double_integrator',
			'r_comm': 0.4,
			'r_sense': 0.4,
		}

		self.robot_teams = {
			'a': {'standard_robot':1,'evasive_robot':0},
			'b': {'standard_robot':1,'evasive_robot':0}
		}
		
		# environment
		l = 1.0 
		self.env_xlim = [0,l]
		self.env_ylim = [0,l]
		self.reset_xlim_A = [0.1*l,0.2*l]
		self.reset_ylim_A = [0.1*l,0.9*l]
		self.reset_xlim_B = [0.8*l,0.9*l]
		self.reset_ylim_B = [0.1*l,0.9*l]
		self.goal = np.array([0.75*l,0.75*l])

		# mcts parameters 
		self.tree_size = 100000
		self.fixed_tree_depth_on = False
		self.fixed_tree_depth = 100
		self.rollout_horizon = 1000
		self.c_param = 1.4
		self.gamma = 1.0
		
		# estimator parameters
		self.initial_state_covariance = 1e-10 # defines initial condition of estimators
		
		# dynamics parameters
		self.process_noise_covariance = 1e-10
		
		# measurement parameters
		self.measurement_noise_covariance = 1e-10

		# MPC policy 
		self.rhc_horizon = 5
		self.lambda_u = 0.01
		self.danger_radius = 0.1

		# path stuff
		self.current_results_dir = '../current_results'
		self.glas_model_A = '../models/il_current_a.pt'
		self.glas_model_B = '../models/il_current_b.pt'
		
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

		self.robots = [] 
		for team, composition in self.robot_teams.items():
			for robot_type, robot_number in composition.items():
				for _ in range(robot_number):
					robot = copy.copy(self.__dict__[robot_type])
					robot["team"] = team 
					self.robots.append(robot)

		num_nodes_A, num_nodes_B = 0,0
		for robot in self.robots:
			if robot["team"] == 'a':
				num_nodes_A += 1
			elif robot["team"] == 'b':
				num_nodes_B += 1

		self.num_nodes_A = num_nodes_A
		self.num_nodes_B = num_nodes_B


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

		self.actions = np.asarray(list(itertools.product(*[[-1,0,1],[-1,0,1]])))
