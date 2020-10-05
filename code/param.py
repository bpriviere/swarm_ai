
import numpy as np
import itertools, copy
import random
import os,sys
from math import cos, sin, sqrt

# todo: make rel-path to abs-path func 

class Param:

	def __init__(self):

		# sim param 
		self.sim_num_trials = 10
		self.sim_dt = 0.25
		self.sim_parallel_on = True

		# these parameters are also used for learning 
		self.policy_dict = {
			'sim_mode' : 				"MCTS", # "MCTS, D_MCTS, RANDOM, PANAGOU, GLAS"
			'path_glas_model_a' : 		None, 	#'../current/models/a0.pt', 
			'path_glas_model_b' : 		None, 	#'../current/models/b0.pt', 
			'mcts_tree_size' : 			100000,
			'mcts_rollout_beta' : 		0.0,
			'mcts_c_param' : 			1.4,
			'mcts_pw_C' : 				1.0,
			'mcts_pw_alpha' : 			0.25,
			'mcts_vf_beta' : 			0.0,
		}

		# max timesteps until the game terminates
		self.rollout_horizon = 100

		# robot types 
		self.robot_types = {
			'standard_robot' : {
				'speed_limit': 0.125,
				'acceleration_limit':0.125,
				'tag_radius': 0.025,
				'dynamics':'double_integrator',
				'r_sense': 0.2,
				'radius': 0.025,
			},
			'evasive_robot' : {
				'speed_limit': 0.125,
				'acceleration_limit':0.2,
				'tag_radius': 0.025,
				'dynamics':'double_integrator',
				'r_sense': 0.2,
				'radius': 0.025,
			}
		}

		self.robot_team_composition = {
			'a': {'standard_robot':1,'evasive_robot':0},
			'b': {'standard_robot':1,'evasive_robot':0}
		}
		
		# environment
		self.env_l = 0.5

		# learning (l) parameters 
		self.device = 'cpu' # cpu, cuda
		self.l_mode = "DAgger" # IL, DAgger, ExIt, MICE
		self.num_cpus = 4 # if device is 'cpu' use up to num_cpus for DistributedDataParallel (None to disable DDP)
		self.l_sync_every = 4 # synchronize after l_sync_every batches in multi-cpu mode
		self.l_parallel_on = True # set to false only for debug 
		self.l_num_iterations = 5
		self.l_num_file_per_iteration = 20 # optimized for num cpu on ben's laptop 
		self.l_num_points_per_file = 1000
		self.l_mcts_rollout_beta = 0.25
		self.l_num_learner_nodes = 5000
		self.l_num_expert_nodes = 100000
		self.l_env_l0 = 0.25
		self.l_env_dl = 0.25
		self.l_training_teams = ["a","b"]
		self.l_robot_team_composition_cases = [
			{
			'a': {'standard_robot':1,'evasive_robot':0},
			'b': {'standard_robot':1,'evasive_robot':0}
			},
			# {
			# 'a': {'standard_robot':2,'evasive_robot':0},
			# 'b': {'standard_robot':1,'evasive_robot':0}
			# },
			# {
			# 'a': {'standard_robot':1,'evasive_robot':0},
			# 'b': {'standard_robot':2,'evasive_robot':0}
			# },
			# {
			# 'a': {'standard_robot':2,'evasive_robot':0},
			# 'b': {'standard_robot':2,'evasive_robot':0}
			# },			
		]

		self.l_subsample_on = False
		self.l_num_subsamples = 5

		self.l_num_samples = 5 # take l_num_samples-best samples from mctsresult (still weighted)

		self.l_state_dim = 4 
		self.l_action_dim = 2 
		self.l_z_dim = 4
		self.l_hidden_dim = 16

		n,m,h,z = self.l_state_dim,self.l_action_dim,self.l_hidden_dim,self.l_z_dim

		self.l_phi_network_architecture = [
			["Linear", n, h],
			["Linear", h, h],
			["Linear", h, h]
		]

		self.l_rho_network_architecture = [
			["Linear", h, h],
			["Linear", h, h],
			["Linear", h, h]
		]

		self.l_conditional_network_architecture = [
			["Linear", 2*h+n, h], 
			["Linear", h, h],
			["Linear", h, h] 
		]

		self.l_encoder_network_architecture = [
			["Linear", m+h, h],
			["Linear", h, h],
			["Linear", h, h],
			["Linear", h, 2*z] 
		]

		self.l_decoder_network_architecture = [
			["Linear", z+h, h], 
			["Linear", h, h],
			["Linear", h, h],
			["Linear", h, m] 
		]

		self.l_value_network_architecture = [
			["Linear", h, h], 
			["Linear", h, h],
			["Linear", h, 1] 
		]


		self.l_network_activation = "relu"
		self.l_test_train_ratio = 0.8
		self.l_max_dataset_size = 10000000000 # n_points 
		self.l_batch_size = 512
		self.l_n_epoch = 1000
		self.l_lr = 1e-3
		self.l_lr_scheduler = None # one of None, 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'
		self.l_wd = 0 
		self.l_log_interval = 1
		self.l_raw_fn = '{DATADIR}raw_team{TEAM}_i{LEARNING_ITER}_numfn{NUM_FILE}'
		self.l_labelled_fn = '{DATADIR}labelled_team{TEAM}_i{LEARNING_ITER}_numa{NUM_A}_numb{NUM_B}_numfn{NUM_FILE}.npy'
		self.l_model_fn = '{DATADIR}{TEAM}{ITER}.pt'

		# path stuff
		self.path_current_results = '../current/results/'
		self.path_current_models = '../current/models/'
		self.path_current_data = '../current/data/'
		
		self.update()


	def to_dict(self):
		return self.__dict__


	def from_dict(self,some_dict):
		for key,value in some_dict.items():
			setattr(self,key,value)

	def make_environment(self):
		self.env_xlim = [0,self.env_l]
		self.env_ylim = [0,self.env_l]
		self.reset_xlim_A = [0.1*self.env_l,0.9*self.env_l]
		self.reset_xlim_B = [0.1*self.env_l,0.9*self.env_l]
		# self.reset_xlim_A = [0.1*self.env_l,0.2*self.env_l]
		# self.reset_xlim_B = [0.8*self.env_l,0.9*self.env_l]
		self.reset_ylim_A = [0.1*self.env_l,0.9*self.env_l]
		self.reset_ylim_B = [0.1*self.env_l,0.9*self.env_l]
		self.goal = np.array([0.75*self.env_l,0.5*self.env_l,0,0])

	def make_initial_condition(self):

		# randomly change enviornment
		# alpha = np.random.randint(4)
		# if alpha == 0:
		# 	# do nothing 
		# 	reset_xlim_A = self.reset_xlim_A
		# 	reset_xlim_B = self.reset_xlim_B
		# 	reset_ylim_A = self.reset_ylim_A
		# 	reset_ylim_B = self.reset_ylim_B
		# 	goal = self.goal

		# if alpha == 1 or alpha == 3: 
		# 	# flip on x = 0.5 l 
		# 	reset_xlim_A = self.reset_xlim_B
		# 	reset_xlim_B = self.reset_xlim_A 
		# 	reset_ylim_A = self.reset_ylim_B
		# 	reset_ylim_B = self.reset_ylim_A 
		# 	goal = self.goal
		# 	goal[0] = self.env_xlim[1] - self.goal[0]
		# if alpha == 2 or alpha == 3:
		# 	# flip on y = x 
		# 	reset_xlim_A = self.reset_ylim_A
		# 	reset_ylim_A = self.reset_xlim_A
		# 	reset_xlim_B = self.reset_ylim_B 
		# 	reset_ylim_B = self.reset_xlim_B 
		# 	goal = self.goal
		# 	goal[1] = self.goal[0]
		# 	goal[0] = self.goal[1]

		state = [] 
		positions = [] 
		velocities = [] 
		radii = [] 
		for robot in self.robots: 

			if robot["team"] == "a":
				xlim = self.reset_xlim_A
				ylim = self.reset_ylim_A
			elif robot["team"] == "b":
				xlim = self.reset_xlim_B
				ylim = self.reset_ylim_B

			count = 0 
			position = self.get_random_position_inside(xlim,ylim)
			velocity = self.get_random_velocity_inside(robot["speed_limit"])
			while self.collision(position,velocity,robot["radius"],positions,velocities,radii):
				position = self.get_random_position_inside(xlim,ylim)
				count += 1 
				if count > 10000:
					exit('infeasible initial condition')

			radii.append(robot["radius"])
			positions.append(np.array((position[0],position[1])))
			velocities.append(np.array((velocity[0],velocity[1])))
			state.append([position[0],position[1],velocity[0],velocity[1]])

		return state

	def collision(self,p1,v1,r1,p2s,v2s,r2s):
		p1_tp1 = np.array(p1) + self.sim_dt * np.array(v1)
		for p2, v2, r2 in zip(p2s,v2s,r2s):
			# check initial condition 
			if np.linalg.norm(np.array(p1)-p2) < (r1 + r2): 
				return True 
			# check next state 
			p2_tp1 = p2 + self.sim_dt * v2
			if np.linalg.norm(p1_tp1-p2_tp1) < (r1 + r2): 
				return True 
		return False 


	def assign_initial_condition(self):

		for robot, x0 in zip(self.robots,self.state): 
			robot["x0"] = x0


	def make_robot_teams(self):

		# make robot teams 
		self.robots = [] 
		for team, composition in self.robot_team_composition.items():
			for robot_type_name, robot_number in composition.items():
				for _ in range(robot_number):
					robot = copy.copy(self.robot_types[robot_type_name])
					robot["team"] = team 
					self.robots.append(robot)		


	def update(self,initial_condition=None):

		self.make_environment()

		self.make_robot_teams()

		if initial_condition is None:
			initial_condition = self.make_initial_condition()
		self.state = initial_condition

		self.assign_initial_condition()		

		num_nodes_A, num_nodes_B = 0,0
		for robot in self.robots:
			if robot["team"] == 'a':
				num_nodes_A += 1
			elif robot["team"] == 'b':
				num_nodes_B += 1

		self.num_nodes_A = num_nodes_A
		self.num_nodes_B = num_nodes_B
		self.num_nodes = self.num_nodes_A + self.num_nodes_B

		self.team_1_idxs = []
		self.team_2_idxs = []
		for i in range(self.num_nodes):
			if i < self.num_nodes_A: 
				self.team_1_idxs.append(i) 
			else:
				self.team_2_idxs.append(i) 

		# actions 
		self.actions = np.asarray(list(itertools.product(*[[-1,0,1],[-1,0,1]])))


	def get_random_position_inside(self,xlim,ylim):

		x = random.random()*(xlim[1] - xlim[0]) + xlim[0]
		y = random.random()*(ylim[1] - ylim[0]) + ylim[0]
		
		return x,y 				

	def get_random_velocity_inside(self,speed_lim):

		th = random.random()*2*np.pi 
		# r  = sqrt(random.random())*speed_lim
		r  = 0*sqrt(random.random())*speed_lim
		return r*cos(th), r*sin(th)	