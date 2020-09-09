
import numpy as np
import itertools, copy
import random
import os,sys
from math import cos, sin

# todo: make rel-path to abs-path func 

class Param:

	def __init__(self,seed=None):

		if seed is None: 
			seed = int.from_bytes(os.urandom(4), sys.byteorder)

		self.seed = seed
		# random.seed(self.seed)

		# sim param 
		self.sim_num_trials = 2
		self.sim_t0 = 0
		self.sim_tf = 20
		self.sim_dt = 0.25
		self.sim_parallel_on = True
		self.sim_mode = "GLAS" # MCTS_GLAS, MCTS_RANDOM, GLAS, PANAGOU (not implemented)

		# robot types 
		self.standard_robot = {
			'speed_limit': 0.125,
			'acceleration_limit':0.125,
			'tag_radius': 0.025,
			'dynamics':'double_integrator',
			'r_sense': 1.0,
			'radius': 0.025,
		}

		self.evasive_robot = {
			'speed_limit': 0.125,
			'acceleration_limit':0.2,
			'tag_radius': 0.025,
			'dynamics':'double_integrator',
			'r_sense': 1.0,
			'radius': 0.025,
		}

		self.robot_team_composition = {
			'a': {'standard_robot':1,'evasive_robot':0},
			'b': {'standard_robot':1,'evasive_robot':0}
		}
		
		# environment
		l = 0.5 
		self.env_xlim = [0,l]
		self.env_ylim = [0,l]
		self.reset_xlim_A = [0.1*l,0.9*l]
		self.reset_ylim_A = [0.1*l,0.9*l]
		self.reset_xlim_B = [0.1*l,0.9*l]
		self.reset_ylim_B = [0.1*l,0.9*l]
		self.goal = np.array([0.75*l,0.75*l,0,0])

		# mcts parameters 
		self.mcts_tree_size = 10000
		self.mcts_rollout_horizon = 1000
		self.mcts_rollout_beta = 0.0 # 0 -> 1 : random -> GLAS
		self.mcts_c_param = 1.4

		# learning (l) parameters 
		self.device = 'cpu'
		self.l_mode = "IL" # IL, DAgger, ExIt, Mice # so far only IL is implemented 
		self.l_parallel_on = True # set to false only for debug 
		self.l_num_iterations = 5
		self.l_num_file_per_iteration = 2 # optimized for num cpu on ben's laptop 
		self.l_num_points_per_file = 20
		self.l_training_teams = ["a","b"]
		self.l_robot_team_composition_cases = [
			{
			'a': {'standard_robot':1,'evasive_robot':0},
			'b': {'standard_robot':1,'evasive_robot':0}
			},
			{
			'a': {'standard_robot':2,'evasive_robot':0},
			'b': {'standard_robot':1,'evasive_robot':0}
			},
			{
			'a': {'standard_robot':1,'evasive_robot':0},
			'b': {'standard_robot':2,'evasive_robot':0}
			},
			# {
			# 'a': {'standard_robot':2,'evasive_robot':0},
			# 'b': {'standard_robot':2,'evasive_robot':0}
			# },			
		]

		n,m,h,l,p = 4,2,16,8,8 # state dim, action dim, hidden layer, output phi, output rho
		self.l_phi_network_architecture = [
			["Linear", n, h],
			["Linear", h, h],
			["Linear", h, l]
		]

		self.l_rho_network_architecture = [
			["Linear", l, h],
			["Linear", h, h],
			["Linear", h, p]
		]

		self.l_psi_network_architecture = [
			["Linear", 2*p+n, h],
			["Linear", h, h],
			["Linear", h, 9]
		]

		self.l_network_activation = "relu"
		self.l_test_train_ratio = 0.8
		self.l_max_dataset_size = 1000000 # n_points 
		self.l_batch_size = 20
		self.l_n_epoch = 10
		self.l_lr = 1e-3
		self.l_wd = 0 
		self.l_log_interval = 1
		self.l_raw_fn = '{DATADIR}raw_{TEAM}train_{NUM_A}a_{NUM_B}b_{IDX_TRIAL}trial'
		self.l_labelled_fn = '{DATADIR}labelled_{TEAM}train_{NUM_A}a_{NUM_B}b_{IDX_TRIAL}trial.npy'
		self.l_model_fn = '{DATADIR}{TEAM}{ITER}.pt'

		# path stuff
		self.path_current_results = '../current/results/'
		self.path_current_models = '../current/models/'
		self.path_current_data = '../current/data/'
		self.path_glas_model_a = os.path.join(self.path_current_models,'a0.pt')
		self.path_glas_model_b = os.path.join(self.path_current_models,'b0.pt')
		
		self.update()


	def to_dict(self):
		return self.__dict__


	def from_dict(self,some_dict):
		for key,value in some_dict.items():
			setattr(self,key,value)

	def make_initial_condition(self):

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
			for robot_type, robot_number in composition.items():
				for _ in range(robot_number):
					robot = copy.copy(self.__dict__[robot_type])
					robot["team"] = team 
					self.robots.append(robot)		


	def update(self,initial_condition=None):

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

		# times 
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)

		# actions 
		self.actions = np.asarray(list(itertools.product(*[[-1,0,1],[-1,0,1]])))


	def get_random_position_inside(self,xlim,ylim):

		x = random.random()*(xlim[1] - xlim[0]) + xlim[0]
		y = random.random()*(ylim[1] - ylim[0]) + ylim[0]
		
		return x,y 				

	def get_random_velocity_inside(self,speed_lim):

		th = random.random()*2*np.pi 
		r  = random.random()*speed_lim

		return r*cos(th), r*sin(th)	