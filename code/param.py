
import numpy as np
import itertools, copy
import random
import os,sys
from math import cos, sin, sqrt

# todo: make rel-path to abs-path func 


# for dynamics: 
double_integrator = {
	"name" : "double_integrator",
	"state_dim" : 4, # per robot 
	"control_dim" : 2, 
	"state_labels" : ["x","y","vx","vy"],
	"control_labels" : ["ax","ay"]
}
single_integrator = {
	"name" : "single_integrator",
	"state_dim" : 2, # per robot 
	"control_dim" : 2, 
	"state_labels" : ["x","y"],
	"control_labels" : ["vx","vy"]
}
dubins_2d = {
	"name" : "dubins_2d",
	"state_dim" : 4, # per robot 
	"control_dim" : 2, 
	"state_labels" : ["x","y","th","speed"],
	"control_labels" : ["acc","omega"]
}
dubins_3d = {
	"name" : "dubins_3d",
	"state_dim" : 5, # per robot 
	"control_dim" : 2, 
	"state_labels" : ["x","y","z","phi","psi"],
	"control_labels" : ["phidot","psidot"]
}


class Param:

	def __init__(self):

		# sim param 
		self.sim_num_trials = 6
		self.sim_dt = 0.1
		self.sim_parallel_on = True

		# these parameters are also used for learning 
		self.policy_dict = {
			'sim_mode' : 				"MCTS", # "MCTS, D_MCTS, RANDOM, PANAGOU, GLAS"
			'path_glas_model_a' : 		None, 	# '../current/models/a1.pt', None
			'path_glas_model_b' : 		None, 	# '../current/models/b1.pt', None
			'path_value_fnc' : 			None, 	# '../current/models/v1.pt', None		
			'mcts_tree_size' : 			10000,
			'mcts_c_param' : 			2.0,
			'mcts_pw_C' : 				1.0,
			'mcts_pw_alpha' : 			0.25,
			'mcts_beta1' : 				0.0,
			'mcts_beta2' : 				0.5,
			'mcts_beta3' : 				0.0,
		}

		self.dynamics = double_integrator # "single_integrator", "double_integrator", "dubins_2d"

		# robot types 
		self.robot_types = {
			'standard_robot' : {
				'speed_limit': 1.0,
				'acceleration_limit':2.0,
				'tag_radius': 0.10,
				'dynamics':'{}'.format(self.dynamics["name"]),
				'r_sense': 3.0,
				'radius': 0.05,
			},
			'evasive_robot' : {
				'speed_limit': 0.0625,
				'acceleration_limit':0.5,
				'tag_radius': 0.0125,
				'dynamics':'{}'.format(self.dynamics["name"]),
				'r_sense': 0.5,
				'radius': 0.025,
			}
		}

		self.robot_team_composition = {
			'a': {'standard_robot':2,'evasive_robot':0},
			# 'a': {'standard_robot':2,'evasive_robot':0},
			'b': {'standard_robot':1,'evasive_robot':0}
		}
		
		# environment
		self.env_l = 2.0

		# learning (l) parameters 
		self.device = 'cuda' # 'cpu', 'cuda'
		self.l_mode = "MICE" # IL, DAgger, ExIt, MICE
		self.num_cpus = 4 # if device is 'cpu' use up to num_cpus for DistributedDataParallel (None to disable DDP)
		self.l_sync_every = 4 # synchronize after l_sync_every batches in multi-cpu mode
		self.l_parallel_on = True # set to false only for debug 
		self.l_num_iterations = 5
		self.l_num_file_per_iteration = 20 
		self.l_num_points_per_file = 2500
		self.l_mcts_c_param = 2.0
		self.l_mcts_pw_C = 1.0
		self.l_mcts_pw_alpha = 0.25
		self.l_mcts_beta1 = 0.0
		self.l_mcts_beta2 = 0.5
		self.l_mcts_beta3 = 0.5
		self.l_num_learner_nodes = 500
		self.l_num_expert_nodes = 5000
		self.l_env_dl = 1.0
		self.l_warmstart = True # warmstart policies between iterations
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

		self.l_desired_game = {
			'Skill_A' : 4, #'a1.pt',
			'Skill_B' : 4, #'b1.pt',
			'EnvironmentLength' : 3.0,
			'NumA' : 3,
			'NumB' : 3,
		}

		self.l_subsample_on = False
		self.l_num_subsamples = 5

		self.l_num_samples = 5 # take l_num_samples-best samples from mctsresult (still weighted)

		self.l_state_dim = self.dynamics["state_dim"]
		self.l_action_dim = self.dynamics["control_dim"] 
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

		self.l_xi_network_architecture = [
			["Linear", 2*h + 3, h], 
			["Linear", h, h],
			["Linear", h, 2] 
		]		

		self.l_policy_network_architecture = [
			["Linear", 2*h+n, h], 
			["Linear", h, h],
			["Linear", h, 2*m] 		
		]

		self.l_gaussian_on = True
		self.l_glas_rollout_on = True

		self.l_network_activation = "relu"
		self.l_test_train_ratio = 0.8
		self.l_max_dataset_size = 10000000000 # n_points 
		self.l_batch_size = 4096 #512
		self.l_n_epoch = 500
		self.l_lr = 1e-3
		self.l_lr_scheduler = None #'ReduceLROnPlateau' # one of None, 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'
		self.l_wd = 0 
		self.l_log_interval = 1
		self.l_raw_fn = '{DATADIR}raw_team{TEAM}_i{LEARNING_ITER}_numfn{NUM_FILE}'
		self.l_raw_value_fn = '{DATADIR}raw_value_i{LEARNING_ITER}_numfn{NUM_FILE}'
		self.l_labelled_fn = '{DATADIR}labelled_team{TEAM}_i{LEARNING_ITER}_numa{NUM_A}_numb{NUM_B}_numfn{NUM_FILE}.npy'
		self.l_labelled_value_fn = '{DATADIR}labelled_value_i{LEARNING_ITER}_numa{NUM_A}_numb{NUM_B}_numfn{NUM_FILE}.npy'
		self.l_model_fn = '{DATADIR}{TEAM}{ITER}.pt'
		self.l_value_model_fn = '{DATADIR}v{ITER}.pt'

		# path stuff
		self.path_current_results = '../current/results/'
		self.path_current_models = '../current/models/'
		self.path_current_data = '../current/data/'

		self.tree_timestep = 10
		self.plot_tree_on = False

		self.init_on_sides = True
		
		self.update()


	def to_dict(self):
		return self.__dict__


	def from_dict(self,some_dict):
		for key,value in some_dict.items():
			setattr(self,key,value)

	def make_environment(self):
		self.env_xlim = [0,self.env_l]
		self.env_ylim = [0,self.env_l]

		if self.init_on_sides: 
			self.reset_xlim_A = [0.1*self.env_l,0.2*self.env_l]
			self.reset_xlim_B = [0.8*self.env_l,0.9*self.env_l]
		else: 
			self.reset_xlim_A = [0.1*self.env_l,0.9*self.env_l]
			self.reset_xlim_B = [0.1*self.env_l,0.9*self.env_l]

		self.reset_ylim_A = [0.1*self.env_l,0.9*self.env_l]
		self.reset_ylim_B = [0.1*self.env_l,0.9*self.env_l]

		self.goal = np.array([0.7*self.env_l,0.5*self.env_l,0,0])

	def make_initial_condition(self):

		state_dim = self.dynamics["state_dim"]
		name = self.dynamics["name"]
		state = np.nan*np.ones((len(self.robots),state_dim))

		for robot in self.robots: 

			if robot["team"] == "a":
				xlim = self.reset_xlim_A
				ylim = self.reset_ylim_A
			elif robot["team"] == "b":
				xlim = self.reset_xlim_B
				ylim = self.reset_ylim_B

			if name in ["single_integrator","double_integrator","dubins_2d"]:
				radius = robot["radius"]
				state_space = np.array((xlim,ylim))
				position = self.get_random_position_inside(state_space)
				count = 0 
				while collision(position,robot,state[:,0:2],self.robots):
					position = self.get_random_position_inside(state_space)
					count += 1 
					if count > 10000:
						exit('infeasible initial condition')

			if name in ["double_integrator","dubins_2d"]:
				velocity = self.get_random_velocity_inside(robot["speed_limit"])
			if name in ["dubins_2d"]:
				orientation = random.random()*2*np.pi 

			if name == "double_integrator":
				state[robot["idx"],0:2] = position
				state[robot["idx"],2:4] = velocity

			if name == "single_integrator":
				state[robot["idx"],0:2] = position

			if name == "dubins_2d":
				state[robot["idx"],0:2] = position 
				state[robot["idx"],2] = orientation
				state[robot["idx"],3] = np.linalg.norm(velocity)

			if name == "dubins_3d":
				state_space = np.array((xlim,ylim,ylim,(0,2*np.pi),(0,2*np.pi)))
				state[robot["idx"],:] = self.get_random_position_inside(state_space)

		return state.tolist() 

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
					robot["idx"] = len(self.robots)
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

		# max timesteps until the game terminates
		# self.rollout_horizon = int(100 * self.env_l)
		num_backnforth = 2
		self.rollout_horizon = int(num_backnforth * self.num_nodes * self.env_l \
			/ (self.robot_types["standard_robot"]["speed_limit"] * self.sim_dt))


	# def get_random_position_inside(self,xlim,ylim):

	# 	x = random.random()*(xlim[1] - xlim[0]) + xlim[0]
	# 	y = random.random()*(ylim[1] - ylim[0]) + ylim[0]
		
	# 	return x,y 	


	def get_random_position_inside(self,statespace):

		state_dim, _ = statespace.shape
		position = np.zeros(state_dim)

		for i_state in range(state_dim):
			position[i_state] = statespace[i_state,0] + random.random()*\
				(statespace[i_state,1] - statespace[i_state,0])
		
		return position


	def get_random_velocity_inside(self,speed_lim):

		th = random.random()*2*np.pi 
		# r  = sqrt(random.random())*speed_lim
		r  = 0*sqrt(random.random())*speed_lim
		return r*cos(th), r*sin(th)	

def collision(pose_i,robot_i,poses,robots):
	for robot_j, pose_j in zip(robots,poses):
		if robot_j is not robot_i and not np.isnan(pose_j).any():
			dist = np.linalg.norm(pose_i - pose_j)
			if dist < robot_i["radius"] + robot_j["radius"]:
				return True 
	return False


