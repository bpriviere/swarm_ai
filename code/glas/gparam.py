
import itertools
import numpy as np 
from torch import nn, tanh, relu

class Gparam: 

	def __init__(self):

		# flags  
		self.make_raw_data_on 		= True
		self.make_labelled_data_on 	= False
		self.train_model_on 		= False
		self.dbg_vis_on			 	= False

		self.learning_module = 'learning/discrete_emptynet.py'

		self.serial_on 				= True # set true only for dbging 
		self.clean_raw_data_on 		= True
		self.clean_labelled_data_on = True

		# generate demonstration data parameters
		self.robot_team_composition_cases = [
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
			}
		]
		self.num_trials = 10 
		self.num_points_per_file = 100
		self.mode = "MCTS_RANDOM" # one of "GLAS", "MCTS_RANDOM", "MCTS_GLAS"
		self.num_nodes = 10000 # in mcts tree 
		self.rollout_beta = 0.0 
		self.demonstration_data_dir = '../../data/demonstration/'
		self.model_dir = '../../models/'

		# train parameters
		self.training_teams = ["a","b"] #,"b"] #["b"] #["a","b"]

		# learning hyperparameters
		self.device = 'cpu'

		n,m,h,l,p = 4,2,16,8,8 # state dim, action dim, hidden layer, output phi, output rho
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(n,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_rho_network_architecture = nn.ModuleList([
			nn.Linear(l,h),
			nn.Linear(h,h),
			nn.Linear(h,p)])

		self.il_psi_network_architecture = nn.ModuleList([
			nn.Linear(2*p+n,h), # because two deepsets 
			nn.Linear(h,h),
			nn.Linear(h,9)])

		self.il_network_activation = relu
		self.il_train_model_fn = self.model_dir + 'il_current_{}.pt'
		self.il_test_train_ratio = 0.8
		self.il_n_points = 1000000
		self.il_batch_size = 2000
		self.il_n_epoch = 500
		self.il_lr = 1e-3
		self.il_wd = 0 
		self.il_log_interval = 1

		self.update()


	def update(self):
		self.actions = np.asarray(list(itertools.product(*[[-1,0,1],[-1,0,1]])))
