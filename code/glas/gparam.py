

from torch import nn, tanh, relu

class Gparam: 

	def __init__(self):

		# flags  
		self.make_raw_data_on 		= True
		self.make_labelled_data_on 	= False
		self.dbg_vis_on			 	= False
		self.train_model_on 		= True

		# raw data param  
		self.serial_on 				= True

		# generate demonstration data parameters
		self.num_nodes_A_lst = [2] # [2,4,6]
		self.num_nodes_B_lst = [1] # [2,4,6]
		self.num_trials = 1
		self.demonstration_data_dir = '../../data/demonstration/'
		self.model_dir = '../../models/'

		# train parameters
		self.training_team = "b"

		# learning hyperparameters
		self.device = 'cpu'

		n,m,h,l,p = 4,2,32,8,8 # state dim, action dim, hidden layer, output phi, output rho
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(n,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_rho_network_architecture = nn.ModuleList([
			nn.Linear(l,h),
			nn.Linear(h,h),
			nn.Linear(h,p)])

		self.il_psi_network_architecture = nn.ModuleList([
			nn.Linear(2*p,h), # because two deepsets 
			nn.Linear(h,h),
			nn.Linear(h,m)])

		self.il_network_activation = relu
		self.il_train_model_fn = self.model_dir + 'il_current.pt'
		self.il_test_train_ratio = 0.8
		self.il_n_points = 1000000
		self.il_batch_size = 5000
		self.il_n_epoch = 100
		self.il_lr = 1e-3
		self.il_wd = 0 
		self.il_log_interval = 1
