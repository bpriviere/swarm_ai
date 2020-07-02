
import numpy as np 
from dynamics.dynamics import Dynamics
	
class Dynamics(Dynamics):

	def __init__(self,param):
		super(Dynamics, self).__init__(param)

		self.state_dim_per_agent = 4
		self.control_dim_per_agent = 2 

		A = np.array((
			[0,0,1,0],
			[0,0,0,1],
			[0,0,0,0],
			[0,0,0,0])) 
		self.A = np.eye(self.state_dim_per_agent) + self.param.sim_dt*A 

		B = np.array((
			[0,0],
			[0,0],
			[1,0],
			[0,1],
			))
		self.B = self.param.sim_dt*B

		self.Q = self.param.process_noise_covariance * np.eye(self.state_dim_per_agent)

	def forward_per_node(self,x_i,u_i):

		# noise 
		w = np.dot(self.Q,np.random.normal(size=((self.state_dim_per_agent,1))))
		
		# solve 
		x_tp1 = np.dot(self.A,x_i) + np.dot(self.B,u_i) # + w 

		return x_tp1 