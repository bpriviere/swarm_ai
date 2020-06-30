
import numpy as np 

# helper classes 
class Node:

	def __init__(self,node_dict):
		for key,value in node_dict.items():
			setattr(self,key,value)

	def dist_to_node(self,node):
		return ((self.p_x - node.p_x)**2. + (self.p_y - node.p_y)**2.)**(1/2)

	def measure(self,full_state):
		return self.measurements.measure_per_node(full_state)

	def forward(self,control):
		self.state = self.dynamics.forward_per_node(self.state,control)
		 


class Dynamics:

	def __init__(self,param):
		self.param = param 
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

	def forward_per_node(self,x_i,u_i):

		# noise 
		Q = self.param.process_noise_covariance * np.eye(self.state_dim_per_agent)
		w = np.dot(Q,np.random.normal(size=((self.state_dim_per_agent,1))))
		
		# solve 
		x_tp1 = np.dot(self.A,x_i) + np.dot(self.B,u_i) # + w 

		return x_tp1 


class Measurements:

	def __init__(self,param):
		self.param = param 
		# self.param.measurement_dim_per_agent = self.param.state_dim
		# self.param.measurement_dim = param.num_nodes * param.measurement_dim_per_agent


	def measure_per_node(self,x):

		state_dim = x.shape[0]
		measurement_dim = state_dim

		C_i = np.eye((state_dim))

		# noise 
		R_i = self.param.measurement_noise_covariance * np.eye(measurement_dim)
		v_i = np.dot(R_i,np.random.normal(size=((measurement_dim,1))))
		
		# solve 
		y_i = np.dot(C_i,x) + v_i

		return y_i


			
