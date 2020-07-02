

import numpy as np 
import cvxpy as cp 
from scipy.optimize import linear_sum_assignment

class Controller: 

	def __init__(self,param,env):
		self.env = env
		self.param = param 


	def policy(self,estimate):
		exit('error: controller needs to overwrite policy func')

	def initial_policy(self):
		actions = dict()
		for node in self.env.nodes: 
			actions[node] = np.zeros((node.dynamics.control_dim_per_agent,1))
		return actions 