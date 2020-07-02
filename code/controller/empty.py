import numpy as np 
from controller.controller import Controller
	
class Controller(Controller):

	def __init__(self,param,env):
		super(Controller, self).__init__(param,env)

	def policy(self,estimate):
		actions = dict() 
		for node, estimate_i in estimate.items():
			actions[node] = np.zeros((node.dynamics.control_dim_per_agent,1))
		return actions