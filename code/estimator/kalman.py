
import numpy as np 
from estimator.estimator import Estimator, Estimate

class Estimate(Estimate):

	def __init__(self,state_mean,state_covariance):
		super(Estimate, self).__init__(state_mean, state_covariance):

	def update_node(self,node):
		node.state_mean = self.state_mean 
		node.state_covariance = self.state_covariance
		return node 


class Estimator(Estimator): 

	def __init__(self,param,env):
		super(Estimator, self).__init__(param,env)

	def estimate(self,observations,actions):

		# extract
		nodes,xbar,Pbar,y,C,R,R_inv,A_i,Q_i = [],dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict()
		for node, observation in observations.items():
			nodes.append(node)
			xbar[node] = node.state_mean
			Pbar[node] = node.state_covariance
			y[node] = observation[0]
			C[node] = observation[1]
			R[node] = observation[2]
			R_inv[node] = np.linalg.pinv(R[node])
			A_i[node] = node.dynamics.A
			Q_i[node] = node.dynamics.Q

		# make global 
		xbar = xbar[nodes[0]]
		Pbar = Pbar[nodes[0]]
		A = np.zeros((self.param.state_dim,self.param.state_dim))
		Q = np.zeros((self.param.state_dim,self.param.state_dim))
		for node in nodes: 
			A[node.global_state_idxs,node.global_state_idxs] = A_i[node]
			Q[node.global_state_idxs,node.global_state_idxs] = Q_i[node]


		estimates = dict()

		for node in self.env.nodes: 
			estimate = Estimate(self.env.state, 0)
			estimates[node] = estimate 

		return estimates 