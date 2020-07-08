
import numpy as np 
from estimator.estimator import Estimator, Estimate

class Estimate(Estimate):

	def __init__(self,state_mean,state_covariance):
		super(Estimate, self).__init__(state_mean, state_covariance)

	def update_node(self,node):
		node.state_mean = self.state_mean 
		node.state_covariance = self.state_covariance
		return node 


class Estimator(Estimator): 

	def __init__(self,param,env):
		super(Estimator, self).__init__(param,env)

	def estimate(self,observations,actions):
		
		# extract
		nodes,xbar,Pbar,z,H,R,R_inv,F_i,B_i,u_i,Q_i,Q_i_inv = [],dict(),dict(),dict(),dict(),\
			dict(),dict(),dict(),dict(),dict(),dict(),dict()
		for node, observation in observations.items():
			nodes.append(node)
			xbar[node] = node.state_mean
			Pbar[node] = node.state_covariance
			u_i[node] = actions[node]
			z[node] = observation[0]
			H[node] = observation[1]
			R[node] = observation[2]
			F_i[node] = node.dynamics.A
			B_i[node] = node.dynamics.B
			Q_i[node] = node.dynamics.Q
			R_inv[node] = np.linalg.pinv(R[node])
			Q_i_inv[node] = np.linalg.pinv(Q_i[node])

		# make global representation
		xbar = xbar[nodes[0]]
		Pbar = Pbar[nodes[0]]
		F = np.zeros((self.param.state_dim,self.param.state_dim))
		Q = np.zeros((self.param.state_dim,self.param.state_dim))
		Q_inv = np.zeros((self.param.state_dim,self.param.state_dim))
		u = np.zeros((self.param.control_dim,1))
		B = np.zeros((self.param.state_dim,self.param.control_dim))
		for node in nodes: 
			F[np.ix_(node.global_state_idxs,node.global_state_idxs)] = F_i[node]
			Q[np.ix_(node.global_state_idxs,node.global_state_idxs)] = Q_i[node]
			Q_inv[np.ix_(node.global_state_idxs,node.global_state_idxs)] = Q_i_inv[node]
			B[np.ix_(node.global_state_idxs,node.global_control_idxs)] = B_i[node]
			u[node.global_control_idxs] = u_i[node]
		
		wiki = False
		if wiki:
			# filter 
			Y_km1km1 = np.linalg.pinv(Pbar)
			y_km1km1 = np.dot(Y_km1km1,xbar)

			# predict 
			F_inv = np.linalg.pinv(F) 
			M_k = np.dot(np.dot(F_inv.T,Y_km1km1),F_inv)
			C_k = np.dot(M_k,np.linalg.pinv(M_k + Q_inv))
			L_k = np.eye(self.param.state_dim) - C_k 
			Y_kkm1 = np.dot(np.dot(L_k,M_k),L_k.T) + np.dot(np.dot(C_k,Q_inv),C_k.T)
			y_kkm1 = np.dot(np.dot(L_k,F_inv.T),y_km1km1)

			# innovate
			I_k = np.zeros((self.param.state_dim,self.param.state_dim))
			i_k = np.zeros((self.param.state_dim,1))
			for node in nodes: 
				i_k += np.dot(np.dot(H[node].T,R_inv[node]),z[node])
				I_k += np.dot(np.dot(H[node].T,R_inv[node]),H[node])

			Y_kk = Y_kkm1 + I_k 
			y_kk = y_kkm1 + i_k 

			Pbar_kk = np.linalg.pinv(Y_kk)
			xbar_kk = np.dot(Pbar_kk,y_kk) 

		else: 

			# predict 
			Pbar_kkm1 = np.dot(np.dot(F,Pbar),F.T) + Q 
			xbar_kkm1 = np.dot(F,xbar) + np.dot(B,u)

			# filter 
			Y_kkm1 = np.linalg.pinv(Pbar_kkm1)
			y_kkm1 = np.dot(Y_kkm1,xbar_kkm1)

			# innovate
			I_k = np.zeros((self.param.state_dim,self.param.state_dim))
			i_k = np.zeros((self.param.state_dim,1))
			for node in nodes: 
				i_k += np.dot(np.dot(H[node].T,R_inv[node]),z[node])
				I_k += np.dot(np.dot(H[node].T,R_inv[node]),H[node])

			Y_kk = Y_kkm1 + I_k 
			y_kk = y_kkm1 + i_k 

			Pbar_kk = np.linalg.pinv(Y_kk)
			xbar_kk = np.dot(Pbar_kk,y_kk) 			

		estimates = dict()
		for node in self.env.nodes: 
			estimate = Estimate(xbar_kk,Pbar_kk)
			estimates[node] = estimate 
		return estimates 
