

import numpy as np 
import cvxpy as cp 
from scipy.optimize import linear_sum_assignment

class Controller: 

	def __init__(self,param,env):
		self.env = env
		self.param = param 


	def policy(self,estimate):

		actions = dict()

		# team A policy is to go to goal 
		actions_A = self.policy_A(self.env.nodes,estimate)

		# team B policy is to do nothing
		actions_B = self.policy_B(self.env.nodes,actions_A,estimate)

		# recombine 
		for node in self.env.nodes:
			if node.team_A:
				actions[node] = actions_A[node]
			elif node.team_B:
				actions[node] = actions_B[node] 

		return actions 


	def policy_A(self,nodes,estimate):
		# for now, estimate = state 

		nodes_A, nodes_B = [], []
		for node in nodes:
			if node.team_A:
				nodes_A.append(node)
			elif node.team_B:
				nodes_B.append(node)

		# param
		lambda_u = self.param.lambda_u
		speed_limit_a = self.param.speed_limit_a
		acceleration_limit_a = self.param.acceleration_limit_a

		# joint space maps 
		node_state_idxs = dict()
		node_control_idxs = dict()
		curr_state_dim = 0 
		curr_control_dim = 0 
		for node in nodes_A:
			node_state_idxs[node] = curr_state_dim + np.arange(0,node.dynamics.state_dim_per_agent)
			node_control_idxs[node] = curr_control_dim + np.arange(0,node.dynamics.control_dim_per_agent)
			curr_state_dim += node.dynamics.state_dim_per_agent
			curr_control_dim += node.dynamics.control_dim_per_agent

		# define initial condition in joint space 
		x0 = np.zeros((curr_state_dim,1))
		for node in nodes_A: 
			x0[node_state_idxs[node]] = node.state # this line should be using the estimated state 

		# some utilities
		extract_pos = dict()
		extract_vel = dict()
		for node in nodes_A: 
			extract_pos[node] = np.zeros((2,node.dynamics.state_dim_per_agent))
			extract_vel[node] = np.zeros((2,node.dynamics.state_dim_per_agent))

		# cost func stuff 
		goal_vec = dict() 
		G = dict()
		for node in nodes_A: 
			goal_vec[node] = self.param.goal_line_x 
			G[node] = np.array([1,0,0,0]) 

		# prepare
		pos_upper_lim = np.array([self.param.env_xlim[1],self.param.env_ylim[1]])
		pos_lower_lim = np.array([self.param.env_xlim[0],self.param.env_ylim[0]])

		# init 
		x_t = cp.Variable((x0.shape[0],self.param.rhc_horizon))
		u_t = cp.Variable((curr_control_dim,self.param.rhc_horizon-1))
		cost = 0
		constr = []

		# initial condition 
		constr.append(x_t[:,0] == x0[:,0])

		for timestep in range(self.param.rhc_horizon-1):

			for node in nodes_A:

				node_state_idx = node_state_idxs[node]
				node_control_idx = node_control_idxs[node]

				# dynamics
				constr.append(
					x_t[node_state_idx,timestep+1] == node.dynamics.A @ x_t[node_state_idx,timestep] + \
					node.dynamics.B @ u_t[node_control_idx,timestep] )

				# state constraints 
				constr.append(
					extract_pos[node] @ x_t[node_state_idx,timestep] <= pos_upper_lim)
				constr.append(
					extract_pos[node] @ x_t[node_state_idx,timestep] >= pos_lower_lim)
				constr.append(
					cp.norm(extract_vel[node] @ x_t[node_state_idx,timestep],2) <= speed_limit_a)

				# control authority constraints
				constr.append(
					cp.norm(u_t[node_control_idx,timestep],2) <= acceleration_limit_a)			

				# cost 
				relative_goal = G[node] @ x_t[node_state_idx,timestep] - goal_vec[node] 
				effort = lambda_u * u_t[node_control_idx,timestep]

				cost += cp.sum_squares(relative_goal)
				cost += cp.sum_squares(effort)

		# solve 
		obj = cp.Minimize(cost)
		prob = cp.Problem(obj, constr)
		prob.solve(verbose=False) 

		# assign 
		actions = dict()
		for node in nodes_A: 
			actions[node] = np.expand_dims(u_t.value[node_control_idxs[node],0],axis=1)

		return actions 


	def policy_B(self,nodes,actions_A,estimate):

		nodes_A, nodes_B = [], []
		for node in nodes:
			if node.team_A:
				nodes_A.append(node)
			elif node.team_B:
				nodes_B.append(node)

		# match 
		dist = np.zeros((len(nodes),len(nodes)))
		for node_a in nodes_A: 
			pos_a = node_a.state[0:2]
			for node_b in nodes_B: 
				pos_b = node_b.state[0:2]
				dist[node_b.idx,node_a.idx] = np.linalg.norm(pos_a-pos_b)

		print(dist)

		node_a_assignment, node_b_assignment = linear_sum_assignment(dist) # tries to minimize this 

		print(node_a_assignment)
		print(node_b_assignment)
		exit()


		# then convex policy 

		actions = dict()
		for node in nodes:
			action = np.zeros((node.dynamics.control_dim_per_agent,1))
			actions[node] = action 	
		return actions 		