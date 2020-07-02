

import numpy as np 
import cvxpy as cp 
from scipy.optimize import linear_sum_assignment

from controller.controller import Controller
	
class Controller(Controller):

	def __init__(self,param,env):
		super(Controller, self).__init__(param,env)

	def policy(self,estimate):

		actions = dict()

		# team A policy is to go to goal 
		states_A, actions_A = self.policy_A(self.env.nodes,estimate)

		# team B policy is to do nothing
		states_B, actions_B = self.policy_B(self.env.nodes, estimate, states_A, actions_A)

		# recombine 
		for node in self.env.nodes:
			if node.team_A:
				actions[node] = np.expand_dims(actions_A[node][:,0],axis=1)
			elif node.team_B:
				actions[node] = np.expand_dims(actions_B[node][:,0],axis=1)

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
		goal_line_x = self.param.goal_line_x
		pos_upper_lim = np.array([self.param.env_xlim[1],self.param.env_ylim[1]])
		pos_lower_lim = np.array([self.param.env_xlim[0],self.param.env_ylim[0]])

		# get utils
		x0_A, u0_A, node_idxs_A, state_idxs_A, control_idxs_A, extract_pos_A, extract_vel_A = self.get_team_util(nodes_A,estimate)

		# 
		G = np.array([1,0,0,0]) 

		# init 
		x_t = cp.Variable((x0_A.shape[0],self.param.rhc_horizon+1))
		u_t = cp.Variable((u0_A.shape[0],self.param.rhc_horizon))
		cost = 0
		constr = []

		# initial condition 
		constr.append(x_t[:,0] == x0_A[:,0])

		for timestep in range(self.param.rhc_horizon):

			for node in nodes_A:

				state_idx = state_idxs_A[node]
				control_idx = control_idxs_A[node]

				# dynamics
				constr.append(
					x_t[state_idx,timestep+1] == node.dynamics.A @ x_t[state_idx,timestep] + \
					node.dynamics.B @ u_t[control_idx,timestep] )

				# state constraints 
				constr.append(
					extract_pos_A[node] @ x_t[state_idx,timestep] <= pos_upper_lim)
				constr.append(
					extract_pos_A[node] @ x_t[state_idx,timestep] >= pos_lower_lim)
				constr.append(
					cp.norm(extract_vel_A[node] @ x_t[state_idx,timestep],2) <= speed_limit_a)

				# control authority constraints
				constr.append(
					cp.norm(u_t[control_idx,timestep],2) <= acceleration_limit_a)			

				# cost 
				relative_goal = G @ x_t[state_idx,timestep] - goal_line_x
				effort = lambda_u * u_t[control_idx,timestep]

				cost += cp.sum_squares(relative_goal)
				cost += cp.sum_squares(effort)

		# solve 
		obj = cp.Minimize(cost)
		prob = cp.Problem(obj, constr)
		prob.solve(verbose=False,solver=cp.GUROBI) 

		# assign 
		states = dict()
		actions = dict()
		for node in nodes_A: 
			states[node] = x_t.value[state_idxs_A[node],:]
			actions[node] = u_t.value[control_idxs_A[node],:]

		return states,actions


	def policy_B(self,nodes,estimate, states_A, actions_A):


		# param
		lambda_u = self.param.lambda_u
		speed_limit_b = self.param.speed_limit_b
		acceleration_limit_b = self.param.acceleration_limit_b
		pos_upper_lim = np.array([self.param.env_xlim[1],self.param.env_ylim[1]])
		pos_lower_lim = np.array([self.param.env_xlim[0],self.param.env_ylim[0]])

		# extract teams 
		nodes_A, nodes_B = [], []
		for node in nodes:
			if node.team_A:
				nodes_A.append(node)
			elif node.team_B:
				nodes_B.append(node)

		# some utils 
		x0_A, u0_A, node_idxs_A, state_idxs_A, control_idxs_A, extract_pos_A, extract_vel_A = self.get_team_util(nodes_A,estimate)
		x0_B, u0_B, node_idxs_B, state_idxs_B, control_idxs_B, extract_pos_B, extract_vel_B = self.get_team_util(nodes_B,estimate)

		# match 
		dist = np.zeros((len(nodes_B),len(nodes_A)))
		for node_A in nodes_A: 
			pos_a = node_A.state[0:2]
			for node_B in nodes_B: 
				pos_b = node_B.state[0:2]
				dist[node_idxs_B[node_B],node_idxs_A[node_A]] = np.linalg.norm(pos_a-pos_b)

		node_a_assignment, node_b_assignment = linear_sum_assignment(dist) # try to minimize this 

		matching = dict()
		for node_B in nodes_B:
			matching[node_B] = nodes_A[node_b_assignment[node_idxs_B[node_B]]]


		# convex policy 

		# cost func stuff 
		G, goal_vec = dict(), dict()
		for node in nodes_B: 
			goal_vec[node] = states_A[matching[node]] # should be time varying predicted state 
			G[node] = np.eye(node.dynamics.state_dim_per_agent)

		# init 
		x_t = cp.Variable((x0_B.shape[0],self.param.rhc_horizon+1))
		u_t = cp.Variable((u0_B.shape[0],self.param.rhc_horizon))
		cost = 0
		constr = []

		# initial condition 
		constr.append(x_t[:,0] == x0_B[:,0])

		for timestep in range(self.param.rhc_horizon):

			# solve for optimal control of nodes in B
			for node in nodes_B:

				state_idx = state_idxs_B[node]
				control_idx = control_idxs_B[node]

				# dynamics
				constr.append(
					x_t[state_idx,timestep+1] == node.dynamics.A @ x_t[state_idx,timestep] + \
					node.dynamics.B @ u_t[control_idx,timestep] )

				# state constraints 
				constr.append(
					extract_pos_B[node] @ x_t[state_idx,timestep] <= pos_upper_lim)
				constr.append(
					extract_pos_B[node] @ x_t[state_idx,timestep] >= pos_lower_lim)
				constr.append(
					cp.norm(extract_vel_B[node] @ x_t[state_idx,timestep],2) <= speed_limit_b)

				# control authority constraints
				constr.append(
					cp.norm(u_t[control_idx,timestep],2) <= acceleration_limit_b)	

				# cost 
				relative_goal = G[node] @ (x_t[state_idx,timestep] - goal_vec[node][:,timestep])
				effort = lambda_u * u_t[control_idx,timestep]

				cost += cp.sum_squares(relative_goal)
				cost += cp.sum_squares(effort)

		# solve 
		obj = cp.Minimize(cost)
		prob = cp.Problem(obj, constr)
		prob.solve(verbose=False,solver=cp.GUROBI)

		# assign 
		states = dict()
		actions = dict()
		for node in nodes_B: 
			states[node] = x_t.value[state_idxs_B[node],:]
			actions[node] = u_t.value[control_idxs_B[node],:]

		return states,actions



	def get_team_util(self,nodes,estimates):

		# joint space maps 
		state_idxs = dict()
		control_idxs = dict()
		node_idxs = dict()
		count = 0 
		curr_state_dim = 0 
		curr_control_dim = 0 
		for node in nodes:
			state_idxs[node] = curr_state_dim + np.arange(0,node.dynamics.state_dim_per_agent)
			control_idxs[node] = curr_control_dim + np.arange(0,node.dynamics.control_dim_per_agent)
			node_idxs[node] = count
			curr_state_dim += node.dynamics.state_dim_per_agent
			curr_control_dim += node.dynamics.control_dim_per_agent
			count += 1 

		# define initial condition in joint space 
		x0 = np.zeros((curr_state_dim,1))
		u0 = np.zeros((curr_control_dim,1))
		curr_state_dim = 0 
		for node in nodes: 
			x0[state_idxs[node]] = estimates[node].state_mean[node.global_state_idxs] 

		extract_pos = dict()
		extract_vel = dict()
		for node in nodes: 
			extract_pos[node] = np.hstack((np.eye(2), np.zeros((2,2)))) 
			extract_vel[node] = np.hstack((np.zeros((2,2)), np.eye(2)))

		return x0, u0, node_idxs, state_idxs, control_idxs, extract_pos, extract_vel