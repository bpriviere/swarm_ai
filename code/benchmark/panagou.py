

import numpy as np 
import sys
from scipy.optimize import fsolve, linear_sum_assignment
from collections import defaultdict

sys.path.append("../")
from param import Param 
import plotter 

class PanagouPolicy:

	def __init__(self,param):
		self.param = param 

	def init_sim(self,initial_condition):

		self.initial_condition = initial_condition
		self.robots = self.param.robots 

		# find angles and times to goal 
		self.theta_noms, self.terminal_times = find_nominal_solns(self.param,self.robots)

		# discretize
		num_theta = 50
		self.thetas = 2*np.pi/num_theta * np.arange(num_theta)
		self.times = np.arange(0,1.5*max(self.terminal_times),self.param.sim_dt)

		# trajectories
		self.R_nom = calculate_nominal_trajectories(self.param,self.robots,self.times,self.theta_noms)
		self.R = calculate_all_trajectories(self.param,self.robots,self.times,self.thetas)

		# intersections
		self.I = calculate_intersections(self.param,self.robots,self.times,self.R)

		# matching polciies
		self.matching_policies = calculate_matching_policies(self.param,self.I,self.R)

		# match
		self.matching = calculate_defender_matching(self.param,self.matching_policies,self.times,self.terminal_times)

	def update_actions(self,state):
		# update conditions, i.e. capture, new attacker in sensing radius, etc. 
		return False 

	def eval(self,state):

		if self.update_actions(state):
			self.matching_policies = calculate_matching_policies(self.param,self.I,self.R)
			self.matching = calculate_defender_matching(self.param,self.matching_policies)

		actions = np.zeros((len(self.robots),2))
		done = [] 
		for i_robot,robot in enumerate(self.robots):

			if i_robot not in done: 

				if robot["team"] == "a":
					if self.matching[i_robot] == None:
						print('not matched')
						actions[i_robot,:] = theta_to_u(robot,self.theta_noms[i_robot])
					else:
						j_robot = self.matching[i_robot]
						(i_theta_star,j_theta_star,_,_) = self.matching_policies[i_robot,j_robot]
						actions[i_robot,:] = theta_to_u(robot,self.thetas[i_theta_star]) 
						actions[j_robot,:] = theta_to_u(robot,self.thetas[j_theta_star])
						done.append(j_robot)

				elif robot["team"] == "b":
					if not i_robot in self.matching.values():
						# decelerate to zero 
						actions[i_robot,:] = -robot["acceleration_limit"] * state[i_robot,2:]

			done.append(i_robot)

		return actions 

	def rollout(self,initial_condition):

		states = [np.array(initial_condition)] 
		for time in self.times: 
			state = np.zeros((len(self.robots),4))
			actions = self.eval(states[-1])
			for i_robot,robot in enumerate(self.robots):
				theta = np.arctan2(actions[i_robot,1],actions[i_robot,0])
				state[i_robot,:] = step(robot, theta, states[-1][i_robot,:], self.param.sim_dt)
			states.append(state)
		states = np.array(states)
		return states 

def calculate_matching_policies(param,I,R):
	# calculate attacker policies (and resulting defender policy) for each possible defender matchup
	policies = dict()
	for (ii_robot, jj_robot), intersections in I.items():

		ii_theta_star = 0
		jj_theta_star = 0
		ii_time_star = 0 
		min_dist_to_goal = np.inf 

		for (ii_theta,jj_theta,ii_time) in intersections:
			intersection_dist_to_goal = np.linalg.norm(R[ii_robot,ii_time,ii_theta,0:2] - param.goal[0:2])
			if min_dist_to_goal > intersection_dist_to_goal:
				ii_theta_star = ii_theta
				jj_theta_star = jj_theta
				ii_time_star = ii_time
				min_dist_to_goal = intersection_dist_to_goal

		policies[ii_robot,jj_robot] = (ii_theta_star,jj_theta_star,ii_time_star,min_dist_to_goal)
	return policies 

def calculate_defender_matching(param,policies,times,terminal_times):
	# greedy match defenders to attackers, who pick to maximize intersection distance to goal 
	matching = dict()
	done = [] 
	for i_robot in param.team_1_idxs: 
		matching[i_robot] = None
		max_dist = 0 
		for (ii_robot,jj_robot), (ii_theta,jj_theta,ii_time,dist_to_goal) in policies.items():
			if i_robot == ii_robot and dist_to_goal > max_dist and not jj_robot in done and times[ii_time] < terminal_times[i_robot]: 
				matching[ii_robot] = jj_robot
				done.append(jj_robot)
				max_dist = dist_to_goal
	return matching 

def calculate_nominal_trajectories(param,robots,times,theta_noms):
	# calculate nominal trajectory to goal 
	R_nom = np.zeros((len(robots),len(times),4))
	for i_robot,robot in enumerate(robots): 
		curr_state = np.array(robot["x0"])
		for i_time,time in enumerate(times):
			R_nom[i_robot,i_time,:] = curr_state
			curr_state = step(robot, theta_noms[i_robot], curr_state, param.sim_dt)			
	return R_nom 

def calculate_intersections(param,robots,times,R):
	# calculate intersections
	I = defaultdict(list) 
	for i_robot in param.team_1_idxs:
		for j_robot in param.team_2_idxs: 
			for i_time,time in enumerate(times):

				R_i = R[i_robot,i_time,:,0:2] # ntheta x 2 
				R_j = R[j_robot,i_time,:,0:2]

				dist = np.zeros((R_i.shape[0],R_j.shape[0]))
				for i_theta in range(R_i.shape[0]): 
					for j_theta in range(R_j.shape[0]):
						dist = np.linalg.norm(R_i[i_theta]-R_j[j_theta]) # matrix of size thetas x thetas 

						# intersection! 
						if dist < robots[j_robot]["tag_radius"]:
						# if dist < robots[i_robot]["radius"] + robots[j_robot]["radius"]:
							key = (i_robot,j_robot) 
							I[key].append((i_theta,j_theta,i_time))
	return I 

def calculate_all_trajectories(param,robots,times,thetas):
	# calculate all possible trajectories
	R = np.zeros((len(robots),len(times),len(thetas),4))
	for i_robot,robot in enumerate(robots): 
		for i_theta,theta in enumerate(thetas):
			curr_state = np.array(robot["x0"])
			for i_time,time in enumerate(times): 
				R[i_robot,i_time,i_theta,:] = curr_state
				curr_state = step(robot, theta, curr_state, param.sim_dt)
	return R 

def find_nominal_solns(param,robots):
	# find longest possible time 
	theta_noms, terminal_times = [], []
	for robot in robots: 
		state = np.array(robot["x0"])
		theta_nom, terminal_time = find_nominal_soln(param,robot,state)
		theta_noms.append(theta_nom)
		terminal_times.append(terminal_time)
	return theta_noms, terminal_times 

def step(robot, theta, x0, dt):
	 
	state = np.zeros(x0.shape)
	u = np.array((np.cos(theta),np.sin(theta)))
	
	# dynamics 
	state[0:2] = x0[0:2] + x0[2:]*dt 
	state[2:] = x0[2:] + robot["acceleration_limit"]*dt*u

	# velocity clipping 
	alpha = np.max((1,np.linalg.norm(state[2:])/robot["speed_limit"]))
	state[2:] /= alpha 

	return state

def integrate(robot, theta, state, times, dt):

	states = np.zeros((len(times)+1,4))
	states[0,:] = state 
	for i_time,time in enumerate(times): 
		states[i_time+1,:] = step(robot, theta, states[i_time,:], dt)
	return states

def find_nominal_soln(param,robot,state):

	def equations(p):
		th, T = p
		times = np.arange(0,T,param.sim_dt)
		states = integrate(robot,th,state,times,param.sim_dt)
		eqns = (
			states[-1,0] - param.goal[0], 
			states[-1,1] - param.goal[1], 
			)
		return eqns

	def approx_equations(p):
		th, T = p
		eqns = (
			state[0] - param.goal[0] + state[2]*T + ((robot["acceleration_limit"]*T**2)/2)*np.cos(th), 
			state[1] - param.goal[1] + state[3]*T + ((robot["acceleration_limit"]*T**2)/2)*np.sin(th),
			)
		return eqns

	th_tilde, T_tilde = fsolve(approx_equations, (0, 1))
	th, T =  fsolve(equations, (th_tilde, T_tilde))

	# check quality 
	times = np.arange(0,T,param.sim_dt)	
	states = integrate(robot,th,np.array(robot["x0"]),times,param.sim_dt)
	if np.linalg.norm(states[-1,0:2] - param.goal[0:2]) > robot["radius"]:
		# exit('bad nominal solution')
		print('bad nominal solution')

	return th,T

def theta_to_u(robot,theta):
	return robot["acceleration_limit"]*np.array((np.cos(theta),np.sin(theta)))

def main():

	set_ic = False
	if set_ic: 
		num_robots_a = 1 
		num_robots_b = 1 
		initial_condition = np.zeros((num_robots_a + num_robots_b,4))
		for i_robot in range(num_robots_a + num_robots_b):
			initial_condition[i_robot,:] = np.array((0.125+0.125*i_robot,0.125+0.125*i_robot,0,0))
		df_param = Param()
		df_param.update(initial_condition=initial_condition)

	else: 
		df_param = Param()
		initial_condition = df_param.state

	pp = PanagouPolicy(df_param)
	pp.init_sim(initial_condition)
	states = pp.rollout(initial_condition)

	plotter.plot_panagou(pp.R,pp.R_nom,pp.I,states,pp.param)
	plotter.save_figs("../plots/panagou_2017.pdf")
	plotter.open_figs("../plots/panagou_2017.pdf")

	# run_sim(df_param)

if __name__ == '__main__':
	main()