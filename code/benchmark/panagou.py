#!/usr/bin/env python3

import numpy as np 
import sys
import os 
import math
from scipy.optimize import fsolve, linear_sum_assignment
from collections import defaultdict

from datetime import datetime

sys.path.append("../")
from param import Param 
import plotter 

"""
Robot State
	X = [ X_pos, Y_pos, X_vel, Y_vel ]

Action Vector
	U = [ X_accel, Y_accel ]

"""

class PanagouPolicy:

	def __init__(self,param):
		self.param = param 

	def init_sim(self,initial_condition):

		self.initial_condition = initial_condition
		self.robots = self.param.robots 

		# Calculate intial actions
		self = self.update_actions(np.array(initial_condition),1)

	def update_actions(self,state,recalculate_actions):
		# update conditions, i.e. capture, new attacker in sensing radius, etc. at each time step
		# Probably a good spot to check that the goal changing is working
		# Re-calculate the game to force a closed-loop game
		if (recalculate_actions) :

			# Update the robots thing so it knows where to start
			for ii in range(len(self.robots)) :
				self.robots[ii]["x0"] = state[ii]

			# find angles and times to goal 
			self.theta_noms, self.terminal_times = find_nominal_solns(self.param,self.robots)

			# discretize
			num_theta = 50
			self.thetas = 2*np.pi/num_theta * np.arange(num_theta)
			self.times = np.arange(0,1.5*max(self.terminal_times),self.param.sim_dt)	# Extend time so that collisions can be calculated
																						# Our survival time might be longer than it takes for
																						# us to reach the goal if there were no attacker

			# trajectories
			self.R_nom = calculate_nominal_trajectories(self.param,self.robots,self.times,self.theta_noms)  # Constant acceleration to goal (only needed for plots)
			self.R = calculate_all_trajectories(self.param,self.robots,self.times,self.thetas)              # Expanding reachable space for each robot

			# intersections
			self.I = calculate_intersections(self.param,self.robots,self.times,self.R)

			# matching polciies
			self.matching_policies = calculate_matching_policies(self.param,self.I,self.R)

			# match
			self.matching = calculate_defender_matching(self.param,self.matching_policies,self.times,self.terminal_times)

		return self 

	def eval(self,state):

		# Recalculate reachable set and attacker/defender matching
		recalculate_actions = 1
		self = self.update_actions(state,recalculate_actions)

		actions = np.zeros((len(self.robots),2))
		terminal_times = np.zeros((1,len(self.robots)))

		done = [] 
		# Calculate robot actions
		for i_robot,robot in enumerate(self.robots):

			
			if (np.any(np.isnan(state[i_robot]))) :
				# Robot is dead, we can consider it done
				done.append(i_robot)

			if i_robot not in done: 

				if robot["team"] == "a":     # attackers
					if self.matching[i_robot] == None:
						# Attacker will win, go straight to the goal
						#print('Attacker wins')
						actions[i_robot,:] = theta_to_u(robot,self.theta_noms[i_robot])

						# Update the terminal time for the robot
						terminal_times[0,i_robot] = self.terminal_times[i_robot]

					else:
						# Attacker will be captured, minimise distance to goal on capture
						#print("Attacker looses")
						j_robot = self.matching[i_robot]
						(i_theta_star,j_theta_star,idx_t_capture,_) = self.matching_policies[i_robot,j_robot]

						# Update the actions with the capture case
						attacker_action_theta = self.thetas[i_theta_star]
						defender_action_guess = self.thetas[j_theta_star]
						actions[i_robot,:] = theta_to_u(robot,attacker_action_theta) 

						# Calculate the best accelerations to minimise the time to intercept the 
						# attacker based on the attacker's known acceleration direction (and hence trajectory).				
						if (0) : print(" Isoline intercept theta %6.2f [ deg ] at t = %5.2f [ s ]" % (defender_action_guess*57.7, self.times[idx_t_capture]))
						
						# If we're close and slow, go for the attacker, otherwise, do the interception thing
						att_pos = state[i_robot,0:2]
						def_pos = state[j_robot,0:2]
						if (np.linalg.norm(att_pos - def_pos) < 2.5*self.robots[j_robot]["tag_radius"]) :
							# Lunge for the attacker
							def_theta = defender_action_guess
							Tend = 1.0

						else :
							# Try and reach the attacker at the predicted interception point
							def_theta,Tend = find_best_intercept( \
								self.robots[i_robot],self.robots[j_robot],attacker_action_theta,defender_action_guess,self.param.sim_dt)
						
						# Convert action into accelX and accelY
						actions[j_robot,:] = theta_to_u(self.robots[j_robot],def_theta) 

						# Mark j_robot as calculated
						done.append(j_robot)

						# Update the terminal time for the robots
						terminal_times[0,i_robot] = Tend
						terminal_times[0,j_robot] = Tend


				elif robot["team"] == "b":   # defenders
					# Match not found for attacker
					if not i_robot in self.matching.values():
						# Decelerate to zero at maximum acceleration
						accel = -1*state[i_robot,2:] / self.param.sim_dt

						# Limit to maximum allowable acceleration
						if np.linalg.norm(accel) > abs(robot["acceleration_limit"]) :
							accel = accel / np.linalg.norm(accel) * abs(robot["acceleration_limit"])

						# Update action
						actions[i_robot,:] = accel

						# Update the terminal time for the robot
						terminal_times[0,i_robot] = self.terminal_times[i_robot]

					else : 
						# do nothing, the action is calculated in the attacker's step
						pass						

			done.append(i_robot)

		# Clamp actions for all robots
		if (1) :
			for ii in range(len(self.robots)) :
				actions[ii,:] = clamp_action(self.robots[ii], state[ii,:], actions[ii,:], self.param.sim_dt)

		# All done, return calculated actions
		return actions 

	def rollout(self,initial_condition):

		states = [np.array(initial_condition)]

		# Simulation end
		t_end = np.max(self.terminal_times)+self.param.sim_dt

		#t_end = 5
		print("t_end = %.2f" % (t_end))
		times = np.arange(0,t_end,self.param.sim_dt)
		game_over = np.zeros((len(self.robots),1))

		# Loop through each of the times in the time vector
		for time in times: 

			print("t: %5.2f || " % (time),end='')

			state = np.zeros((len(self.robots),4))

			# Calculate what action to take
			actions = self.eval(states[-1])  
			
			# Step a time step for each robot
			for i_robot,robot in enumerate(self.robots):
				if (0): print("(%d) " % i_robot,end='')
				distance_to_goal = 1e6
				distance_to_capture = np.empty((0))

				# Work out the robot's distances to things
				if (robot["team"] == "a") :     # attacker
					print("A%d > " % i_robot, end='')

					# Calculate distances to goal
					distance_to_goal    = math.sqrt((states[-1][i_robot][0]-self.param.goal[0])**2 + (states[-1][i_robot][1]-self.param.goal[1])**2)
					print("(goal): %6.3f, " % (distance_to_goal),end='')

					# Calculate distances to all attackers
					for j_robot in self.param.team_2_idxs: 
						distance_to_this_attacker = math.sqrt((states[-1][i_robot][0]-states[-1][j_robot][0])**2 + (states[-1][i_robot][1]-states[-1][j_robot][1])**2)
						distance_to_capture  = np.append(distance_to_capture, np.array([distance_to_this_attacker]), axis=0)
						print("(D%d): %6.3f, " % (j_robot,distance_to_this_attacker), end='')

				else :
					# Robot is defender so we don't need to calculate game-state stuff
					distance_to_capture = 1e6
					pass

				# Step the state
				if (distance_to_goal < robot["tag_radius"]) : 
					# Attacker has won, propogate out the last state for all robots
					print("A%d WINS || " % (i_robot) ,end='')
					state[:,:] = states[-1][:,:]

					# End the game for everyone
					game_over.fill(1)

				elif (np.any(distance_to_capture < robot["tag_radius"])) : 
					# Defender has won, propogate out the last state to only the captured robot
					j_robot = self.param.team_2_idxs[np.argmin(distance_to_capture)]

					print("A%d DEF  || " % (i_robot), end='')
					state[i_robot,:] = states[-1][i_robot,:]
					state[j_robot,:] = states[-1][j_robot,:]

					# End the game for the defender and attacker
					game_over[i_robot] = 1
					game_over[j_robot] = 1

				elif (np.all(game_over == 1)) :
					# Game has ended (probably from a capture)
					if (robot["team"] == "a") : print("        || ", end='')
					state[i_robot,:] = states[-1][i_robot,:]

				else :
					# Game is still on!
					if (robot["team"] == "a") : print("        || ", end='')
					state[i_robot,:] = step(robot, states[-1][i_robot,:], actions[i_robot,:], self.param.sim_dt)
				
				if (game_over[i_robot]) :
					# Game over for this robot, put it back to the previous state
					state[i_robot,:] = states[-1][i_robot,:]

				if (0): print(" <> ",end='')

			# Add the current state to the state matrix
			states.append(state)
			if (1): print("\n",end='')
		states = np.array(states)
		return states 

def calculate_matching_policies(param,I,R):
	# calculate attacker policies (and resulting defender policy) for each possible defender matchup
	# Policies
	# 	[ att_heading_idx, def_heading_idx, idx_time, min_distance_to_goal ]
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
	# policies = [ att_heading_idx, def_heading_idx, idx_time, min_distance_to_goal ]
	matching = dict()
	done = [] 

	# Loop through each attacker
	for i_robot in param.team_1_idxs: 
		matching[i_robot] = None
		max_dist = 0 

		# ii_robot = attacker, jj_robot = defender
		for (ii_robot,jj_robot), (ii_theta,jj_theta,ii_time,dist_to_goal) in policies.items():
			# Match the robots up
			if i_robot == ii_robot and dist_to_goal > max_dist and not jj_robot in done :
				t_goal = terminal_times[ii_robot]
				t_capture = times[ii_time]

				if (t_capture < t_goal+1.0) : # Chase the attacker if there's a chance we might catch him
				# Check to see if the time until intercept is less than the time for the attacker to reach the goal,
				# if so, this would be a capture and we should add it to the matching list
					matching[ii_robot] = jj_robot
					done.append(jj_robot)
					max_dist = dist_to_goal

		#if (matching[i_robot] == None) :
			# print("No match found!")
			# Pause when there's no match found for debugging
	
	return matching 

def calculate_nominal_trajectories(param,robots,times,theta_noms):
	# calculate nominal trajectory to goal 
	R_nom = np.zeros((len(robots),len(times),4))
	for i_robot,robot in enumerate(robots): 
		curr_state = np.array(robot["x0"])

		if np.any(np.isnan(curr_state)) :
			# Robot dead, ignore it
			pass

		else :
			for i_time,time in enumerate(times):
				R_nom[i_robot,i_time,:] = curr_state
				U = theta_to_u(robot,theta_noms[i_robot])
				curr_state = step(robot, curr_state, U, param.sim_dt)
	
	# All done
	return R_nom 

def calculate_intersections(param,robots,times,R):
	"""
	For each angle (theta), calculate the timestep where
	the distance is minimised between the two agents
	"""

	# R = [ robot, time, theta, X[posX,poxY,Vx,Vy] ]
	# I[att_ID, def_ID] = (att_idx_theta, def_idx_theta,idx_time) 

	# calculate intersections
	I = defaultdict(list) 
	
	# Loop through each attacker
	for i_robot in param.team_1_idxs:

		# Check with each defender for intersections
		for j_robot in param.team_2_idxs:
			
			# Check both robots are alive
			if ( np.any(np.isnan(R[i_robot])) or np.any(np.isnan(R[j_robot])) ) :
				# One of the robots is dead, don't add anything to the I matrix
				pass

			else : 

				# Loop through each theta
				for idx_theta in range(R.shape[2]):
					#print("theta idx %d" % (idx_theta))

					min_dist2_store = math.inf 
					idx_dist2_store = 0
					idx_t_store     = 0

					# Loop through each time step
					max_allowed_distance2 = np.power(robots[j_robot]["tag_radius"],2)
					max_allowed_distance2 = 0.01**2
					getting_closer = 1

					for idx_time in range(R.shape[1]):
						# Find the distance between the agents across all the defender's thetas
						posX1 = R[i_robot,idx_time,idx_theta,0] # attacker
						posY1 = R[i_robot,idx_time,idx_theta,1]

						posX2s = R[j_robot,idx_time,:,0] # defender
						posY2s = R[j_robot,idx_time,:,1]

						dist2 = np.power(posX2s-posX1,2) + np.power(posY2s-posY1,2) 

						# Find the closest (smallest) distance index
						idx_dist2 = np.where(dist2 == min(dist2))[0]
						idx_dist2 = idx_dist2[0]
						min_dist2 = dist2[idx_dist2]

						# Replace theta and time if closer than previous best
						# If we're moving away, we also don't want to store the value
						if (min_dist2 <= min_dist2_store) and getting_closer:
							min_dist2_store = min_dist2
							idx_dist2_store = idx_dist2  # attacker's theta idx
							idx_t_store = idx_time
							#print("\t\tFound a closer point (%f at idx %d)" % (min_dist2_store, idx_dist2_store ))

						else :
							getting_closer = 0
							
					# Check how close the defender actually got for this theta angle
					# If sufficently close, store the indicies for
					#     attacker_theta
					#     defender_theta
					#     time
					if (min_dist2_store < max_allowed_distance2): 
						if (0) :
							print("Storing Intersection : dist2: %.4f" % (min_dist2_store), end='')
							print(", idx_dist2: %d" % (idx_dist2_store), end='')
							print(", idx_t: %d\n" % (idx_t_store), end='')
						key = (i_robot,j_robot) 
						I[key].append((idx_theta,idx_dist2_store,idx_t_store))

						#print("t_death = %5.2f [ s ]" % (times[idx_t_store]))
	
	# All done
	return I 

def calculate_all_trajectories(param,robots,times,thetas):
	# calculate all possible trajectories
	R = np.zeros((len(robots),len(times),len(thetas),4))
	for i_robot,robot in enumerate(robots): 
		for i_theta,theta in enumerate(thetas):
			curr_state = np.array(robot["x0"])

			for i_time,time in enumerate(times): 

				R[i_robot,i_time,i_theta,:] = curr_state

				if np.any(np.isnan(curr_state)) :
					# Robot dead, ignore it
					curr_state = np.array([np.nan, np.nan, np.nan, np.nan])
				else : 
					U = theta_to_u(robot,theta)
					curr_state = step(robot, curr_state, U, param.sim_dt)
	# All done
	return R 

def find_nominal_solns(param,robots):
	# find longest possible time 
	theta_noms, terminal_times = [], []
	for robot in robots: 
		state = np.array(robot["x0"])

		if np.any(np.isnan(state)) :
			# Robot dead, ignore it
			theta_nom = 0
			terminal_time = 0
		else :
			# Robot alive, calculate solutions
			theta_nom, terminal_time = find_nominal_soln(param,robot,state)

		theta_noms.append(theta_nom)
		terminal_times.append(terminal_time)
	
	# Return, we have what we need
	return theta_noms, terminal_times 

def clamp_action(robot, X0, U0, dt) :
	# Runs the sim and back-calculates the action applied
	# Used to correct the accelerations 

	# run the step function to get new state
	X = step(robot, X0, U0, dt)

	# back-calculate what the actual control input was
	U = (X[2:]-X0[2:]) / dt

	# Return the new control input
	return U

def step(robot, x0, U, dt):  # Change to an X and Y acceleration
	# x0 = [ posX, posY, Vx, Vy ]
	#  U = [ accX, accY ]
	# 
	# The inputs to this function changed so hopefully nothing broke

	# Check inputs are ok
	if not x0.size == 4: print("X0 Input Incorrect Size")
	if not  U.size == 2: print("U  Input Incorrect Size")

	 
	state = np.zeros(x0.shape) # Output state vector

	# Extract useful stuff
	posX = x0[0]
	posY = x0[1]
	Vx   = x0[2]
	Vy   = x0[3]

	ax = U[0]
	ay = U[1]

	# Iterate a step
	posX_new = posX + Vx*dt
	posY_new = posY + Vy*dt

	Vx_new = Vx + ax*dt
	Vy_new = Vy + ay*dt
	V_new = math.sqrt(Vx_new*Vx_new + Vy_new*Vy_new)

	speed_limit = robot["speed_limit"]

	# Limit the velocity
	if (V_new > speed_limit):
		V_scalar = min((speed_limit / V_new),1.0)

	else:
		V_scalar = 1
	
	# Calc new limits
	Vx_new_lim = Vx_new * V_scalar
	Vy_new_lim = Vy_new * V_scalar

	# Assemble for return
	state[0] = posX_new
	state[1] = posY_new
	state[2] = Vx_new_lim
	state[3] = Vy_new_lim
	
	# Debugging
	if (0):
		#print("ax: %7.3f, ay: %7.3f, |a|: %4.1f, h: %6.1f |  " % (ax, ay, a, ah*57.7), end='')
		print("ax: %7.3f, ay: %7.3f |  " % (ax, ay), end='')
		#print("Vx: %6.3f (%6.3f), Vy: %6.3f (%6.3f), |V|: %5.3f(%6.3f), h: %6.1f (%6.1f)"    % (Vx_new_lim, Vx_new, Vy_new_lim, Vy_new, V_new_lim, V_new, Vh_new_lim*57.7, Vh_new*57.7, ), end='')
		print("Vx: %6.3f (%6.3f), Vy: %6.3f (%6.3f)"    % (Vx_new_lim, Vx_new, Vy_new_lim, Vy_new), end='')
		#print("\n", end='')

	return state

def integrate(robot, state, U, times, dt):
    # originally robot, theta, state, times, dt
	# Here dt probably should be automatically calculated but that's a future matt problem...
  
	states = np.zeros((len(times)+1,4))
	states[0,:] = state 

	for i_time,time in enumerate(times): 
		states[i_time+1,:] = step(robot, states[i_time,:], U, dt)
	return states

def find_nominal_soln(param,robot,state):

	def equations(p):
		th, T = p
		times = np.arange(0,T,param.sim_dt)

		# Convert theta (th) into U = [ accX, accY ]
		U = theta_to_u(robot,th)

		# Simulate system
		states = integrate(robot,state,U,times,param.sim_dt)

		# Extract useful information
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

	# Take a better guess at the initial conditions
	theta_guess = np.arctan2(param.goal[1]-state[1],param.goal[0]-state[0]) 
	Tend_guess = 3

	# Solve using the approximate equations to improve the initial guess
	th_tilde, T_tilde = fsolve(approx_equations, (theta_guess, Tend_guess))

	# Calculate the best solution using the full dynamics
	th, T =  fsolve(equations, (th_tilde, T_tilde))

	# Make sure the goal was acheived 
	if (0) :
		times = np.arange(0,T,param.sim_dt)	
		U = theta_to_u(robot,th)
		states = integrate(robot,np.array(robot["x0"]),U, times,param.sim_dt)

		if np.linalg.norm(states[-1,0:2] - param.goal[0:2]) > 1.1*robot["radius"]:
			# exit('bad nominal solution')
			print('\tbad nominal solution')
			# Wait here so we can debug what happened
			a = 1
	
	# Print debugging information
	#print("[ %c ] Theta: %5.2f [ rad ] , Terminal time: %5.2f [ s ]" % (robot["team"], T_tilde, T))

	return th,T

def find_best_intercept(att_robot,def_robot,att_theta,defender_action_guess,sim_dt) :
	# Calculates the trajectory to minimum-time-to-intercept for an attacker/defender 
	# pair when the attacker's trajectory is known.

	def equations(p):
		def_theta, Tend = p

		# This should be 10 iterations between now and capture (rather than a fixed dt time)
		times = np.arange(0,max(Tend,sim_dt*2),sim_dt)

		# Convert theta (th) into U = [ accX, accY ]
		att_U = theta_to_u(att_robot,att_theta)
		def_U = theta_to_u(def_robot,def_theta)

		# Simulate system
		att_states = integrate(att_robot,att_robot["x0"],att_U,times[1:],sim_dt)
		def_states = integrate(def_robot,def_robot["x0"],def_U,times[1:],sim_dt)

		# Interpolate to find the state at Tend (rather than times[-1])
		Tsample = max(Tend,sim_dt/3) 

		att_X = np.interp(Tsample, times, att_states[:,0])
		att_Y = np.interp(Tsample, times, att_states[:,1])
		def_X = np.interp(Tsample, times, def_states[:,0])
		def_Y = np.interp(Tsample, times, def_states[:,1])

		# Calculate the distance between the attacker and defender in the x- and y-axes
		eqns = (
			att_X - def_X, 
			att_Y - def_Y, 
			)
		return eqns

	def approx_equations(p):
		def_theta, Tend = p
		eqns = (
			(att_robot['x0'][0] + att_robot['x0'][2]*Tend + ((att_robot["acceleration_limit"]*Tend**2)/2)*np.cos(att_theta)) - (def_robot['x0'][0] + def_robot['x0'][2]*Tend + ((def_robot["acceleration_limit"]*Tend**2)/2)*np.cos(def_theta)), 
			(att_robot['x0'][1] + att_robot['x0'][3]*Tend + ((att_robot["acceleration_limit"]*Tend**2)/2)*np.sin(att_theta)) - (def_robot['x0'][1] + def_robot['x0'][3]*Tend + ((def_robot["acceleration_limit"]*Tend**2)/2)*np.sin(def_theta)),
			)
		return eqns

	# Initial conditions for approx equations come from inputs into function
	# tbh we probably don't need ot use this step and can just use those calcualted before
	def_theta_approx, Tend_approx = fsolve(approx_equations, (defender_action_guess, 3))
	if (0) : print("\t      Approx intercept theta %6.2f [ deg ] at t = %5.2f [ s ]" % (def_theta_approx*57.7, Tend_approx))

	# Solve using the full simulator
	def_theta, Tend =  fsolve(equations, (def_theta_approx, Tend_approx))
	if (0) : print("\t       Exact intercept theta %6.2f [ deg ] at t = %5.2f [ s ]" % (def_theta*57.7, Tend))

	return def_theta,Tend

def theta_to_u(robot,theta):
	return robot["acceleration_limit"]*np.array((np.cos(theta),np.sin(theta)))

def main():

	set_ic = False
	#set_ic = True
	if set_ic: 
		print("====\nUsing Fixed Initial Conditions\n====")
		# make sure this matches the teams match up in the param file
		initial_condition = np.array( [ \
			[   0.166,   0.675,   0.000,   0.000 ], \
			[   0.862,   0.852,  -0.000,   0.000 ] ]) 

		df_param = Param()
		df_param.update(initial_condition=initial_condition)

		# Set the goal
		df_param.goal = [0.6, 0.5, 0.   , 0.   ]

	else: 
		print("====\nUsing Random Initial Conditions\n====")
		df_param = Param()
		initial_condition = df_param.state

	print("\n\n====== Initial Conditions =====\n")
	print("goal: ( %.3f, %.3f )\n" % (df_param.goal[0],df_param.goal[1]))
	for ii in range(np.size(initial_condition,0)) :
		print(" [ %7.3f, %7.3f, %7.3f, %7.3f ], \ " % \
			(initial_condition[ii][0],initial_condition[ii][1],initial_condition[ii][2],initial_condition[ii][3]))

	print("\n\n====== Panagou Policy =====\n")
	pp = PanagouPolicy(df_param)

	print("\n\n====== Sim Init =====\n")
	pp.init_sim(initial_condition)

	print("\n\n====== Rollout =====\n")
	states = pp.rollout(initial_condition)

	plotter.plot_panagou(pp.R,pp.R_nom,pp.I,states,pp.param)
	
	# Save and open results
	filename = os.path.join("..","plots","panagou_2017.pdf")
	plotter.save_figs(filename)
	plotter.open_figs(filename)

	# run_sim(df_param)

if __name__ == '__main__':
	main()