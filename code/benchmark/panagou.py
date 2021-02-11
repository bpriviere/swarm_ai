#!/usr/bin/env python3

import numpy as np 
import sys
import os 
import math
from scipy.optimize import fsolve, minimize, linear_sum_assignment
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

'''
TODO List
'''

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
			self.times = np.arange(0,1.5*max(self.terminal_times),self.param.sim_dt)	# Extend time so that collisions can be calculated
																						# Our survival time might be longer than it takes for
																						# us to reach the goal if there were no attacker

			# We base our next iteration initial guess from our previous guess
			# For the first iteration, this won't exist so we need to make it
			if not hasattr(self, 'best_actions') :
				self.best_actions = estimate_actions(self.param,self.robots)
					
			# Calculate the best actions
			self.best_actions = find_best_actions(self.param,self.robots,self.best_actions)

			# Calculate who each defender should target
			self.matching2 = calculate_matching_optimal(self.best_actions,self.robots,self.param)

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

			
			if (robot_dead(state[i_robot])) :
				# Robot is dead, we can consider it done
				done.append(i_robot)

			if i_robot not in done: 

				if robot["team"] == "a":     # attackers
					if self.matching2[i_robot] == None:
						# Attacker will win, go straight to the goal
						#print('Attacker wins')
						actions[i_robot,:] = theta_to_u(robot,self.theta_noms[i_robot])

						# Update the terminal time for the robot
						terminal_times[0,i_robot] = self.terminal_times[i_robot]

					else:
						#print("Attacker looses")
						# Calculate matching robot
						j_robot = self.matching2[i_robot]

						# We already know the best angles to accelerate at for 
						# both the attacker and defender, so we can extract those
						att_theta = self.best_actions[i_robot,j_robot][2]
						def_theta = self.best_actions[i_robot,j_robot][3]

						# Put the action into [ accX , accY ] space
						actions[i_robot,:] = theta_to_u(self.robots[i_robot],att_theta)
						actions[j_robot,:] = theta_to_u(self.robots[j_robot],def_theta) 

						# Update the terminal time for the robots
						terminal_times[0,i_robot] = self.best_actions[i_robot,j_robot][4]
						terminal_times[0,j_robot] = self.best_actions[i_robot,j_robot][4]

						# Mark j_robot as calculated
						done.append(j_robot)

				elif robot["team"] == "b":   # defenders
					# Match not found for attacker
					if not i_robot in self.matching2.values():
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

		#t_end = 2.5  # Sim_end
		
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
					if (0) : print("A%d > " % i_robot, end='')

					# Calculate distances to goal
					distance_to_goal    = math.sqrt((states[-1][i_robot][0]-self.param.goal[0])**2 + (states[-1][i_robot][1]-self.param.goal[1])**2)
					if (0) : print("(goal): %6.3f, " % (distance_to_goal),end='')

					# Calculate distances to all attackers
					for j_robot in self.param.team_2_idxs: 
						distance_to_this_attacker = math.sqrt((states[-1][i_robot][0]-states[-1][j_robot][0])**2 + (states[-1][i_robot][1]-states[-1][j_robot][1])**2)
						distance_to_capture  = np.append(distance_to_capture, np.array([distance_to_this_attacker]), axis=0)
						if (0) : print("(D%d): %6.3f, " % (j_robot,distance_to_this_attacker), end='')

				else :
					# Robot is defender so we don't need to calculate game-state stuff
					distance_to_capture = 1e6
					pass

				# Step the state
				if (distance_to_goal < robot["goal_radius"]) : 
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
					if (robot["team"] == "a" and 0) : print("        || ", end='')
					state[i_robot,:] = step(robot, states[-1][i_robot,:], actions[i_robot,:], self.param.sim_dt)
				
				if (game_over[i_robot]) :
					# Game over for this robot, put it back to the previous state
					state[i_robot,:] = states[-1][i_robot,:]

				if (0): print(" <> ",end='')

			# Add the current state to the state matrix
			states.append(state)
			print("\n",end='')
		states = np.array(states)
		return states 

def calculate_matching_optimal(best_actions,robots,param) :
	# Calculates which attacker each defender should target
	# The cost element is the distance to goal for the attacker 
	# on capture.
	#
	# Only works for equal number of attackers and defenders
	
	print_debug = 0

	matching = dict()

	# Pre-allocate cost matrix
	n_att = len(param.team_1_idxs)
	n_def = len(param.team_2_idxs)
	cost_matrix = np.empty((n_att,n_def))
	raw_cost    = np.empty((n_att,n_def))

	# Fill cost matrix
	for ii in range(n_att) :
		i_robot = param.team_1_idxs[ii]

		# Pre-allocate 'None' the matching dict() in case no defender is assigned
		matching[ii] = None

		for jj in range(n_def) :
			j_robot = param.team_2_idxs[jj]

			cost_element = best_actions[i_robot,j_robot][5]

			if cost_element < max(robots[i_robot]['tag_radius'],0.001) :
				# We can't catch this attacker, put a high price on chasing him
				cost_matrix[ii,jj] = 1e10
				raw_cost[ii,jj] = 0.0

			else : 
				# Add element to the cost matrix - rows = attackers (jobs), columns = defenders (workers)
				# We want to maximise the distance rather than minimise it as linear_sum_assignment does,
				# so take the distance away from a semi-large number (100)
				cost_matrix[ii,jj] = 100 - cost_element
				raw_cost[ii,jj] = cost_element

	# Solve the cost matrix
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	total_cost = raw_cost[row_ind, col_ind].sum() 

	# Create the matching storage by looping through defenders
	for ii in range(len(col_ind)) :
		# Calculate robot numbers of the match
		def_idx = param.team_2_idxs[col_ind[ii]]
		att_idx = param.team_1_idxs[ii]

		# Check if we should be chasing this attacker or not
		dist = best_actions[att_idx,def_idx][5]

		if (dist > max(robots[i_robot]['tag_radius'],0.001)) :
			matching[att_idx] = def_idx
			if (print_debug) : print("[ Def %d ] > [ Att %d ] ( %7.4f), " % (def_idx,att_idx,dist), end="")

		else :
			matching[att_idx] = None
			if (print_debug) : print("[ Def %d ] > [ None  ], " % (def_idx), end="")

	if (print_debug) : print(" Total: %.4f" % total_cost)

	# Each defender is matched
	return matching

def estimate_actions(param,robots) :
	# Provides a quick estimate of what actions a robot should take
	# This is far from optimal and is used as an initial condition for later

	# Pre-allocate matricies
	best_actions = dict()
	best_actions[0,0] = "att_robot, def_robot, att_theta, def_theta, t_end, dist2goal"

	for i_robot in param.team_1_idxs : 
		# Assign attacking robot
		att_robot = robots[i_robot]

		for j_robot in param.team_2_idxs :
			# Assign defending robot
			def_robot = robots[j_robot]

			# Calculate stuff
			att_theta_init = np.arctan2(param.goal[1]-att_robot["x0"][1],param.goal[0]-att_robot["x0"][0]) 
			def_theta_init = np.arctan2(att_robot["x0"][1]-def_robot["x0"][1],att_robot["x0"][0]-def_robot["x0"][0])
			t_init = np.linalg.norm(att_robot["x0"][0:2] - param.goal[0:2]) / att_robot["speed_limit"]

			# Add to the dict
			best_actions[i_robot,j_robot] = (i_robot, j_robot, att_theta_init, def_theta_init, t_init, 0.0)

	return best_actions

def find_best_actions(param,robots,prev_best) :
	# Finds the best attacker action to minimise distance to the goal
	# Calculates the best defender action based on this attacker action
	#
	# Outputs an array with
	#     defender_actions[i_robot,j_robot] = [att_id, def_id1, att_theta, def_theta, t_end, dist2goal ;
	#                                         att_id, def_id2, att_theta, def_theta, t_end, dist2goal ; 
	#                                         ...
	#                                         att_id, def_idN, att_theta, def_theta, t_end, dist2goal ]

	def func_dist_to_goal(p,def_theta_guess) :
		temp = p
		att_theta = temp[0]

		# Calculate the defender's best theta and corresponding time to capture for the given new attacker input
		def_theta_guess,t_capture = find_best_intercept(att_robot,def_robot,att_theta,def_theta_guess,param.sim_dt)

		# Integrate the attacker's state with the new theta guess
		U = theta_to_u(att_robot,att_theta)
		states = integrate(att_robot, att_robot["x0"], U, t_capture)

		# Calculate the distance to the goal
		dist2goal = np.linalg.norm(states[0:2] - param.goal[0:2])

		return dist2goal

	print_debug = 0
	attacker_direct_to_goal = 1

	if (print_debug) : print("\ndirect_to_goal()")
	def_theta_guess = 0

	# Pre-allocate matricies
	best_actions = dict()
	best_actions[0,0] = "att_robot, def_robot, att_theta, def_theta, t_end, dist2goal"

	if (print_debug) : print("")

	# Loop through each attacker
	for i_robot in param.team_1_idxs: 
		# Assign attacking robot
		att_robot = robots[i_robot]

		# If robot is alive, find the nominal solution
		if not (robot_dead(att_robot["x0"])) :
			# Check time to goal for attacker using nominal solution
			att_theta_nom, att_terminal_time = find_nominal_soln(param,att_robot,np.array(att_robot["x0"]))

		for j_robot in param.team_2_idxs:
			# Assign defender robot
			def_robot = robots[j_robot]

			# If robot is dead / at the goal, we don't need to do any of this
			if robot_dead(att_robot["x0"]) or robot_dead(def_robot["x0"]) :
				att_theta_best = 0.0
				def_theta_best = 0.0
				t_end = 0.0
				dist2goal = 0.0  # even if dead, pretend robot is at the goal to take it out of the matching equation

			else :
				# Check the time to capture for defender if attacker is using nominal solution
				# We use the previous estimate for the best capture if available

				# Extract the previous best data
				att_theta_prev = prev_best[i_robot,j_robot][2]
				def_theta_prev = prev_best[i_robot,j_robot][3]
				t_end_prev     = prev_best[i_robot,j_robot][4]

				# Calculate how long it will take for this defender to capture the attacker
				def_theta_guess = np.arctan2(att_robot["x0"][1]-def_robot["x0"][1],att_robot["x0"][0]-def_robot["x0"][0])
				#def_theta_guess = def_theta_prev	
				def_theta_nom, t_end = find_best_intercept(att_robot,def_robot,att_theta_nom,def_theta_guess,param.sim_dt)

				if (att_terminal_time < t_end) :
					# Attacker will win, use the nominal attacker results
					att_theta_best = att_theta_nom
					def_theta_best = 0.0
					dist2goal = 0.0
					t_end = att_terminal_time

				else :
					# Defender should be able to intercept attacker,
					# work out what the attacker will do
					if (attacker_direct_to_goal) :
						# Go straight to the goal
						att_theta_best = att_theta_nom
						def_theta_best = def_theta_nom

						# We know what the attacker will do from before,
						# so we don't need to calculate that (and t_end) again

					else : 
						# Try to minimise the distance to the goal upon capture
						# This is temporamental to say the least...
						res = minimize(func_dist_to_goal, att_theta_prev, args=(def_theta_prev), options={'maxiter': 11})
						if not res.success :
							# Iteration thing didn't work, let's just got with the nominal solution
							# for the attacker and the previous value for the defender
							att_theta_best = att_theta_nom
							def_theta_best = def_theta_nom

						else : 
							# We have a solution, roll with it
							att_theta_best = res["x"][0]

							# Simulate the results to get the results we need (from the defender's side)
							def_theta_best,t_end = find_best_intercept(att_robot,def_robot,att_theta_best,def_theta_guess,param.sim_dt)

					# Calculate the attacker poition at t_end
					U = theta_to_u(att_robot,att_theta_best)
					att_states = integrate(att_robot, att_robot["x0"], U, t_end)

					# Calculate distance to goal
					dist2goal = np.linalg.norm(att_states[0:2] - param.goal[0:2])

			# Store the results
			best_actions[i_robot,j_robot] = (i_robot, j_robot, att_theta_best, def_theta_best, t_end, dist2goal)

			# Debug printing
			if (print_debug) :
				print("\t[ Att %d ] theta: %7.2f [ deg ], [ Def %d ] theta: %7.2f [ deg ], t_end: %.2f [ s ], dist2goal: %.6f [ m ]" \
					% (i_robot,att_theta_best*57.7,j_robot,def_theta_best*57.7,t_end,dist2goal))

	return best_actions

def find_nominal_solns(param,robots):
	# find longest possible time 
	theta_noms, terminal_times = [], []
	for robot in robots: 
		state = np.array(robot["x0"])

		if robot_dead(state) :
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

	if robot_dead(X0) :
		# Robot dead, return action
		return U0

	# run the step function to get new state
	X = step(robot, X0, U0, dt)

	# back-calculate what the actual control input was
	U = (X[2:]-X0[2:]) / dt

	# Return the new control input
	return U

def robot_dead(state) :
	# Checks if the robot is dead
	if np.any(np.isfinite(state)) :
		return False
	else :
		return True

	# No idea how we got here, probably safest to kill the robot...
	return True

def step(robot, x0, U, dt):  # Change to an X and Y acceleration
	# x0 = [ posX, posY, Vx, Vy ]
	#  U = [ accX, accY ]
	# 
	# The inputs to this function changed so hopefully nothing broke

	# Check inputs are ok
	if not x0.size == 4: print("X0 Input Incorrect Size")
	if not  U.size == 2: print("U  Input Incorrect Size")


	if robot_dead(x0) :
		# don't both with all this stuff, just give it the x0 back
		return x0

	 
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

def find_nominal_soln(param,robot,state):

	def equations(p):
		th, Tend = p

		# Convert theta (th) into U = [ accX, accY ]
		U = theta_to_u(robot,th)

		# Simulate system
		states = integrate(robot,state,U,Tend)

		# Extract useful information
		eqns = (
			np.linalg.norm(states[0] - param.goal[0]), 
			np.linalg.norm(states[1] - param.goal[1]), 
			)
		return eqns

	print_debug = 0

	# Take a better guess at the initial conditions
	#print("Nominal Guess")
	theta_guess = np.arctan2(param.goal[1]-state[1],param.goal[0]-state[0]) 
	dist = np.linalg.norm(robot["x0"][0:2] - param.goal[0:2])
	Tend_guess = dist / robot["speed_limit"]

	#if Tend_guess < 1.0 :
	#	# We're close to the goal, everything is going to stuff up
	#	# so hack in the "direct to goal, infinite turning" solution
	#	return theta_guess,Tend_guess

	# Calculate the best solution using the full dynamics
	#print("Nominal Exact")
	theta_exact, Tend_exact =  fsolve(equations, (theta_guess, Tend_guess))

	# Debugging
	if (print_debug) : 
		print("Guess: %7.2f [ deg ], %.2f  [ s ], Approx: %7.2f [ deg ], %.2f  [ s ], Exact: %7.2f [ deg ], %.2f  [ s ]" % \
			(theta_guess*57.7, Tend_guess, \
			theta_approx*57.7, Tend_approx, \
			theta_exact*57.7, Tend_exact))


	return theta_exact,Tend_exact

def find_best_intercept(att_robot,def_robot,att_theta,defender_action_guess,sim_dt) :
	# Calculates the trajectory to minimum-time-to-intercept for an attacker/defender 
	# pair when the attacker's trajectory is known.

	def equations(p):
		def_theta, Tend = p
	
		# Convert theta (th) into U = [ accX, accY ]
		att_U = theta_to_u(att_robot,att_theta)
		def_U = theta_to_u(def_robot,def_theta)

		# Calculate final state
		att_state = integrate(att_robot,att_robot["x0"],att_U,Tend) 
		def_state = integrate(def_robot,def_robot["x0"],def_U,Tend) 

		# Calculate the distance between the attacker and defender in the x- and y-axes
		eqns = (
			np.linalg.norm(att_state[0] - def_state[0]),
			np.linalg.norm(att_state[1] - def_state[1]),
			)
		return eqns

	print_debug = 0

	if (print_debug) : print("find_best_intercept()")

	# Check that we are not already within capture radius of the robot
	dist = np.linalg.norm(att_robot["x0"][0:2] - def_robot["x0"][0:2])
	if (dist < def_robot["tag_radius"]) :
		# This robot is captured, don't bother doing calcs for it
		def_theta = 0
		Tend = 0

	else :
		# Initial conditions for approx equations come from inputs into function
		# tbh we probably don't need the use this step and can just use those calcualted before
		dist2att = np.linalg.norm(att_robot["x0"][0:2] - def_robot["x0"][0:2])
		angle2att = np.arctan2(att_robot["x0"][1]-def_robot["x0"][1],att_robot["x0"][0]-def_robot["x0"][0])
		t_end_guess = dist2att / (att_robot["speed_limit"] + def_robot["speed_limit"])

		# In some cases, we are never able to catch the attacker as we've started behind them,
		# so we need to catch this case.  This assumption will be no good for assymetric cases
		if (abs(angle2att - att_theta) < 90/57.7) :
			# We're behind the robot, just go for it
			def_theta_approx = angle2att
			def_theta = angle2att

			Tend = 10

		else : 
			# Solve using the full simulator
			def_theta, Tend =  fsolve(equations, (defender_action_guess, t_end_guess), maxfev=21)

		if (print_debug) : print("\t       Guess intercept theta %7.2f [ deg ] at t = %5.2f [ s ]" % (defender_action_guess*57.7, t_end_guess))
		if (print_debug) : print("\t      Approx intercept theta %7.2f [ deg ] at t = %5.2f [ s ]" % (def_theta_approx*57.7, Tend_approx))
		if (print_debug) : print("\t       Exact intercept theta %7.2f [ deg ] at t = %5.2f [ s ]" % (def_theta*57.7, Tend))

	return def_theta,Tend

def integrate(robot,x0,U,t_end) :
	# State = [ posX posY Vx Vy ]
	# U = [ aX aY ]

	# Magnitude of acceleration
	a = np.linalg.norm(U)

	# Magnitude of starting velocity
	v0 = np.linalg.norm(x0[2:])

	# time at which max velocity is reached
	if a > 0.0 :
		t_maxV = (robot["speed_limit"]-v0) / a
	else :
		t_maxV = np.inf

	if t_end <= t_maxV :
		# Final positions (constant acceleration)
		posX_final = x0[0] + x0[2]*t_end + 0.5*U[0]*t_end**2
		posY_final = x0[1] + x0[3]*t_end + 0.5*U[1]*t_end**2

		# Final velocities 
		Vx_final = x0[2] + U[0]*t_end
		Vy_final = x0[3] + U[1]*t_end

	else :
		# Calculate the final velocities
		Vx_final = robot["speed_limit"] * U[0] / a
		Vy_final = robot["speed_limit"] * U[1] / a

		# Constant acceleration phase
		posX_final = x0[0] + x0[2]*t_maxV + 0.5 *U[0]*t_maxV**2
		posY_final = x0[1] + x0[3]*t_maxV + 0.5 *U[1]*t_maxV**2

		# Constant velocity phase
		posX_final += Vx_final * (t_end - t_maxV)
		posY_final += Vy_final * (t_end - t_maxV)

	state = [posX_final, posY_final, Vx_final, Vy_final]


	return state

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

		initial_condition = np.array( [ \
			[   0.10,   0.90,   0.000,   0.000 ], \
			[   0.20,   0.35,   0.000,   0.000 ], \
			[   0.80,   0.75,   0.000,   0.000 ], \
			[   0.85,   0.10,   0.000,   0.000 ] ]) 

		initial_condition = np.array( [ \
			[   0.394,   1.113,  -0.000,   0.000 ], \
			[   1.731,   1.080,   0.000,  -0.000 ] ]) 


		df_param = Param()
		df_param.update(initial_condition=initial_condition)

		# Set the goal
		df_param.goal = [1.2, 0.7, 0.   , 0.   ]

	else: 
		print("====\nUsing Random Initial Conditions\n====")
		df_param = Param()
		initial_condition = df_param.state

	print("\n\n====== Initial Conditions =====\n")
	print("goal: ( %.3f, %.3f )\n" % (df_param.goal[0],df_param.goal[1]))
	for ii in range(np.size(initial_condition,0)) :
		print("[ %7.3f, %7.3f, %7.3f, %7.3f ], \ " % \
			(initial_condition[ii][0],initial_condition[ii][1],initial_condition[ii][2],initial_condition[ii][3]))

	print("\n\n====== Panagou Policy =====\n")
	pp = PanagouPolicy(df_param)

	print("\n\n====== Sim Init =====\n")
	pp.init_sim(initial_condition)

	print("\n\n====== Rollout =====\n")
	states = pp.rollout(initial_condition)

	plotter.plot_panagou(states,pp.param)
	
	# Save and open results
	filename = os.path.join("..","plots","panagou_2017.pdf")
	plotter.save_figs(filename)
	plotter.open_figs(filename)

	# run_sim(df_param)

if __name__ == '__main__':
	main()
