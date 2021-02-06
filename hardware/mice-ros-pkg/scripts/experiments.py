#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

# ros package 
import rospy
from std_msgs.msg import String

# standard packages 
import sys
import time 
import numpy as np 
sys.path.append("/home/whoenig/projects/caltech/swarm_ai/code")
sys.path = ['/home/whoenig/projects/caltech/swarm_ai/hardware/mice-ros-pkg/scrips', '/opt/ros/noetic/lib/python3/dist-packages', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/home/whoenig/.local/lib/python3.8/site-packages', '/usr/local/lib/python3.8/dist-packages', '/usr/lib/python3/dist-packages', '/home/whoenig/projects/caltech/swarm_ai/code']

# my packages 
from cpp.buildRelease import mctscpp as mctscpp
from param import Param
from cpp_interface import create_cpp_policy, create_cpp_value, state_to_cpp_game_state, param_to_cpp_game
from learning_interface import global_to_local, local_to_global


# def d_mcts(param,ros_state,mctssettings,policy_dict_a,policy_dict_b,policy_a,policy_b,valuePredictor_a,valuePredictor_b):

#     state = ros_state_to_cpp_state(param,ros_state)
#     action = np.nan*np.zeros((state.shape[0],2))

#     g = param_to_cpp_game(param.robot_team_composition,param.robot_types,param.env_xlim,param.env_ylim,\
#         param.sim_dt,param.goal,param.rollout_horizon)
#     gs = state_to_cpp_game_state(state,"a",param.team_1_idxs,param.team_2_idxs)
#     gs.depth = 0
#     invalid_team_action = [np.nan*np.ones(2) for _ in range(param.num_nodes)]
#     team_action = list(invalid_team_action)

#     for i in range(2): # 2 teams 

#         if gs.turn == mctscpp.GameState.Turn.Attackers:
#             policy_dict = policy_dict_a
#             team_idx = param.team_1_idxs
#             my_policy = policy_a
#             other_policies = [policy_b]
#             valuePredictor = valuePredictor_a
#             team = 'a'
#         elif gs.turn == mctscpp.GameState.Turn.Defenders:
#             policy_dict = policy_dict_b
#             team_idx = param.team_2_idxs
#             my_policy = policy_b
#             other_policies = [policy_a]
#             valuePredictor = valuePredictor_b
#             team = 'b'

#         for robot_idx in team_idx: 
#             if not np.isfinite(state[robot_idx,:]).all(): # non active robot 
#                 continue
#             o_a,o_b,goal = global_to_local(state,param,robot_idx)
#             state_i, robot_team_composition_i, self_idx, team_1_idxs_i, team_2_idxs_i = \
#                 local_to_global(param,o_a,o_b,goal,team)    
#             game_i = param_to_cpp_game(robot_team_composition_i,param.robot_types,param.env_xlim,param.env_ylim,\
#                 param.sim_dt,param.goal,param.rollout_horizon)
#             gamestate_i = state_to_cpp_game_state(state_i,team,team_1_idxs_i,team_2_idxs_i)
#             gamestate_i.depth = 0
#             mctsresult = mctscpp.search(game_i, gamestate_i, \
#                 my_policy,
#                 other_policies,
#                 valuePredictor,
#                 mctssettings)
           
#             if mctsresult.success: 
#                 action_i = mctsresult.bestAction
#                 action[robot_idx,:] = action_i[self_idx]
#             else: 
#                 action[robot_idx,:] = np.zeros(2) 

#         success = g.step(gs,action,gs)

#     return action 

def d_mcts_i(param,state,robot_idx,mctssettings,policy_dict_a,policy_dict_b,policy_a,policy_b,valuePredictor_a,valuePredictor_b):

    action_i = np.zeros(2)

    if robot_idx in param.team_1_idxs:
        policy_dict = policy_dict_a
        team_idx = param.team_1_idxs
        my_policy = policy_a
        other_policies = [policy_b]
        valuePredictor = valuePredictor_a
        team = 'a'
    elif robot_idx in param.team_2_idxs:
        policy_dict = policy_dict_b
        team_idx = param.team_2_idxs
        my_policy = policy_b
        other_policies = [policy_a]
        valuePredictor = valuePredictor_b
        team = 'b'

    if np.isfinite(state[robot_idx,:]).all(): # active robot 
        
        o_a,o_b,goal = global_to_local(state,param,robot_idx)
        state_i, robot_team_composition_i, self_idx, team_1_idxs_i, team_2_idxs_i = \
            local_to_global(param,o_a,o_b,goal,team)    
        game_i = param_to_cpp_game(robot_team_composition_i,param.robot_types,param.env_xlim,param.env_ylim,\
            param.sim_dt,param.goal,param.rollout_horizon)
        gamestate_i = state_to_cpp_game_state(state_i,team,team_1_idxs_i,team_2_idxs_i)
        gamestate_i.depth = 0
        mctsresult = mctscpp.search(game_i, gamestate_i, \
            my_policy,
            other_policies,
            valuePredictor,
            mctssettings)
       
        if mctsresult.success: 
            action = mctsresult.bestAction
            action_i = action[self_idx]

    return action_i 


def ros_state_to_cpp_state(param,ros_state):
    cpp_state = ros_state
    return cpp_state


def get_ros_state():

    # initial_condition = np.array( [ \
        # [   0.166,   0.675,   0.000,   0.000 ], \
        # [   0.862,   0.852,  -0.000,   0.000 ] ]) 

    initial_condition = np.array( [ \
        [   0.10,   0.90,   0.000,   0.000 ], \
        [   0.20,   0.35,   0.000,   0.000 ], \
        # [   0.80,   0.75,   0.000,   0.000 ], \
        [   0.85,   0.10,   0.000,   0.000 ] ]) 

    # initial_condition = np.array( [ \
    #     [   0.17646,   0.10618,   0.000,   0.000 ], \
    #     [   0.18721,   0.8071,   0.000,   0.000 ], \
    #     [   0.77,   0.59908,   0.000,   0.000 ], \
    #     [   0.89076,   0.87235,   0.000,   0.000 ] ]) 

    state = initial_condition

    return state 


def make_policy_dicts(param):

    policy_dict_a = param.policy_dict
    policy_dict_b = param.policy_dict 
    param.policy_dict_a = policy_dict_a 
    param.policy_dict_b = policy_dict_b 

    return policy_dict_a, policy_dict_b


def get_mcts_settings(param):
    mctssettings = mctscpp.MCTSSettings()
    mctssettings.num_nodes = param.policy_dict["mcts_tree_size"]
    mctssettings.Cp = param.policy_dict["mcts_c_param"]
    mctssettings.pw_C = param.policy_dict["mcts_pw_C"]
    mctssettings.pw_alpha = param.policy_dict["mcts_pw_alpha"]
    mctssettings.beta1 = param.policy_dict["mcts_beta1"]
    mctssettings.beta3 = param.policy_dict["mcts_beta3"]
    mctssettings.export_tree = False 
    mctssettings.export_root_reward_over_time = False
    return mctssettings 


def load_heuristics(policy_dict_a,policy_dict_b):

    policy_a = create_cpp_policy(policy_dict_a, 'a')
    policy_b = create_cpp_policy(policy_dict_b, 'b')

    valuePredictor_a = create_cpp_value(None)
    if policy_dict_a["sim_mode"] in ["MCTS","D_MCTS"]:
        if policy_dict_a["path_glas_model_a"] is not None:
            model_num = int(os.path.basename(policy_dict_a["path_glas_model_a"])[1])
        if policy_dict_a["path_value_fnc"] is not None:
            valuePredictor_a = create_cpp_value(policy_dict_a["path_value_fnc"])

    valuePredictor_b = create_cpp_value(None)
    if policy_dict_b["sim_mode"] in ["MCTS","D_MCTS"]:
        value_path_b = None 
        if policy_dict_b["path_glas_model_b"] is not None:
            model_num = int(os.path.basename(policy_dict_b["path_glas_model_b"])[1])
        if policy_dict_b["path_value_fnc"] is not None:
            valuePredictor_b = create_cpp_value(policy_dict_b["path_value_fnc"])

    return policy_a, policy_b, valuePredictor_a, valuePredictor_b


def get_robot_idx():
    return 0 


def experiments():

    # ros stuff 
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # hz

    # define game 
    param = Param()
    mctssettings = get_mcts_settings(param)
    policy_dict_a, policy_dict_b = make_policy_dicts(param) 
    policy_a, policy_b, valuePredictor_a, valuePredictor_b = load_heuristics(policy_dict_a,policy_dict_b)
    
    while not rospy.is_shutdown():

        ros_state = get_ros_state()
        cpp_state = ros_state_to_cpp_state(param,ros_state)
        robot_idx = get_robot_idx()

        start = time.time()
        action = d_mcts_i(param,cpp_state,robot_idx,mctssettings,policy_dict_a,policy_dict_b,policy_a,policy_b,valuePredictor_a,valuePredictor_b)
        duration = time.time() - start 

        timeit_str = "exec time: {}".format(duration)
        action_str = "action: {}".format(action)

        rospy.loginfo(timeit_str)
        pub.publish(timeit_str)
        rospy.loginfo(action_str)
        pub.publish(action_str)

        rate.sleep()


if __name__ == '__main__':
    try:
        experiments()
    except rospy.ROSInterruptException:
        pass
