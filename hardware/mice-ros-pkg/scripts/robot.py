#!/usr/bin/env python3

# standard packages 
import sys
import time 
import yaml
import numpy as np 
import os
# sys.path.append("/home/whoenig/projects/caltech/swarm_ai/code")
sys.path.append("/home/ben/projects/swarm_ai/code")
import random

# ROS packages
import rospy
from tf import TransformListener
from pycrazyswarm.crazyflie import Crazyflie
from visualization_msgs.msg import Marker

# my packages 
from cpp.buildRelease import mctscpp as mctscpp
from param import Param
from cpp_interface import create_cpp_policy, create_cpp_value, state_to_cpp_game_state, param_to_cpp_game
from learning_interface import global_to_local, local_to_global


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

def get_status(param,robot_idx,cpp_state):
    # cpp_state num agents x 4 nd array 

    if robot_idx in param.team_1_idxs: 
        # check tagged
        for defender_robot_idx in param.team_2_idxs: 
            dist = np.linalg.norm(cpp_state[robot_idx,0:2] - cpp_state[defender_robot_idx,0:2])
            if dist < param.robot_types["standard_robot"]["tag_radius"] : 
                return "Tagged" 

        # check reached goal
        dist = np.linalg.norm(cpp_state[robot_idx,0:2] - param.goal[0:2])
        if dist < param.robot_types["standard_robot"]["goal_radius"] : 
            return "ReachedGoal"

    # check bounds 
    speed_lim = param.robot_types["standard_robot"]["speed_limit"]
    # speed_lim = VEL_LIMIT
    speed = np.linalg.norm(cpp_state[robot_idx,3:5])
    if  cpp_state[robot_idx,0] < param.env_xlim[0] or \
        cpp_state[robot_idx,0] > param.env_xlim[1] or \
        cpp_state[robot_idx,1] < param.env_ylim[0] or \
        cpp_state[robot_idx,1] > param.env_ylim[1] or \
        speed > speed_lim:
        return "OutOfBounds"

    return "Active"



def ros_state_to_cpp_state(param,ros_state):
    cpp_state = ros_state
    return cpp_state


def get_ros_state(tf, cfids, last_state, dt, VEL_LIMIT):

    # result is x,y,vx,vy (one row per robot)
    result = np.zeros((len(cfids), 4))
    for i, cfid in enumerate(cfids):
        # get latest transform
        position, quaternion = tf.lookupTransform("/world", "/cf" + str(cfid), rospy.Time(0))
        latestCommonTime = tf.getLatestCommonTime("/world", "/cf" + str(cfid))
        d = rospy.Time.now() - latestCommonTime
        if d.to_sec() < 2 and position[2] > 0.2:
            result[i,0:2] = position[0:2]

            if last_state is not None:
                v = np.clip((result[i,0:2] - last_state[i,0:2]) / dt, -VEL_LIMIT, VEL_LIMIT)
                # v = (result[i,0:2] - last_state[i,0:2]) / dt 
                alpha = 0.1
                result[i,2:4] = (1-alpha) * v + alpha * last_state[i,3:5]
            else:
                result[i,2:4] = [0,0]
        else:
            result[i] = np.nan

    print('ros_state', result)
    return result


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


def run(cf, tf, cfids, robot_idx):

    # if not robot_idx in [0,1] :
    #     return 

    # some tuning parameters
    HEIGHT = 0.5
    VEL_LIMIT = 0.25
    # VEL_LIMIT = 0.5
    # VEL_LIMIT = 1.0
    ACC_LIMIT = 0.5
    SEED = 3  

    # fix the seed so that all nodes compute the same initial condition
    random.seed(SEED)
    np.random.seed(SEED)

    # define game 
    param = Param()
    mctssettings = get_mcts_settings(param)
    policy_dict_a, policy_dict_b = make_policy_dicts(param) 
    policy_a, policy_b, valuePredictor_a, valuePredictor_b = load_heuristics(policy_dict_a,policy_dict_b)

    pub = rospy.Publisher("/visualization_marker", Marker, queue_size=1)
    # print(param.state)
    # exit()

    cf.setParam("ring/effect", 7) # enable solid color LED ring
    cf.takeoff(0.3 + 0.2 * robot_idx, 2.0)
    time.sleep(2.0)

    # first robot visualizes goal
    if robot_idx == 0:

        marker = Marker()
        marker.header.frame_id = "world"
        marker.ns = "goal"
        marker.id = 0
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = param.robot_types["standard_robot"]["goal_radius"]
        marker.scale.y = param.robot_types["standard_robot"]["goal_radius"]
        marker.scale.z = param.robot_types["standard_robot"]["goal_radius"]
        marker.color.a = 0.2
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.pose.position.x = param.goal[0]
        marker.pose.position.y = param.goal[1]
        marker.pose.position.z = HEIGHT
        marker.pose.orientation.w = 1.0
        pub.publish(marker)


    # define colors 
    if robot_idx in param.team_1_idxs:
        cf.setLEDColor(0,0,1) # blue == attacker
    elif robot_idx in param.team_2_idxs:
        cf.setLEDColor(1,0,0) # red == defender

    x_des = np.array([
        param.state[robot_idx][0],  # x
        param.state[robot_idx][1],  # y
        0.3 + 0.2 * robot_idx,                 # z
        0,                      # vx
        0,                      # vy
        0,                      # vz
    ])

    cf.goTo(x_des[0:3], 0, 5)
    time.sleep(5)
    
    x_des[2] = HEIGHT 
    cf.goTo(x_des[0:3], 0, 2)
    time.sleep(2)

    dt = 0.1
    rate = rospy.Rate(1/dt) # hz
    ros_state = None

    durations = [] 


    while not rospy.is_shutdown():

        ros_state = get_ros_state(tf, cfids, ros_state, dt, VEL_LIMIT)
        cpp_state = ros_state_to_cpp_state(param,ros_state)

        # check status
        status = get_status(param,robot_idx,cpp_state)

        # print("robot idx: {}, x_curr: {}, status: {}".format(robot_idx,cpp_state[robot_idx,:],status))

        if status == "ReachedGoal":
            cf.setLEDColor(0,1,0) # green
            break
        elif status in ["Tagged","OutOfBounds"]:
            # cf.setParam("ring/effect", 0) # disable LED ring
            break
        # elif status == "Tagger":
        #     time.sleep(5)

        start = time.time()
        action = d_mcts_i(param,cpp_state,robot_idx,mctssettings,policy_dict_a,policy_dict_b,policy_a,policy_b,valuePredictor_a,valuePredictor_b)
        duration = time.time() - start 

        durations.append(duration)

        print('duration',duration)
        print('mean duration',np.mean(np.array(durations)))
        print('std duration', np.std(np.array(durations)))

        # timeit_str = "exec time: {}".format(duration)
        # action_str = "action: {}".format(action)
        # rospy.loginfo(timeit_str)
        # rospy.loginfo(action_str)

        # propagate desired state
        action = action * np.min((ACC_LIMIT/np.linalg.norm(action),1))
        acc = np.array([action[0],action[1],0])
        x_des[0:2] = x_des[0:2] + x_des[3:5] * dt
        x_des[3:5] = np.clip(x_des[3:5] + acc[0:2] * dt, -VEL_LIMIT, VEL_LIMIT)


        cf.cmdFullState(x_des[0:3], x_des[3:6], acc, yaw=0, omega=[0,0,0])
        # print(x_des)

        rate.sleep()

    # drop/land the robot

    # # option 1: aggressive motor cut
    cf.cmdStop()

    # option 2: "fast" landing
    # cf.notifySetpointsStop()
    # cf.land(0, 3.0)


if __name__ == '__main__':
    rospy.init_node("CrazyflieDistributed", anonymous=True)

    with open(rospy.get_param("crazyflies_yaml"), 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    tf = TransformListener()

    cf = None
    cfids = []
    my_robot_idx = None
    for crazyflie in cfg["crazyflies"]:
        cfid = int(crazyflie["id"])
        cfids.append(cfid)
        if cfid == rospy.get_param("~cfid"):
            initialPosition = crazyflie["initialPosition"]
            cf = Crazyflie(cfid, initialPosition, tf)
            my_robot_idx = len(cfids) - 1
        # Make sure we have this cf in the tf tree
        tf.waitForTransform("/world", "/cf" + str(cfid), rospy.Time(0), rospy.Duration(5))

    if cf is None:
        exit("No CF with required ID found!")

    run(cf, tf, cfids, my_robot_idx)

    # rospy.sleep(2.5)
    # cf.land(0.02, 2.0)
