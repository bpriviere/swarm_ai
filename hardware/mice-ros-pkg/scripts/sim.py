#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg

import yaml
import numpy as np
from crazyflie_driver.srv import *
from crazyflie_driver.msg import TrajectoryPolynomialPiece, FullState, Position, VelocityWorld
from pycrazyswarm.crazyflieSim import TimeHelper, Crazyflie

# Z = 1.0

class TimeHelperROS(TimeHelper):
    def __init__(self, vis, dt, writecsv, disturbanceSize):
        super().__init__(vis, dt, writecsv, disturbanceSize)
        self.br = tf2_ros.TransformBroadcaster()
        self.transform = geometry_msgs.msg.TransformStamped()
        self.transform.header.frame_id = "world"
        self.transform.transform.rotation.x = 0
        self.transform.transform.rotation.y = 0
        self.transform.transform.rotation.z = 0
        self.transform.transform.rotation.w = 1

    def step(self, duration):
        super().step(duration)
        self.transform.header.stamp = rospy.Time.now()
        for cf in self.crazyflies:
            cfid = cf.id
            x, y, z = cf.position()
            self.transform.child_frame_id = "/cf" + str(cfid)
            self.transform.transform.translation.x = x
            self.transform.transform.translation.y = y
            self.transform.transform.translation.z = z
            self.br.sendTransform(self.transform)


class CrazyflieROS(Crazyflie):
    def __init__(self, id, initialPosition, timeHelper):
        super().__init__(id, initialPosition, timeHelper)

        prefix = "/cf" + str(id)
        rospy.Service(prefix + '/set_group_mask', SetGroupMask, self.handle_set_group_mask)
        rospy.Service(prefix + '/takeoff', Takeoff, self.handle_takeoff)
        rospy.Service(prefix + '/land', Land, self.handle_land)
        rospy.Service(prefix + '/go_to', GoTo, self.handle_go_to)
        rospy.Service(prefix + '/upload_trajectory', UploadTrajectory, self.handle_upload_trajectory)
        rospy.Service(prefix + '/start_trajectory', StartTrajectory, self.handle_start_trajectory)
        rospy.Service(prefix + '/notify_setpoints_stop', NotifySetpointsStop, self.handle_notify_setpoints_stop)
        rospy.Service(prefix + '/update_params', UpdateParams, self.handle_update_params)

        rospy.Subscriber(prefix + "/cmd_full_state", FullState, self.handle_cmd_full_state)

    def handle_set_group_mask(self, req):
        self.setGroupMask(req.groupMask)
        return SetGroupMaskResponse()

    def handle_takeoff(self, req):
        self.takeoff(req.height, req.duration.to_sec(), req.groupMask)
        return TakeoffResponse()

    def handle_land(self, req):
        self.land(req.height, req.duration.to_sec(), req.groupMask)
        return LandResponse()

    def handle_go_to(self, req):
        print("ERROR NOT IMPLEMENTED!")

    def handle_upload_trajectory(self, req):
        print("ERROR NOT IMPLEMENTED!")

    def handle_start_trajectory(self, req):
        print("ERROR NOT IMPLEMENTED!")

    def handle_notify_setpoints_stop(self, req):
        print("ERROR NOT IMPLEMENTED!")

    def handle_update_params(self, req):
        print("ERROR NOT IMPLEMENTED!")

    def handle_cmd_full_state(self, msg):
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        vel = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        acc = [msg.acc.x, msg.acc.y, msg.acc.z]
        omega = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
        # TODO: extract yaw from quat?
        self.cmdFullState(pos, vel, acc, 0, omega)

class CrazyflieServerROS:
    def __init__(self, timehelper, crazyflies_yaml="../launch/crazyflies.yaml"):
        """Initialize the server.

        Args:
            crazyflies_yaml (str): If ends in ".yaml", interpret as a path and load
                from file. Otherwise, interpret as YAML string and parse
                directly from string.
        """
        if crazyflies_yaml.endswith(".yaml"):
            with open(crazyflies_yaml, 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
        else:
            cfg = yaml.safe_load(crazyflies_yaml)

        self.crazyflies = []
        self.crazyfliesById = dict()
        for crazyflie in cfg["crazyflies"]:
            id = int(crazyflie["id"])
            initialPosition = crazyflie["initialPosition"]
            cf = CrazyflieROS(id, initialPosition, timeHelper)
            self.crazyflies.append(cf)
            self.crazyfliesById[id] = cf


if __name__ == "__main__":

    rospy.init_node("CrazyflieROSSim", anonymous=False)

    timeHelper = TimeHelperROS("null", 0.1, False, 0)
    srv = CrazyflieServerROS(timeHelper, rospy.get_param("crazyflies_yaml"))
    timeHelper.crazyflies = srv.crazyflies

    while not rospy.is_shutdown():
        timeHelper.sleep(1)