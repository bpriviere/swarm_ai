#!/usr/bin/env python

import rospy
from tf import TransformListener

import yaml
import numpy as np
from crazyflie_driver.srv import *
from pycrazyswarm.crazyflieSim import TimeHelper, Crazyflie

# Z = 1.0

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

    timeHelper = TimeHelper("mpl", 0.1, False, 0)
    srv = CrazyflieServerROS(timeHelper)
    timeHelper.crazyflies = srv.crazyflies

    timeHelper.sleep(10)