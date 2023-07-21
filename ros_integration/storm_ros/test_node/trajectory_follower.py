#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time
import argparse
from typing import List, Tuple, Any

import rospy


class TrajectoryFollowerNode:
    def __init__(self):
        """
        Initializes the subscribers, loads the data from file, and loads the model.
        """
        rospy.init_node("trajectory_follower_node")


        self.joint_state_publisher = rospy.Publisher(
            "/storm/joint_states", JointState, queue_size=1
        )

        self.ref_trajectory_subscriber = rospy.Subscriber(
            "/mpinets/plan",
            JointTrajectory,
            self.trajectory_update_callback,
            queue_size=1,
        )
        rospy.loginfo("the follower node is set up")
    
    def trajectory_update_callback(self, msg: JointTrajectory):
        """
        Receives and update the new planned jointTrajectory from mpinets for reference
        """

        reference_trajectory_list = []
        for i in range(len(msg.points)):
            reference_trajectory_list.append(msg.points[i].positions)
        reference_trajectory_npary = np.array(reference_trajectory_list)

        # we should replace the current ref_trajectory with the new one
        reference_trajectory = reference_trajectory_npary
        print("reference_trajectory is ",reference_trajectory)
        rospy.loginfo("i was called")

if __name__ == "__main__":
    TrajectoryFollowerNode()
    rospy.spin()
