import os
import rospy
import numpy as np
from math import pi, cos, sin, tan, radians, degrees
import math3d as m3d
import random
import heightmap

from arc_rotate import *
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

import socket
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import math3d as m3d
import logging

import matplotlib.pyplot as plt
import cv2
import time

import serial
import signal
import time

#serialPort = "/dev/ttyACM0"
#baudRate = 57600
#ser = serial.Serial(serialPort, baudRate, timeout=0.5)

def myHandler(signum, frame):
    pass

class Robot():

    def __init__(self, tcp_host_ip, is_testing):

        self.camera_width = 640
        self.camera_height = 480
        self.maxVelocity = 1.
        self.maxForce = 200.
        self.workspace_limits = np.asarray([[-0.3, -0.15], [0.675, 0.825], [0.01, 0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.heightmap_resolution = 0.000375#0.00225 / 4

        logging.basicConfig(level=logging.WARN)
        self.rob = urx.Robot(tcp_host_ip) #"192.168.1.102"

        self.resetFT300Sensor(tcp_host_ip)
        #action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        #self.robotiq_client = actionlib.SimpleActionClient(action_name, CommandRobotiqGripperAction)
        #self.robotiq_client.wait_for_server()

        self.resetRobot(1)
        self.setCamera()

    def resetRobot(self, aperture):
        self.go_to_safe()
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)
        self.gp_control(195) #184
        time.sleep(1)
        self.go_to_home()
        #self.gp_control(aperture)
        #self.rob.set_tcp((-0.001, 0.0, 0.31292, 0, 0, 0))
        self.rob.set_tcp((0, 0.0, 0, 0, 0, 0))
        time.sleep(.5)
        #print(self.rob.get_pose())
        self.baseTee = self.rob.get_pose()
        self.baseTee.orient =  np.array([[0,  1, 0], [ 1,  0,  0], [ 0, 0, -1]])
        self.rob.set_tcp((0.005, 0.001, 0.33815, 0, 0, 0))
        time.sleep(.5)
#        self.fg_length_control('60')
        print("reset")


    def resetFT300Sensor(self, tcp_host_ip):
        HOST = tcp_host_ip
        PORT = 63351
        self.serialFT300Sensor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serialFT300Sensor.connect((HOST, PORT))

    def getFT300SensorData(self):
        while True:
            data = str(self.serialFT300Sensor.recv(1024),"utf-8").replace("(","").replace(")","").split(",")
            try:
                data = [float(x) for x in data]
                if len(data) == 6:
                    break
            except:
                pass
        return data

    def setCamera(self):
        self.cam_intrinsics = np.asarray([[612.0938720703125, 0, 321.8862609863281], [0, 611.785888671875, 238.18316650390625], [0, 0, 1]])
        #eeTcam = m3d.Transform()
        #eeTcam_pos = (0.076173+0.05, 0.0074811, 0.0934057+0.03)
        self.eeTcam = np.array([[0, -1, 0, 0.142],
                           [1, 0, 0, -0.003], #0.0074811
                           [0, 0, 1, 0.0934057+0.03],
                           [0, 0, 0, 1]])

        #baseTee = m3d.Transform()
        #baseTee.pos = -0.24706, 0.59010, 0.59051#(-0.23191, 0.62426, 0.64795)
        #baseTee.orient = np.array([[-0.01101442,  0.99981683, -0.01565198],
        #   [ 0.99987911,  0.01118421,  0.01080192],
        #   [ 0.010975  , -0.01553111, -0.99981915]])
        #baseTee = self.rob.get_pose()
        self.baseTcam = np.matmul(self.baseTee.get_matrix(), self.eeTcam)
        '''
        self.baseTcam = np.array([[1, 0, 0, -0.25007318],
                           [0, -1, 0, 0.73211299], #0.0074811
                           [0, 0, -1, 0.46702988],
                           [0, 0, 0, 1]])
        '''

    def getCameraData(self):
        # Set the camera settings.
        #self.setCamera()
        # Setup:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, 30)
        #config.enable_record_to_file('frames/0.bag')
        profile = pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print("depth_scale", self.depth_scale)

        # Skip 5 first frames to give the Auto-Exposure time to adjust
        #for x in range(10):
        #    pipeline.wait_for_frames()
        # Store next frameset for later processing:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Cleanup:
        pipeline.stop()
        print("Frames Captured")

        color = np.asanyarray(color_frame.get_data())

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        # Show the two frames together:
        #images = np.hstack((color, colorized_depth))
        #plt.imshow(images)
        #plt.show()
        print("robot test1")
        return color, depth_image*self.depth_scale


    def gp_control(self, aperture, delay_time = 0.7):
        self.robotiqgrip.gripper_action(aperture)
        time.sleep(delay_time)

    def fg_length_control(self, d):
        signal.signal(signal.SIGALRM, myHandler)
        signal.setitimer(signal.ITIMER_REAL, 0.001)
        ser.write(str.encode(d))
        signal.setitimer(signal.ITIMER_REAL, 0)
        time.sleep(.5)

    def go_to_home(self):
        #home_position = [126.30,-83.73,-101.82,-83.43,89.96,-53.34]
        home_position = [124.16,-90.64,-118.18,-60.17,89.93,-55.66]
        Hong_joint0 = math.radians(home_position[0])
        Hong_joint1 = math.radians(home_position[1])
        Hong_joint2 = math.radians(home_position[2])
        Hong_joint3 = math.radians(home_position[3])
        Hong_joint4 = math.radians(home_position[4])
        Hong_joint5 = math.radians(home_position[5])

        self.rob.movej((Hong_joint0, Hong_joint1, Hong_joint2, Hong_joint3, Hong_joint4, Hong_joint5), 0.5, 1)

    def go_to_safe(self):
        #home_position = [126.30,-83.73,-101.82,-83.43,89.96,-53.34]
        home_position = [124.08,-86.00,-109.65,-73.34,89.97,-55.64]
        Hong_joint0 = math.radians(home_position[0])
        Hong_joint1 = math.radians(home_position[1])
        Hong_joint2 = math.radians(home_position[2])
        Hong_joint3 = math.radians(home_position[3])
        Hong_joint4 = math.radians(home_position[4])
        Hong_joint5 = math.radians(home_position[5])

        self.rob.movej((Hong_joint0, Hong_joint1, Hong_joint2, Hong_joint3, Hong_joint4, Hong_joint5), 0.5, 1)

    def Frame(self, pos, ori):
        mat = R.from_quat(ori).as_matrix()
        F = np.concatenate(
            [np.concatenate([mat, [[0, 0, 0]]], axis=0), np.reshape([*pos, 1.], [-1, 1])], axis=1
        )
        return F

    def exec_dig_grasp(self, pos, rot_z, rot_y):
        # Align
        pos[2] = pos[2]+0.005

        #self.rob.set_tcp((-0.001, 0.0, 0.31292, 0, 0, 0))
        #time.sleep(.5)
        #print(self.rob.get_pose())
        eefPose = self.rob.get_pose()
        eefPose = eefPose.get_pose_vector()
        self.rob.movel((0,0,0.05,0,0,0), acc=0.3, vel=0.8,relative=True)
        self.rob.movel((pos[0]-eefPose[0]+0.01,pos[1]-eefPose[1]+0.005,0,0,0,0), acc=0.3, vel=0.8,relative=True)
        #self.rob.movel((0,0,pos[2]-eefPose[2],rot_z,0,0), acc=0.05, vel=0.05,relative=True)
        move = m3d.Transform((0,0,-(pos[2]-eefPose[2])+0.05,0,0,0))
        self.rob.add_pose_tool(move, acc=0.2, vel=0.3, wait=True, command="movel", threshold=None)
        input("check")
        move = m3d.Transform((0,0,0,0,0,rot_z))
        self.rob.add_pose_tool(move, acc=0.2, vel=0.3, wait=True, command="movel", threshold=None)
        move = m3d.Transform((0,0,0,0,20*pi/180,0))
        self.rob.add_pose_tool(move, acc=0.2, vel=0.3, wait=True, command="movel", threshold=None)
        #self.rob.movel_tool((pos[0]-eefPose[0],pos[1]-eefPose[1],pos[2]-eefPose[2],0,0,0),acc=0.05, vel=0.05)


        # Dig into clutter
        dig_dist = 0.04#0.055
        dig_duration = 3.5#2.5
        safe_fz_threshold = 55
        init_fz = self.getFT300SensorData()[2]
        delta_z_last = 0
        delta_z_last_2 = 0
        delta_z_window = 0
        time0 = time.time()
        self.rob.translate_tool((0, 0, dig_dist), acc=0.03, vel=0.07, wait=False)
        while True:
            current_fz = self.getFT300SensorData()[2]
            delta_z = abs(current_fz - init_fz)
            delta_z_avg = (delta_z + delta_z_last + delta_z_last_2)/3
            if delta_z_window > 3:
                if delta_z_avg > safe_fz_threshold:
                    print("robot test 1")
                    self.rob.stopl()
                    time.sleep(.7)
                    self.gp_control(211)
                    time.sleep(.3)
                    grasp_success = 0
                    break
                elif time.time()-time0 > dig_duration:
                    time.sleep(.7)
                    self.gp_control(211)
                    time.sleep(.3)
                    break
            delta_z_last_2 = delta_z_last
            delta_z_last = delta_z
            delta_z_window += 1

        #self.gp_control(209) # close gripper

        #self.gp_control(0.008)
        #time.sleep(.5)
        #self.go_to_home()
        #self.gp_control(0.04)
        #grasp_success = input("Is grasp successful? 1/0: ")
        self.rob.movel((0,0,-pos[2]+eefPose[2]+0.15,0,0,0), acc=0.2, vel=0.3,relative=True)
        grasp_success = input("Is grasp successful? 1/0: ")
        self.go_to_home()
        self.gp_control(195)

        return int(grasp_success)


'''
if __name__ == '__main__':
    #wl = np.asarray([[-0.39, 0], [0.6, 0.77], [-0.02, 0.3]])
    #wl = np.asarray([[0.23, 0.47], [-0.7, -0.5], [-0.6, 0.0]])
    wl = np.asarray([[0.07, 0.35], [-0.8, -0.6], [-0.1, 0]])
    heightmap_resolution = 0.002
    robot = Robot(is_sim=1,num_obj=20, workspace_limits=wl,is_testing=0)
    color_img, depth_img = robot.getCameraData()
    print(np.array(robot.view_matrix).reshape(4,4))
    #cam_intrinsics = np.asarray([[robot.proj_matrix[0], 0, 0], [0, robot.proj_matrix[5], 0], [0, 0, 1]])
    #cam_pose = np.transpose(np.array(robot.view_matrix).reshape(4,4))
    #cam_intrinsics = np.asarray([[robot.proj_matrix[0]*(480/2), 0, 0], [0, robot.proj_matrix[5]*(480/2), 0], [0, 0, 1]])
    cam_intrinsics = np.asarray([[612.0938720703125, 0, 321.8862609863281], [0, 611.785888671875, 238.18316650390625], [0, 0, 1]])
    cam_pose = np.transpose(np.array(robot.view_matrix).reshape(4,4))
    #cam_pose[:3,3] = np.array([-0.23,0.7,0.45])
    color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, robot.workspace_limits, heightmap_resolution, robot.is_sim)
    print(color_heightmap.shape, depth_heightmap.shape)

    fig = plt.figure(3)
    fig.add_subplot(1,2,1)
    plt.imshow(cv2.cvtColor(color_heightmap,cv2.COLOR_BGR2RGB))
    fig.add_subplot(1,2,2)
    plt.imshow(depth_heightmap)
    plt.colorbar(label='Distance to Ground')

    plt.show()
'''
