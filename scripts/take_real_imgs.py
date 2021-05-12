import time
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from math import pi, sin, cos
#from robot import Robot
from robot_all import Robot
import heightmap
import util
from dipn_detection import Detection

import math3d as m3d


def main():
    num_obj = 8
    htmap_h = 100
    htmap_w = 100
    random_seed = 30
    data_order = 0
    is_random = False

    # Set random seed
    np.random.seed(random_seed)
    robot = Robot("192.168.1.102", is_testing=0)
    detection = Detection()
    workspace_limits = robot.workspace_limits
    heightmap_resolution = robot.heightmap_resolution
    data_order = 0


    while(True):
        is_random = input("is reshuffle?:")
        color_img, depth_img = robot.getCameraData()
        color_img_copy = color_img.copy()
        depth_img_copy = depth_img.copy()
        color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.baseTcam, workspace_limits, heightmap_resolution)


        # Save Data
        if not os.path.exists('./data_annotation/data_order.txt'):
            data_order += 1
            cv2.imwrite('./data_annotation/color_img/'+str(data_order)+'_color_img.png', color_img)
            cv2.imwrite('./data_annotation/color_heightmap/'+str(data_order)+'_color_heightmap.png', color_heightmap)
            cv2.imwrite('./data_annotation/depth_img/'+str(data_order)+'_depth_img.png', depth_img)
            cv2.imwrite('./data_annotation/depth_heightmap/'+str(data_order)+'_depth_heightmap.png', depth_heightmap)
            np.save('./data_annotation/color_img/'+str(data_order)+'_color_img.npy', color_img.astype(np.uint8))
            np.save('./data_annotation/color_heightmap/'+str(data_order)+'_color_heightmap.npy', color_heightmap.astype(np.uint8))
            np.save('./data_annotation/depth_img/'+str(data_order)+'_depth_img.npy', depth_img.astype(np.uint8))
            np.save('./data_annotation/depth_heightmap/'+str(data_order)+'_depth_heightmap.npy', depth_heightmap.astype(np.uint8))
            np.savetxt("./data_annotation/data_order.txt", [data_order], delimiter=' ')
        else:
            if data_order != 0:
                data_order += 1
                print("test 0", data_order)
            else:
                data_order_history = np.loadtxt('./data_annotation/data_order.txt')
                if data_order_history.shape == ():
                    data_order = int(data_order_history) + 1
                    print("test 1", data_order)
                else:
                    data_order = int(data_order_history[-1]) + 1
                    print("test 1.1", data_order)
            cv2.imwrite('./data_annotation/color_img/'+str(data_order)+'_color_img.png', color_img)
            cv2.imwrite('./data_annotation/color_heightmap/'+str(data_order)+'_color_heightmap.png', color_heightmap)
            cv2.imwrite('./data_annotation/depth_img/'+str(data_order)+'_depth_img.png', depth_img)
            cv2.imwrite('./data_annotation/depth_heightmap/'+str(data_order)+'_depth_heightmap.png', depth_heightmap)
            np.save('./data_annotation/color_img/'+str(data_order)+'_color_img.npy', color_img.astype(np.uint8))
            np.save('./data_annotation/color_heightmap/'+str(data_order)+'_color_heightmap.npy', color_heightmap.astype(np.uint8))
            np.save('./data_annotation/depth_img/'+str(data_order)+'_depth_img.npy', depth_img.astype(np.uint8))
            np.save('./data_annotation/depth_heightmap/'+str(data_order)+'_depth_heightmap.npy', depth_heightmap.astype(np.uint8))
            #np.save('./data_collection/'+str(data_order)+'_depth_img.npy', depth_img)
            with open("./data_annotation/data_order.txt", 'ab') as f:
                np.savetxt(f,[data_order])

if __name__ == '__main__':
    main()
