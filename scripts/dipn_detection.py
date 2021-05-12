import cv2
import numpy as np
import matplotlib.pyplot as plt
#import pyrealsense2 as rs
import os
import sys
import scipy
import random
import math
#import skimage.io
import datetime
import open3d as o3d

import time
import actionlib

env=os.path.expanduser(os.path.expandvars('~/dipn')) # "source" directory with python script
sys.path.insert(0, env)
import torch
import dig_maskRCNN

class Detection():

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = dig_maskRCNN.load_model(self.device,"maskrcnn.pth")
        self.cam_intrinsics = [321.8862609863281, 238.18316650390625, 612.0938720703125, 611.785888671875]
        # All shape
        self.heightmap_resolution = 0.000375#0.00225 / 4#
        self.workspace_limits = np.asarray([[-0.3, -0.15], [0.675, 0.825], [0.01, 0.1]]) #np.asarray([[-0.325-0.025, -0.15+0.025], [0.65-0.025, 0.825+0.025], [0.1, 0.41]])#
        # Domino
        #self.heightmap_resolution = 0.0013/4#0.00225 / 4
        #self.workspace_limits = np.asarray([[-0.01, 0.12], [0.63, 0.76], [0.02, 0.08]]) #np.asarray([[-0.325-0.025, -0.15+0.025], [0.65-0.025, 0.825+0.025], [0.1, 0.41]])
        self.edge_detection = False

    def pixel_to_camera(self,pixel, intrin, depth):
        #depth = depth #/ 1000
        X = (pixel[0]-intrin[0]) * depth / intrin[2]
        Y = (pixel[1]-intrin[1]) * depth / intrin[3]
        return [X, Y]

    def find_center(self,mask):
        g = np.mgrid[0:(mask.shape[0]),0:(mask.shape[1])]
        multiple_ = np.stack([mask,mask],0)*g
        total_sum = np.sum(multiple_,axis = (1,2))
        total_number = np.sum(mask)
        average = total_sum/total_number
        return average.astype(int)

    def find_depth(self,mask,depth_im, depth_value_percentage_taken = 25.0):

        masked_depth_im = depth_im * mask
        depth_list = masked_depth_im[masked_depth_im > 0]
        #print("dipn test0", depth_list)
        #print("dipn test 0.1", depth_list.shape)
        lower_bound = np.percentile(depth_list, 50 - depth_value_percentage_taken/2)
        upper_bound = np.percentile(depth_list, 50 + depth_value_percentage_taken/2)

        cut_depth_list1 = depth_list[depth_list >= lower_bound]
        cut_depth_list2 = cut_depth_list1[cut_depth_list1 <= upper_bound]

        depth_cut_average = np.average(cut_depth_list2)
        depth_max = np.amin(cut_depth_list2)/1000.0
        depth = depth_cut_average/1000.0
        return depth, depth_max

    def find_mask(self,im,depth_im,masks,center_point,distance_axis):
        masks = np.stack(masks,axis = -1)
        mask_store = 5
        center_y, center_x = center_point
        mask_size = np.sum(masks,axis = (0,1))
        mask_index = np.argsort(np.array(mask_size))[-min(mask_store + 2,masks.shape[2]):-2]
        min_distance2 = 10000000000
        most_center_point = []
        most_center_mask = []
        most_center_depth = []
        largest_area = 0

        for m in mask_index:
            if True:
                mask = masks[:,:,m]
                mask = mask.astype(np.uint8)
                [contours,hierarchy] = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cnt = contours[0]
                M = cv2.moments(cnt)
                [center_point_y,center_point_x] = self.find_center(mask)

                distance2 = (center_point_x - center_x) **2 *distance_axis[1] + (
                    center_point_y -center_y) **2 *distance_axis[0]

                if distance2 < min_distance2:#True:#largest_area < mask_size[m] :#and mask_size[m] < 2000000: #
                    largest_area = mask_size[m]
                    depth2, depth_max = self.find_depth(mask,depth_im, depth_value_percentage_taken = 90.0)
                    depth = depth_im[center_point_y,center_point_x]/1000.0
                    print('depth',depth)
                    #plt.imshow(mask)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))
                    mask_dilated = cv2.dilate(mask, kernel)
                    #plt.imshow(mask_dilated)

                    depth_env, depth_max_env = self.find_depth(mask_dilated,depth_im, depth_value_percentage_taken = 25.0)

                    if True:#depth_max_env > depth - 0.007 :# - 0.002 :#and  depth_max - depth  < 0.01:
                        min_distance2 = distance2
                        most_center_point = [center_point_y,center_point_x]
                        most_center_mask = mask
                        most_center_depth = [depth, depth_max]
        print('most_center_point',most_center_point)
        print('most_center_mask')
        #plt.imshow(most_center_mask)
        #plt.show()

        img_copy = im.copy()

        cv2.circle(img_copy, (center_point[1], center_point[0]), 8, (0, 0, 255), 8)
        cv2.circle(img_copy, (most_center_point[1], most_center_point[0]), 8, (255, 0, 255), 8)

        [contours,hierarchy] = cv2.findContours(most_center_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        for j in range(len(contours)):
            if(len(contours[j]) > len(cnt)):
                cnt = contours[j]
        hull = cv2.convexHull(cnt,returnPoints = True)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #img_copy = im.copy()
        cv2.drawContours(img_copy,[box],0,(255,255,255),2)
        plt.figure(figsize=(20,10))
        plt.imshow(img_copy[::-1,::-1,[2,1,0]])#[:,:,::-1])
        plt.show()

        return most_center_mask,most_center_point,most_center_depth,box

    def find_pose(self,depth_im, mask, center,depth,mask_box):

        #plt.imshow((depth_im*mask) != 0)
        #plt.show()
        depth_intrin = self.cam_intrinsics
        position = self.pixel_to_camera(center[::-1], depth_intrin, depth[0] )
        lower_point = []

        # Counterclockwise is positive direction
        if(np.linalg.norm(mask_box[0]-mask_box[1]) > np.linalg.norm(mask_box[1]-mask_box[2])):
            rotation = math.atan2((mask_box[2]-mask_box[1])[1], (mask_box[2]-mask_box[1])[0])
            print("rotation 111")
        else:
            rotation = math.atan2((mask_box[1]-mask_box[0])[1], (mask_box[1]-mask_box[0])[0])
            print("rotation 222")

        '''
        if rotation > math.pi:   ### reduce 180> x >360 to -180<x<0
            rotation = rotation - 2 * math.pi
            print("rotation 212")
        '''
        print('rotation,',math.degrees(rotation))

        pose={
                'x':position[0],
                'y':position[1],
                'z':depth[0] ,
                'yaw':rotation,
                'pitch':0,
                'normal':0
        }

        return pose

    def compute_normal(self,depth_array0 ,mask, heightmap_resolution, is_heightmap):
        kernel = np.ones((5,5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations = 3)
        #plt.imshow(mask_eroded)
        #plt.show()
        #print('dipn test 1.2', depth_array0)
        depth_array = depth_array0#*mask_eroded
        depth_copy = depth_array#depth_image.copy()
        #print("dipn test1", depth_copy)
        #print("dipn test 1.1", depth_copy.shape)

        mask_size = []
        points = []
        point_show = []
        points_ero = []
        point_show_ero = []
        mask_store = 5
        window = 10
        depth_intrin = self.cam_intrinsics

        point_all0 = []
        if is_heightmap == True:
            for i in range(depth_copy.shape[0]):
                for j in range(depth_copy.shape[1]):
                    if mask_eroded[i, j]>0 :#and depth_array[i, j]/1000 < 0.2:   True:#
                        #xy = self.pixel_to_camera([j, i], depth_intrin, depth_array[i, j])
                        #print('xy', xy)
                        point_x = j * self.heightmap_resolution + self.workspace_limits[0][0]
                        point_y = i * self.heightmap_resolution + self.workspace_limits[1][0]
                        #point_all0.append([point_x, point_y, (depth_array[i, j]+self.workspace_limits[2][0])*1000])
                        point_all0.append([j, i, depth_array[i, j]/self.heightmap_resolution])
                        #print('point',xy)
        else:
            for i in range(depth_copy.shape[0]):
                for j in range(depth_copy.shape[1]):
                    if mask_eroded[i, j]>0 :#and depth_array[i, j]/1000 < 0.2:   True:#
                        xy = self.pixel_to_camera([j, i], depth_intrin, depth_array[i, j])
                        #print('xy', xy)
                        point_all0.append([xy[0], xy[1], depth_array[i, j]])
                        #print('point',xy)
        print("dipn test 2", np.array(point_all0).shape)
        depth_average = np.mean(np.array(point_all0)[:,2])
        point_all = []

        '''
        for i in range(len(point_all0)):
            if point_all0[i][2] > depth_average:
                point_all.append(point_all0[i])
        #print("dipn test 2", point_all)
        pptest = np.array(point_all0)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pptest[:,0], pptest[:,1], pptest[:,2])
        plt.show()
        '''


        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(point_all0)
        #depth_array0[mask_eroded==0] = 0
        #pcd_all = o3d.geometry.create_point_cloud_from_depth_image(depth_array0*1000,depth_intrin)
        #o3d.io.write_point_cloud("test_ros_all.ply", pcd_all)

        #pcd_load = o3d.io.read_point_cloud("test_ros_all.ply")
        downpcd = o3d.geometry.voxel_down_sample(pcd_all, voxel_size=1)

        color = np.array([1,0,0])
        color = np.tile(color, (67, 1)) # 67 is number of points
        downpcd.colors = o3d.utility.Vector3dVector(color)
        #o3d.visualization.draw_geometries([downpcd])
        print("Recompute the normal of the downsampled point cloud")
        o3d.geometry.estimate_normals(downpcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=10, max_nn=30))
        print("Normal as a numpy array")

        for i in range(np.asarray(downpcd.normals).shape[0]):
            if downpcd.normals[i][2] > 0:
                downpcd.normals[i][0] = -downpcd.normals[i][0]
                downpcd.normals[i][1] = -downpcd.normals[i][1]
                downpcd.normals[i][2] = -downpcd.normals[i][2]
            downpcd.normals[i] = 3 * downpcd.normals[i]
        normals = np.asarray(downpcd.normals)
        #normals = normals / np.linalg.norm(normals)
        surf_normal = np.sum(normals, axis=0) / normals.shape[0]
        #o3d.visualization.draw_geometries([downpcd])

        return surf_normal#, depth

    def pose_detection(self, image, depth_array, heightmap_resolution, is_heightmap):
        masks,full_mask = dig_maskRCNN.find_mask(image,self.edge_detection,self.device, self.model)
        center = [int(image.shape[0]/2), int(image.shape[1]/2)]
        img_copy = image.copy()
        cv2.circle(img_copy, (int(center[1]), int(center[0])), 8, (255, 0, 255), 8)
        mask_max, most_center_point, depth,box = self.find_mask(image,depth_array,masks,center,[1,1])
        pose2 = self.find_pose(depth_array , mask_max, most_center_point, depth, box )
        normal = self.compute_normal(depth_array ,mask_max, heightmap_resolution, is_heightmap)

        return [pose2['x'], pose2['y'], pose2['z']], pose2['yaw'], normal
