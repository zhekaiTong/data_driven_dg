import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math3d as m3d
import cv2
from scipy import ndimage

import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pcpt_res, dig_res

def get_pointcloud(color_img, depth_img, camera_intrinsics):

    depth_img = depth_img # for sim
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)


    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)
    #print("test0", surface_pts)
    #print("test0.0", np.where(surface_pts!=0))

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))
    #print("test0.0.1", surface_pts)
    #print("test0.0.2", surface_pts.shape)
    #print("test0", surface_pts[:,2])

    # Sort surface points by z value
    #sort_z_ind = np.argsort(surface_pts[:,2])
    sort_z_ind = np.argsort(np.array(surface_pts)[:,2])
    #print("test0.1", sort_z_ind.shape)
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    #print("test0.1.1", color_pts.shape)
    #print("test0.2", surface_pts)

    # Filter out surface points outside heightmap boundaries
    #print("test1", surface_pts.shape)
    #print("test1.1", workspace_limits[0][0])
    #print("test1.2", np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]))
    #print("test1.3", np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]))
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    #print("test1.4", heightmap_valid_ind.shape)
    #if is_sim:
    #    heightmap_valid_ind = heightmap_valid_ind.reshape(heightmap_valid_ind.shape[0],1) # for sim
    heightmap_valid_ind = np.array(heightmap_valid_ind)[:,0]
    #print("test2", heightmap_valid_ind)
    #print("test2.0", np.where(heightmap_valid_ind==True))
    #print("test2.0.1", np.array(heightmap_valid_ind)[:,0])
    #print("test2.1", surface_pts)
    #print("test2.1.1", surface_pts.shape)
    surface_pts = surface_pts[heightmap_valid_ind,:]
    color_pts = color_pts[heightmap_valid_ind]
    #print("test2.2", color_pts)

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1],1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1],1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1],1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    #print("test3", color_heightmap_r.shape)
    #print("test3.0", heightmap_pix_x)
    #print("test3.0.1", heightmap_pix_y)
    #print("test3.1", color_pts)
    #print("test3.1", color_pts.shape)
    #print("test3.1.1", color_heightmap_r[heightmap_pix_y,heightmap_pix_x].shape)
    #print("test3.2", depth_heightmap[heightmap_pix_y,heightmap_pix_x].shape)
    #print("test3.2.1", surface_pts[:,2])
    #if is_sim:
    #    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    #    heightmap_pix_x = heightmap_pix_x.reshape(heightmap_pix_x.shape[0],1) # for sim
    #    heightmap_pix_y = heightmap_pix_y.reshape(heightmap_pix_y.shape[0],1) # for sim
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x,0] = color_pts[:,[0]]
    #print("test3.2", color_heightmap_r.shape)
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x,0] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x,0] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    #print("test3.3", color_heightmap.shape)
    #plt.imshow(depth_heightmap[::-1])
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap = depth_heightmap + 0.01
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    #print("test4", color_heightmap.shape)
    #print("test4.1", depth_heightmap.shape)
    '''
    fig = plt.figure(3)
    fig.add_subplot(1,2,1)
    plt.imshow(cv2.cvtColor(color_heightmap,cv2.COLOR_BGR2RGB))
    fig.add_subplot(1,2,2)
    plt.imshow(depth_heightmap)
    plt.colorbar(label='Distance to Ground')

    plt.show()
    '''
    return color_heightmap, depth_heightmap

if __name__ == '__main__':
    color_img = cv2.imread("/home/zhekai/tensorflow_proj/Mask_RCNN/samples/stones/JPEGImages/0.jpeg")
    #color_img = scipy.misc.imread("/home/zhekai/tensorflow_proj/Mask_RCNN/samples/stones/JPEGImages/0.jpeg")
    #color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    depth_img = np.load("/home/zhekai/tensorflow_proj/Mask_RCNN/samples/stones/depth/0.npy")
    print(color_img.shape)

    cam_intrinsics = np.asarray([[612.0938720703125, 0, 321.8862609863281], [0, 611.785888671875, 238.18316650390625], [0, 0, 1]])

    eeTcam = m3d.Transform()
    eeTcam.pos = (0.076173, -0.0934057, 0.0074811)
    eeTcam_e = np.array([-0.4836677963432222, 1.5227704838700455, -0.4651199335909967])
    eeTcam_e = np.array([0, 0, 0])
    eeTcam.orient.rotate_xb(eeTcam_e[0])
    eeTcam.orient.rotate_yb(eeTcam_e[1])
    eeTcam.orient.rotate_zb(eeTcam_e[2])
    baseTee = m3d.Transform()
    baseTee.pos = (-0.23202, 0.62931, 0.40371)
    baseTee.orient = np.array([[-0.01316639,  0.99982403,  0.01336236],
           [ 0.99944137,  0.01356954, -0.03054217],
           [-0.03071812,  0.01295276, -0.99944416]])
    print(eeTcam)

    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray([[-0.39, -0.26], [0.63, 0.77], [-0.02, 0.3]])
    heightmap_resolution = 0.0005
    baseTcam = np.matmul(baseTee.get_matrix(), eeTcam.get_matrix())
    print("baseTcam", baseTcam)

    color_heightmap, depth_heightmap = get_heightmap(color_img, depth_img, cam_intrinsics, baseTcam, workspace_limits, heightmap_resolution, is_sim=True)
    '''

    # FROM trainer.py
    # Apply 2x scale to input heightmaps
    color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
    depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
    assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

    # Add extra padding (to handle rotations inside network)
    diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
    diag_length = np.ceil(diag_length/32)*32
    padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
    color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
    color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
    color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
    color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
    depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

    # Pre-process color image (scale and normalize)
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    #input_color_image = color_heightmap_2x.astype(float)/255
    input_color_image = color_heightmap.astype(float)/255
    for c in range(3):
        input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

    # Pre-process depth image (normalize)
    image_mean = [0.01, 0.01, 0.01]
    image_std = [0.03, 0.03, 0.03]
    #depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
    #input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
    depth_heightmap.shape = (depth_heightmap.shape[0], depth_heightmap.shape[1], 1)
    input_depth_image = depth_heightmap
    for c in range(1):
        input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

    # Construct minibatch of size 1 (b,c,h,w)
    input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
    input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
    input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
    input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)



    # FROM models.py
    num_rotations = 16
    f = plt.figure(4)
    for rotate_idx in range(num_rotations):
        rotate_theta = np.radians(rotate_idx*(360/num_rotations))

        # Compute sample grid for rotation BEFORE neural network
        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
        #print("test0",affine_mat_before)
        affine_mat_before.shape = (2,3,1)
        #print("test0.1",affine_mat_before)
        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
        #print("test0.1.1",affine_mat_before)

        flow_grid_before = F.affine_grid(affine_mat_before.cuda(), input_color_data.size())
        #print("test0.2",flow_grid_before.shape)

        # Rotate images clockwise
        rotate_color = F.grid_sample(input_color_data.cuda(), flow_grid_before, mode='nearest')
        rotate_depth = F.grid_sample(input_depth_data.cuda(), flow_grid_before, mode='nearest')

        #print(input_color_data.shape)

        f.add_subplot(4, 4, rotate_idx+1)
        plt.imshow(cv2.cvtColor(rotate_color.cpu().squeeze(0).permute(1, 2, 0).numpy(),cv2.COLOR_BGR2RGB))
    plt.show()
    '''
