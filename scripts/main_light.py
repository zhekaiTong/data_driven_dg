import time
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sin, cos
import cv2
import glob
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.autograd import Variable
from torchsummary import summary
from PIL import Image

import light_res
from models_light import DigGraspNet

from robot_all import Robot
import heightmap

def crop_image(image, depth_array):
    crop_size = [200,200]
    crop = []
    crop_depth = []
    for row in range(3):
        for col in range(3):
            crop_up_lim = row*int(crop_size[0]/2)
            crop_bot_lim = (row+1)*(int(crop_size[0]/2))+100
            crop_left_lim = col*int(crop_size[1]/2)
            crop_right_lim = (col+1)*int(crop_size[1]/2)+100
            crop.append(image[crop_up_lim:crop_bot_lim,crop_left_lim:crop_right_lim])
            crop_depth.append(depth_array[crop_up_lim:crop_bot_lim,crop_left_lim:crop_right_lim])
    return crop, crop_depth


htmap_h = 200
htmap_w = 200
num_rotations = 8

robot = Robot("192.168.1.102", is_testing=0)
workspace_limits = robot.workspace_limits
heightmap_resolution = robot.heightmap_resolution

def get_prediction_vis(predictions, color_heightmap, best_ind_in_all):

    canvas = None
    num_rotations = predictions.shape[0]
    for canvas_row in range(int(num_rotations/4)):
        tmp_row_canvas = None
        for canvas_col in range(4):
            rotate_idx = canvas_row*4+canvas_col
            prediction_vis = predictions[rotate_idx,1,:,:].copy()
            prediction_rot_back = predictions[rotate_idx,1,:,:].copy()
            # Normalize prediction to (0-1)
            #prediction_vis = (prediction_vis-np.ones((prediction_vis.shape[0],prediction_vis.shape[1]))*np.min(prediction_vis))/(np.max(prediction_vis)-np.min(prediction_vis))

            best_ind = np.unravel_index(np.argmax(prediction_vis), prediction_vis.shape)
            '''
            prediction_rot_back[tuple(best_ind)] = 138
            prediction_rot_back = np.array(Image.fromarray(prediction_rot_back).rotate(angle=-rotate_idx*45, fillcolor=0))
            best_ind_after_rot = np.array(np.where(prediction_rot_back==138))
            print("test visual", best_ind_after_rot)
            if best_ind_after_rot.shape[1] > 1:
                best_ind_after_rot = best_ind_after_rot[:,0]
            '''
            #print("new best rot", np.array(best_ind_after_rot).shape)
            best_ind_after_rot = point_rotation2([best_ind[0], htmap_h-best_ind[1]],[htmap_w/2,htmap_h/2],math.radians(rotate_idx*45))
            best_ind_after_rot = [int(best_ind_after_rot[0]), int(htmap_h-best_ind_after_rot[1])]
            prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
            #if rotate_idx == best_pix_ind[0]:
            #cv2.circle(prediction_vis, (int(best_ind[1]), int(best_ind[0])), 7, (255,0,0), 2)
            background_image = Image.fromarray(color_heightmap)
            background_image = np.array(background_image.rotate(angle=rotate_idx*45, fillcolor=(255,255,255)))
            #background_image = np.array(Image.open('./data_annotation_heightmap/raw_image/1_'+str(rotate_idx)+'.png'))
            prediction_vis = (0.5*cv2.cvtColor(background_image[:,:,[2,1,0]], cv2.COLOR_RGB2BGR) + 0.5*prediction_vis[:,:,[2,1,0]]).astype(np.uint8)
            cv2.circle(prediction_vis, (int(best_ind[1]), int(best_ind[0])), 10, (255,0,0), 4)
            cv2.circle(prediction_vis, (int(best_ind_after_rot[1]), int(best_ind_after_rot[0])), 7, (255,255,255), 4)
            if rotate_idx == best_ind_in_all[0]:
                cv2.circle(prediction_vis, (int(best_ind_in_all[2]), int(best_ind_in_all[1])), 10, (0,0,0), 4)
#                cv2.circle(prediction_vis, (int(best_ind_all_after_rot[1]), int(best_ind_all_after_rot[0])), 10, (0,0,255), 4)

            if tmp_row_canvas is None:
                tmp_row_canvas = prediction_vis
            else:
                tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
        if canvas is None:
            canvas = tmp_row_canvas
        else:
            canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

    return canvas

def point_rotation2(point, rotation_pole, rot_angle):
    current_disp = list(np.array(point)-np.array(rotation_pole))
    rot_matrix = np.array([[cos(rot_angle), -sin(rot_angle)],
                           [sin(rot_angle), cos(rot_angle)]])
    current_disp = np.expand_dims(current_disp, axis=1)
    temp = np.dot(rot_matrix, current_disp)
    after = [list(rotation_pole[0]+temp[0])[0], list(rotation_pole[1]+temp[1])[0]]
    return after

def crop_to_orig(crop_ind, pixel):
    row = int(crop_ind/3)
    col = int(crop_ind%3)
    pix_x_in_orig = col*int(htmap_w/2) + pixel[0]
    pix_y_in_orig = row*int(htmap_h/2) + pixel[1]
    return [pix_x_in_orig, pix_y_in_orig]

while(True):
    input("next turn")
    # Initialize and get images
#    robot.fg_length_control('0')
    color_img, depth_img = robot.getCameraData()
    color_img_copy = color_img.copy()
    depth_img_copy = depth_img.copy()
    color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.baseTcam, workspace_limits, heightmap_resolution)

    #model = DigGraspNet().cuda()
    model = light_res.myResNet(light_res.BasicBlock,[1,1,1],light_res.BasicBlock,[1,1,1],200,200).cuda()
    #summary(model,(3,200,200))
    model.load_state_dict(torch.load('./weights/weights_0511_1/weights-000099.pth'))

    model.eval()

    test_image_ = color_heightmap#np.load('./data_annotation/color_heightmap/2_color_heightmap.npy')#np.array(Image.open('./data_annotation/color_heightmap/1_color_heightmap.jpg'))
    test_depth_ = depth_heightmap
    crop, crop_depth = crop_image(test_image_, test_depth_)

    for crop_ind in range(len(crop)):
        test_image = crop[crop_ind]

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        # Construct minibatch of size 1 (b,c,h,w)
        with torch.no_grad():
            for rotate_idx in range(num_rotations):
                test_image_rot = Image.fromarray(test_image[:,:,[2,1,0]])
                test_image_rot = np.array(test_image_rot.rotate(angle=rotate_idx*45, fillcolor=(255,255,255)))
                input_color_image = test_image_rot.astype(float)/255
                for c in range(3):
                    input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]
                input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
                input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1).cuda()
                output_predict = model(input_color_data)

                output_predict = output_predict.permute(0,2,3,1)
                soft_max_f = nn.Softmax(dim=3)
                output_predict = soft_max_f(output_predict)
                output_predict = output_predict.permute(0,3,1,2)

                if rotate_idx == 0:
                    contact_predict = output_predict.cpu().data.numpy()
                else:
                    contact_predict = np.concatenate((contact_predict,output_predict.cpu().data.numpy()),axis=0)


        contact_predict.shape = (1,contact_predict.shape[0],contact_predict.shape[1],contact_predict.shape[2],contact_predict.shape[3])
        if crop_ind == 0:
            all_crop_predict = contact_predict
        else:
            all_crop_predict = np.concatenate((all_crop_predict,contact_predict),axis = 0) # shape(9,8,3,400,400)
    print("all_crop_predict shape", all_crop_predict.shape)

    # Execute robot
    bin_radius = 130#35
    class_good = all_crop_predict[:,:,1,:,:] # shape (9,8,400,400)
    best_ind_in_all = np.unravel_index(np.argmax(class_good), class_good.shape) # [crop_ind, rotate_idx, pix_y, pix_x]
    print("class_good", class_good.shape)
    print("class_good index", best_ind_in_all)
    print("class_good max value", class_good[tuple(best_ind_in_all)])
    while(True):
        rotate_idx = best_ind_in_all[1]
        best_ind_all_after_rot = point_rotation2([best_ind_in_all[2], htmap_h-best_ind_in_all[3]],[htmap_w/2,htmap_h/2],math.radians(rotate_idx*45))
        best_ind_all_after_rot = [int(best_ind_all_after_rot[0]), int(htmap_h-best_ind_all_after_rot[1])]
        pix_x_in_orig, pix_y_in_orig = crop_to_orig(best_ind_in_all[0], [best_ind_all_after_rot[1],best_ind_all_after_rot[0]])
        #if False:
        #    print("false")
        if np.linalg.norm(np.array([pix_x_in_orig,pix_y_in_orig])-np.array([htmap_w,htmap_h])) > bin_radius \
        or (best_ind_all_after_rot[0]>htmap_h or best_ind_all_after_rot[0]<0) or (best_ind_all_after_rot[1]>htmap_w or best_ind_all_after_rot[1]<0) :
            class_good[tuple(best_ind_in_all)] = 0
            best_ind_in_all = np.array(np.unravel_index(np.argmax(class_good), class_good.shape))
        else:
            print("after bin radius")
            print("class_good index", best_ind_in_all)
            print("class_good max value", class_good[tuple(best_ind_in_all)])
            break

    select_crop_img = crop[best_ind_in_all[0]].copy()
    select_crop_img_rot = Image.fromarray(select_crop_img[:,:,[2,1,0]])
    select_crop_img_rot = select_crop_img_rot.rotate(angle=rotate_idx*45, fillcolor=(255,255,255))
    select_crop_img_rot = np.array(select_crop_img_rot)
    cv2.circle(select_crop_img_rot, (int(best_ind_in_all[3]), int(best_ind_in_all[2])), 7, (255,0,0), 4)
    cv2.circle(select_crop_img, (int(best_ind_all_after_rot[1]), int(best_ind_all_after_rot[0])), 7, (255,255,255), 4)
    print("after rot index", best_ind_all_after_rot)
    select_orig_img = test_image_.copy()

    print("pix_in_orig", pix_x_in_orig, pix_y_in_orig)
    cv2.circle(select_orig_img, (pix_x_in_orig, pix_y_in_orig), 7, (255,0,0), 4)
    '''
    canvas = get_prediction_vis(all_crop_predict[best_ind_in_all[0],:,:,:,:], crop[best_ind_in_all[0]], best_ind_in_all[1:])
    plt.imshow(canvas)
    plt.show()
    '''

    f = plt.figure(1)
    f.add_subplot(2,2,1)
    plt.imshow(select_crop_img_rot)
    plt.text(25, 25, str(rotate_idx), fontsize=40, color='r')
    f.add_subplot(2,2,2)
    plt.imshow(select_crop_img[:,:,[2,1,0]])
    f.add_subplot(2,2,3)
    plt.imshow(select_orig_img[:,:,[2,1,0]])
    plt.show()


    #best_pos_after_rot = point_rotation2([best_ind_in_all[1], best_ind_in_all[2]],[htmap_w,htmap_w],math.radians(-rotate_idx*45))
    best_pix_x = pix_x_in_orig
    best_pix_y = pix_y_in_orig
    if rotate_idx in range(5):
        yaw = -rotate_idx*math.radians(45)
    else:
        yaw = (rotate_idx-3)*math.radians(45)
    pos = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

    print("main robot param", math.degrees(yaw), pos)

    robot.exec_dig_grasp(pos, yaw, 0)







#if __name__ == '__main__':

#    main()
