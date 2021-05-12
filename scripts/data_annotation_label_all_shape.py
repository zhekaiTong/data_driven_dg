import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from math import sin, cos, pi
import math
from PIL import Image
#from scipy import ndimage
#from robot import Robot
import heightmap_shape_prob
from dipn_detection import Detection

env=os.path.expanduser(os.path.expandvars('~/dipn')) # "source" directory with python script
sys.path.insert(0, env)
import dig_maskRCNN
detection = Detection()
crop_img_num = 9
crop_size = [200,200]
num_rotations = 8
total_img_num = 2155#1254
test = 0

def crop_image(image, depth_array, is_heightmap=True):
    if is_heightmap:
        crop_size = [200,200]
        crop = []
        crop_depth = []
        for row in range(3):
            for col in range(3):
                crop_up_lim = row*int(crop_size[0]/2)
                crop_bot_lim = (row+1)*(int(crop_size[0]/2))+100
                crop_left_lim = col*int(crop_size[1]/2)
                crop_right_lim = (col+1)*int(crop_size[1]/2)+100
                crop.append([image[crop_up_lim:crop_bot_lim,crop_left_lim:crop_right_lim], crop_left_lim, crop_right_lim, crop_up_lim, crop_bot_lim])
                crop_depth.append(depth_array[crop_up_lim:crop_bot_lim,crop_left_lim:crop_right_lim])
    else:
        crop_size = [200,200]
        crop = []
        crop_depth = []
        for row in range(3):
            for col in range(3):
                crop_up_lim = row*crop_size[0]
                crop_bot_lim = (row+1)*crop_size[0]
                crop_left_lim = 118+col*crop_size[1]
                crop_right_lim = 118+(col+1)*crop_size[1]
                crop.append(image[crop_up_lim:crop_bot_lim,crop_left_lim:crop_right_lim])
                crop_depth.append(depth_array[crop_up_lim:crop_bot_lim,crop_left_lim:crop_right_lim])

    return crop, crop_depth

def main(image, depth_array, img_record_num, heightmap_resolution, is_heightmap):
    global total_img_num
    masks,full_mask = dig_maskRCNN.find_mask(image,detection.edge_detection,detection.device, detection.model)
    crop, crop_depth = crop_image(image, depth_array, is_heightmap)
    print(len(masks))

    mask_accept_thed = 40*40 #80*40
    mask_accept = []
    d = 10#30

    for crop_ind in range(crop_img_num): #crop_img_num
        crop_copy = crop[crop_ind][0].copy()
        #label = np.ones((crop_copy.shape[0],crop_copy.shape[1]))*255
        label_rot = np.ones((crop_copy.shape[0],crop_copy.shape[1],num_rotations))*255
        mask_accept = []
        for mask_ind in range(len(masks)):
            if np.sum(masks[mask_ind]) >= mask_accept_thed*255:
                mask_in_crop = True
                #print("mask ind shape", masks[mask_ind].shape)
                # Now finding Contours         ###################
                #[contours,hierarchy] = cv2.findContours(masks[mask_ind], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                #cnt = contours[0]
                #for j in range(len(contours)):
                #    if(len(contours[j]) > len(cnt)):
                #        cnt = contours[j]
                # Judge whether the mask is in the croped image
                #hull = cv2.convexHull(cnt,returnPoints = True)
                #rect = cv2.minAreaRect(hull)
                #box = cv2.boxPoints(rect)
                #box = np.int0(box)
                #for corner in range(len(box)):
                #    if box[corner][0]<crop[crop_ind][1] or box[corner][0]>crop[crop_ind][2] or box[corner][1]<crop[crop_ind][3] or box[corner][1]>crop[crop_ind][4]:
                #        mask_in_crop = False
                mask_yx_orig = np.where(masks[mask_ind]==255)
                mask_yx_orig = np.array(mask_yx_orig) # [x
                                                      #  y]
                valid_mask_ind = np.logical_and(np.logical_and(np.logical_and(mask_yx_orig[0,:]>crop[crop_ind][3],mask_yx_orig[0,:]<crop[crop_ind][4]),mask_yx_orig[1,:]>crop[crop_ind][1]),mask_yx_orig[1,:]<crop[crop_ind][2])
                if np.min(valid_mask_ind.astype(np.uint8)) == 0:
                    mask_in_crop = False
                #mask_yx_orig = mask_yx_orig[:,valid_mask_ind]
                if mask_in_crop == True:
                    #mask_yx_orig = np.where(masks[mask_ind]==255)
                    #mask_yx_orig = np.array(mask_yx_orig) # [x
                                                          #  y]
                    mask_yx_crop = mask_yx_orig - np.ones((mask_yx_orig.shape[0],mask_yx_orig.shape[1]))*np.array([[crop[crop_ind][3]],[crop[crop_ind][1]]])
                    mask_yx_crop = mask_yx_crop.astype(np.uint8)
                    mask_crop = np.zeros((crop_size[0], crop_size[1])).astype(np.uint8)
                    mask_crop[tuple(mask_yx_crop)] = 255
                    if test == 1:
                        curr_mask = mask_crop#.copy()
                        overlay = np.zeros((curr_mask.shape[0],curr_mask.shape[1],3))
                        overlay[:,:,1] = curr_mask.astype(np.uint8)
                        #overlay = np.stack([np.zeros(curr_mask.shape),curr_mask*255/np.max(curr_mask),np.zeros(curr_mask.shape)],-1)
                        img4 = (0.7*crop_copy[:,:,[2,1,0]].copy() + 0.3*overlay).astype(np.uint8)
                        img4 = cv2.cvtColor(img4[:,:,[2,1,0]], cv2.COLOR_RGB2BGR)

                    # Recognize object shapes
                    [contours,hierarchy] = cv2.findContours(mask_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    for j in range(len(contours)):
                        if(len(contours[j]) > len(cnt)):
                            cnt = contours[j]
                    approx = cv2.approxPolyDP(cnt, 0.07 * cv2.arcLength(cnt, True), True) # approx [[[]] [[]] [[]]]hull = cv2.convexHull(cnt,returnPoints = True)
                    #print("approx", approx)
                    if len(approx) == 3:
                        obj_shape = 'triangle'
                    else:
                        hull = cv2.convexHull(cnt,returnPoints = True)
                        rect = cv2.minAreaRect(hull)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        if np.linalg.norm(box[0]-box[1])>1.5*np.linalg.norm(box[1]-box[2]) or 1.5*np.linalg.norm(box[0]-box[1])<np.linalg.norm(box[1]-box[2]):
                            obj_shape = 'domino'
                        else:
                            obj_shape = 'go_stone'
                    #print("obj_shape", obj_shape)
                    if test == 1:
                        plt.imshow(img4)
                        plt.text(25,25,str(crop_ind),fontsize=30, color='r')
                        plt.text(25, 50, obj_shape, fontsize=30, color='r')
                        plt.show()

                    if obj_shape == 'triangle':
                        apex_direction = []
                        approx = np.array(approx).squeeze()
                        #cv2.drawContours(crop_copy, [cnt], 0, (255,255,255), 3)
                        [c_y,c_x] = detection.find_center(mask_crop)
                        center = np.zeros(approx.shape) + np.array([[c_x,c_y]])
                        apex_direction_vec = approx - center
                        normal = detection.compute_normal(crop_depth[crop_ind],mask_crop, heightmap_resolution, True)
                        yaw = math.atan2(normal[0],normal[1])
                        yaw_text = yaw
                        yaw_difference = 100000
                        for apex_ind in range(apex_direction_vec.shape[0]): # in heightmap frame, right x, down y
                            apex_direction.append(math.atan2(apex_direction_vec[apex_ind][1],apex_direction_vec[apex_ind][0]))
                            if abs(yaw-apex_direction[apex_ind]) < yaw_difference:
                                yaw_difference = abs(yaw-apex_direction[apex_ind])
                                apex_select = apex_ind
                        yaw = apex_direction[apex_select] - pi/2 # in tcp frame in heightmap
                        if apex_direction[apex_select]<=-pi/2 and apex_direction[apex_select]>-pi:
                            yaw = apex_direction[apex_select] + 3*pi/2 # in tcp frame in heightmap

                        yaw_ = -yaw # in robot tcp frame
                        print("rotate check yaw", math.degrees(yaw_))
                        if yaw_ <= 0:
                            yaw_ = yaw_ + 2*pi
                        if yaw_ >=0 and yaw_ < math.radians(45/2):
                            rot_ind = 0
                        else:
                            rot_ind = -int((yaw_-math.radians(45/2))/math.radians(45))+(num_rotations-1)
                        print("rot ind", rot_ind)

                        d = 15
                        good_pt_x = int(c_x-(d+3)*sin(yaw))
                        good_pt_y = int(c_y+(d+3)*cos(yaw))
                        bad_pt_x = int(c_x+d*sin(yaw))
                        bad_pt_y = int(c_y-d*cos(yaw))
                        #cv2.circle(img4, (good_pt_x, good_pt_y), 2, (0, 255, 0), -1)
                        #cv2.circle(img4, (bad_pt_x, bad_pt_y), 2, (255, 0, 0), -1)
                        # Mesh grid for bad points
                        bad_area_h = 10
                        bad_area_w = 10
                        good_area_h = 5
                        good_area_w = 5
                        bad_x, bad_y = np.meshgrid(np.arange(bad_area_h), np.arange(bad_area_w))
                        bad_x = bad_x.flatten()
                        bad_x = bad_x[:,np.newaxis]
                        bad_y = bad_y.flatten()
                        bad_y = bad_y[:,np.newaxis]
                        bad_pts = np.concatenate((bad_x,bad_y),axis=1)
                        rot_matrix = np.array([[cos(yaw+pi), -sin(yaw+pi)],
                                                [sin(yaw+pi), cos(yaw+pi)]])
                        bad_pts = bad_pts @ rot_matrix
                        shift = np.ones((bad_pts.shape[0],bad_pts.shape[1]))
                        shift[:,0] = shift[:,0]*(bad_pt_y+7*cos(yaw))
                        shift[:,1] = shift[:,1]*(bad_pt_x-7*sin(yaw))
                        bad_pts = bad_pts + shift
                        for bad_pts_ind in range(bad_pts.shape[0]):
                            if int(bad_pts[bad_pts_ind][0]>0) and int(bad_pts[bad_pts_ind][0]<crop_size[0]) and int(bad_pts[bad_pts_ind][1]>0) and int(bad_pts[bad_pts_ind][1]<crop_size[1]):
                                if int(bad_pts[bad_pts_ind][0]%2) == 0:
                                    label_rot[int(bad_pts[bad_pts_ind][0]),int(bad_pts[bad_pts_ind][1]),rot_ind] = 0
                                    cv2.circle(crop_copy, (int(bad_pts[bad_pts_ind][1]), int(bad_pts[bad_pts_ind][0])), 2, (0, 0, 255), -1)

                        good_x, good_y = np.meshgrid(np.arange(good_area_h), np.arange(good_area_w))
                        good_x = good_x.flatten()
                        good_x = good_x[:,np.newaxis]
                        good_y = good_y.flatten()
                        good_y = good_y[:,np.newaxis]
                        good_pts = np.concatenate((good_x,good_y),axis=1)
                        rot_matrix = np.array([[cos(yaw+pi), -sin(yaw+pi)],
                                                [sin(yaw+pi), cos(yaw+pi)]])
                        good_pts = good_pts @ rot_matrix
                        shift = np.ones((good_pts.shape[0],good_pts.shape[1]))
                        shift[:,0] = shift[:,0]*(good_pt_y+4*cos(yaw))
                        shift[:,1] = shift[:,1]*(good_pt_x-4*sin(yaw))
                        good_pts = good_pts + shift
                        print(good_pts.shape)
                        #print(good_pts[:,1].flatten())
                        #label[good_pt_y, good_pt_x] = 128
                        for good_pts_ind in range(good_pts.shape[0]):
                            if int(good_pts[good_pts_ind][0]>0) and int(good_pts[good_pts_ind][0]<crop_size[0]) and int(good_pts[good_pts_ind][1]>0) and int(good_pts[good_pts_ind][1]<crop_size[1]):
                                label_rot[int(good_pts[good_pts_ind][0]),int(good_pts[good_pts_ind][1]),rot_ind] = 128
                                cv2.circle(crop_copy, (int(good_pts[good_pts_ind][1]), int(good_pts[good_pts_ind][0])), 2, (0, 255, 0), -1)

                    elif obj_shape == 'go_stone':
                        d = 10#30
                        #cv2.drawContours(crop_copy, [cnt], 0, (255,255,255), 3)
                        #label = np.ones((crop_copy.shape[0],crop_copy.shape[1]))*255
                        [c_y,c_x] = detection.find_center(mask_crop)

                        normal = detection.compute_normal(crop_depth[crop_ind] ,mask_crop, heightmap_resolution, is_heightmap)
                        print("normal", normal)
                        yaw = math.atan2(normal[0],normal[1])+pi # already in tcp frame in real
                        if yaw > pi:
                            yaw = yaw - 2*pi

                        # Save rotated heightmap
                        yaw_ = yaw # in robot tcp frame
                        print("rotate check yaw", math.degrees(yaw_))
                        if yaw_ <= 0:
                            yaw_ = yaw_ + 2*pi
                        if yaw_ >=0 and yaw_ < math.radians(45/2):
                            rot_ind = 0
                        else:
                            rot_ind = -int((yaw_-math.radians(45/2))/math.radians(45))+(num_rotations-1)
                            #rot_ind = int((yaw-45/2)/45)
                            #label_rot[:,:,rot_ind] = ndimage.rotate(label, rot_ind*45, reshape=False)
                        #else:
                            #rot_ind = -int((yaw+45/2)/45)+num_rotations
                        print("rot ind", rot_ind)

                        yaw = -yaw #math.atan2(normal[0],normal[1])#math.atan2(normal[1],normal[0])+pi # for drawing on the label image, in image axis
                        good_pt_x = int(c_x-d*sin(yaw))
                        good_pt_y = int(c_y+d*cos(yaw))
                        bad_pt_x = int(c_x+(d+5)*sin(yaw))
                        bad_pt_y = int(c_y-(d+5)*cos(yaw))
                        #good_pt_x = int(c_x+d*cos(yaw))
                        #good_pt_y = int(c_y+d*sin(yaw))
                        #bad_pt_x = int(c_x-d*cos(yaw))
                        #bad_pt_y = int(c_y-d*sin(yaw))
                        cv2.circle(crop_copy, (good_pt_x, good_pt_y), 2, (0, 255, 0), -1)
                        cv2.circle(crop_copy, (bad_pt_x, bad_pt_y), 2, (0, 0, 255), -1)


                        # Mesh grid for bad points
                        bad_area_h = 10#35
                        bad_area_w = 10#25
                        good_area_h = 5
                        good_area_w = 5#15
                        bad_x, bad_y = np.meshgrid(np.arange(bad_area_h), np.arange(bad_area_w))
                        bad_x = bad_x.flatten()
                        bad_x = bad_x[:,np.newaxis]
                        bad_y = bad_y.flatten()
                        bad_y = bad_y[:,np.newaxis]
                        bad_pts = np.concatenate((bad_x,bad_y),axis=1)
                        rot_matrix = np.array([[cos(yaw+pi), -sin(yaw+pi)],
                                                [sin(yaw+pi), cos(yaw+pi)]])
                        bad_pts = bad_pts @ rot_matrix
                        shift = np.ones((bad_pts.shape[0],bad_pts.shape[1]))
                        shift[:,0] = shift[:,0]*(bad_pt_y+7*cos(yaw))
                        shift[:,1] = shift[:,1]*(bad_pt_x-7*sin(yaw))
                        bad_pts = bad_pts + shift
                        for bad_pts_ind in range(bad_pts.shape[0]):
                            if int(bad_pts[bad_pts_ind][0]>0) and int(bad_pts[bad_pts_ind][0]<crop_size[0]) and int(bad_pts[bad_pts_ind][1]>0) and int(bad_pts[bad_pts_ind][1]<crop_size[1]):
                                if int(bad_pts[bad_pts_ind][0]%2) == 0:
                                    #label[int(bad_pts[bad_pts_ind][0]),int(bad_pts[bad_pts_ind][1])] = 0
                                    label_rot[int(bad_pts[bad_pts_ind][0]),int(bad_pts[bad_pts_ind][1]),rot_ind] = 0
                                    cv2.circle(crop_copy, (int(bad_pts[bad_pts_ind][1]), int(bad_pts[bad_pts_ind][0])), 2, (0, 0, 255), -1)

                        good_x, good_y = np.meshgrid(np.arange(good_area_h), np.arange(good_area_w))
                        good_x = good_x.flatten()
                        good_x = good_x[:,np.newaxis]
                        good_y = good_y.flatten()
                        good_y = good_y[:,np.newaxis]
                        good_pts = np.concatenate((good_x,good_y),axis=1)
                        rot_matrix = np.array([[cos(yaw+pi), -sin(yaw+pi)],
                                                [sin(yaw+pi), cos(yaw+pi)]])
                        good_pts = good_pts @ rot_matrix
                        shift = np.ones((good_pts.shape[0],good_pts.shape[1]))
                        shift[:,0] = shift[:,0]*(good_pt_y+7*cos(yaw))
                        shift[:,1] = shift[:,1]*(good_pt_x-7*sin(yaw))
                        good_pts = good_pts + shift
                        print(good_pts.shape)
                        #print(good_pts[:,1].flatten())
                        #label[good_pt_y, good_pt_x] = 128
                        for good_pts_ind in range(good_pts.shape[0]):
                            if int(good_pts[good_pts_ind][0]>0) and int(good_pts[good_pts_ind][0]<crop_size[0]) and int(good_pts[good_pts_ind][1]>0) and int(good_pts[good_pts_ind][1]<crop_size[1]):
                                #label[int(good_pts[good_pts_ind][0]),int(good_pts[good_pts_ind][1])] = 128
                                label_rot[int(good_pts[good_pts_ind][0]),int(good_pts[good_pts_ind][1]),rot_ind] = 128
                                cv2.circle(crop_copy, (int(good_pts[good_pts_ind][1]), int(good_pts[good_pts_ind][0])), 2, (0, 255, 0), -1)

                        #label_rot[good_pts[:,0].astype(int),good_pts[:,1].astype(int),rot_ind] = 128#ndimage.rotate(label, rot_ind*45, reshape=False)
                        #label_rot[bad_pts[:,0].astype(int),bad_pts[:,1].astype(int),rot_ind] = 0
                    elif obj_shape == 'domino':
                        d = 35#25#30
                        [c_y,c_x] = detection.find_center(mask_crop)
                        [contours,hierarchy] = cv2.findContours(mask_crop,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                        cnt = contours[0]
                        for j in range(len(contours)):
                            if(len(contours[j]) > len(cnt)):
                                cnt = contours[j]
                        hull = cv2.convexHull(cnt,returnPoints = True)
                        rect = cv2.minAreaRect(hull)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        #cv2.drawContours(crop_copy,[box],0,(255,255,255),2)
                        if(np.linalg.norm(box[0]-box[1]) > np.linalg.norm(box[1]-box[2])):
                            yaw = math.atan2((box[2]-box[1])[1], (box[2]-box[1])[0])
                            #print("rotation 111", math.degrees(yaw))
                        else:
                            yaw = math.atan2((box[1]-box[0])[1], (box[1]-box[0])[0])
                            #print("rotation 222", math.degrees(yaw))
                        #yaw = -yaw # for comparing with normal. in robot tcp frame (in world)
                        normal = detection.compute_normal(crop_depth[crop_ind] ,mask_crop, heightmap_resolution, is_heightmap)
                        print("normal", normal)
                        yaw_difference = abs(math.atan2(normal[0],normal[1]) - yaw)
                        if  yaw_difference > pi/2.0 and yaw_difference < 3.0 * pi/2.0:
                            yaw = yaw + pi
                        # Manually adjust yaw by 180d
                        #yaw = yaw + pi
                        if yaw > pi: # keep yaw in (-pi, pi)
                            yaw = yaw - 2*pi
                        # Save rotated heightmap
                        yaw_ = -yaw # in robot tcp frame
                        print("rotate check yaw", math.degrees(yaw_))
                        if yaw_ <= 0:
                            yaw_ = yaw_ + 2*pi
                        if yaw_ >=0 and yaw_ < math.radians(45/2):
                            rot_ind = 0
                        else:
                            rot_ind = -int((yaw_-math.radians(45/2))/math.radians(45))+(num_rotations-1)
                        print("rot ind", rot_ind)
                        #yaw = -yaw # for drawing on the label image, in image axis
                        good_pt_x = int(c_x-(d+5)*sin(yaw))
                        good_pt_y = int(c_y+(d+5)*cos(yaw))
                        bad_pt_x = int(c_x+d*sin(yaw))
                        bad_pt_y = int(c_y-d*cos(yaw))
                        # Mesh grid for bad points
                        bad_area_h = 40#35
                        bad_area_w = 30#25
                        good_area_h = 10#5
                        good_area_w = 20#15
                        bad_x, bad_y = np.meshgrid(np.arange(bad_area_h), np.arange(bad_area_w))
                        bad_x = bad_x.flatten()
                        bad_x = bad_x[:,np.newaxis]
                        bad_y = bad_y.flatten()
                        bad_y = bad_y[:,np.newaxis]
                        bad_pts = np.concatenate((bad_x,bad_y),axis=1)
                        rot_matrix = np.array([[cos(yaw+pi), -sin(yaw+pi)],
                                                [sin(yaw+pi), cos(yaw+pi)]])
                        bad_pts = bad_pts @ rot_matrix
                        shift = np.ones((bad_pts.shape[0],bad_pts.shape[1]))
                        shift[:,0] = shift[:,0]*(c_y+11*cos(yaw-pi/2))
                        shift[:,1] = shift[:,1]*(c_x-11*sin(yaw-pi/2))
                        bad_pts = bad_pts + shift
                        for bad_pts_ind in range(bad_pts.shape[0]):
                            if int(bad_pts[bad_pts_ind][0]>0) and int(bad_pts[bad_pts_ind][0]<crop_size[0]) and int(bad_pts[bad_pts_ind][1]>0) and int(bad_pts[bad_pts_ind][1]<crop_size[1]):
                                if int(bad_pts[bad_pts_ind][0]%2) == 0:
                                    label_rot[int(bad_pts[bad_pts_ind][0]),int(bad_pts[bad_pts_ind][1]),rot_ind] = 0
                                    cv2.circle(crop_copy, (int(bad_pts[bad_pts_ind][1]), int(bad_pts[bad_pts_ind][0])), 2, (0, 0, 255), -1)

                        good_x, good_y = np.meshgrid(np.arange(good_area_h), np.arange(good_area_w))
                        good_x = good_x.flatten()
                        good_x = good_x[:,np.newaxis]
                        good_y = good_y.flatten()
                        good_y = good_y[:,np.newaxis]
                        good_pts = np.concatenate((good_x,good_y),axis=1)
                        rot_matrix = np.array([[cos(yaw+pi), -sin(yaw+pi)],
                                                [sin(yaw+pi), cos(yaw+pi)]])
                        good_pts = good_pts @ rot_matrix
                        shift = np.ones((good_pts.shape[0],good_pts.shape[1]))
                        shift[:,0] = shift[:,0]*(good_pt_y+7*cos(yaw-pi/2))
                        shift[:,1] = shift[:,1]*(good_pt_x-7*sin(yaw-pi/2))
                        good_pts = good_pts + shift
                        print(good_pts.shape)
                        #print(good_pts[:,1].flatten())
                        #label[good_pt_y, good_pt_x] = 128
                        for good_pts_ind in range(good_pts.shape[0]):
                            if int(good_pts[good_pts_ind][0]>0) and int(good_pts[good_pts_ind][0]<crop_size[0]) and int(good_pts[good_pts_ind][1]>0) and int(good_pts[good_pts_ind][1]<crop_size[1]):
                                label_rot[int(good_pts[good_pts_ind][0]),int(good_pts[good_pts_ind][1]),rot_ind] = 128
                                cv2.circle(crop_copy, (int(good_pts[good_pts_ind][1]), int(good_pts[good_pts_ind][0])), 2, (0, 255, 0), -1)

                    mask_accept.append([mask_crop, yaw, [c_x,c_y], [good_pt_x, good_pt_y], [bad_pt_x, bad_pt_y]])

        if mask_accept != []:
##            np.save('./data_annotation_heightmap/raw_image/'+str(total_img_num)+'.npy', crop[crop_ind])
##            cv2.imwrite('./data_annotation_heightmap/raw_image/'+str(total_img_num)+'.png', crop[crop_ind])
#            cv2.imwrite('./data_annotation_heightmap/test/'+str(img_record_num)+'_'+str(crop_ind)+'_crop.jpg',crop_copy)
            for i in range(num_rotations):
                if np.min(label_rot[:,:,i])<255:
                    crop_rot_temp = Image.fromarray(crop[crop_ind][0][:,:,[2,1,0]])
                    crop_rot_temp = crop_rot_temp.rotate(angle=i*45, fillcolor=(255,255,255))
                    crop_rot_temp.save('./data_annotation_heightmap_0512/raw_image/'+str(total_img_num)+'_'+str(i)+'.png','PNG')
                    #cv2.imwrite('./data_annotation_heightmap/raw_image/'+str(total_img_num)+'_'+str(i)+'.png', crop[crop_ind])

                    print(str(crop_ind)+"_rotated", i*45)
                    label_rot_temp = Image.fromarray(label_rot[:,:,i])
                    label_rot[:,:,i] = np.array(label_rot_temp.rotate(angle=i*45, fillcolor=255))
                    #label_rot[:,:,i] = ndimage.rotate(label_rot[:,:,i], i*45, reshape=False)
##                np.save('./data_annotation_heightmap/label/'+str(total_img_num)+'_'+str(i)+'.npy', label_rot[:,:,i].reshape((crop_copy.shape[0],crop_copy.shape[1])))
                    cv2.imwrite('./data_annotation_heightmap_0512/label/'+str(total_img_num)+'_'+str(i)+'.png', label_rot[:,:,i].reshape((crop_copy.shape[0],crop_copy.shape[1])))
##                np.save('./data_annotation_heightmap/label/'+str(total_img_num)+'_'+str(i)+'.npy',label_rot[:,:,i].reshape((crop_copy.shape[0],crop_copy.shape[1])))
            print("total_img_num", total_img_num)
            total_img_num += 1


if __name__ == '__main__':


    #robot = Robot("192.168.1.102", is_testing=0)
    #workspace_limits = robot.workspace_limits
    #heightmap_resolution = robot.heightmap_resolution / 4
    heightmap_resolution = 0.000375#0.00225 / 4
    for img_num in range(250,300): # 191 200
        #img_num = 171
        #color_heightmap = np.load('/home/zhekai/RL_Projects/Reinforcement-Learning/dg_learning_real/data_annotation/'+str(img_num+1)+'_color_heightmap.npy')
        color_heightmap = cv2.imread('/home/zhekai/RL_Projects/Reinforcement-Learning/dg_learning_real/data_annotation/color_heightmap/'+str(img_num+1)+'_color_heightmap.png')
        depth_heightmap = np.load('/home/zhekai/RL_Projects/Reinforcement-Learning/dg_learning_real/data_annotation/depth_heightmap/'+str(img_num+1)+'_depth_heightmap.npy')


        is_heightmap = True
        main(color_heightmap, depth_heightmap, img_num+1, heightmap_resolution, is_heightmap)
        print("nth heightmap", img_num)
