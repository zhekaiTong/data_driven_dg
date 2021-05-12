# data_driven_dg

## Overview
This package provides an end-to-end data driven approach for picking thin objects from a dense clutter. It is developed based on our pevious model-based manipulation technique named 'dig-grasping', where the object is picked and singulated out of the cluster by a direct physical interaction with gripper. In this package, we train the robot to learn how to plan for optimized dig-grasping parameters to pick objects in different shapes. 

## Dataset
### 1. For pre-training the network
This dataset includes a) RGB heightmap images showing the real scene of objects, b) 2-D arrays for ground truth labels . The groud truth label is of the same size as the RGB heightmap, and obtained following our our model-based dig-grasping method. Each point in the label is associated with a class that represents for a background or successful/failed dig-grasp. Here shows an example:
<p align = "center">
<img src="files/Github_Go_stone_pick_place.gif" width="360" height="202"> 
<img src="files/Github_capsule_pick_place.gif" width="360" height="202"> 
</p>
