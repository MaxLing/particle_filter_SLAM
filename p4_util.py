"""""""""""""""""""""""""""""""""""""""""""""
Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania
Date: Feb 2017

DESCRIPTION:
	The replay_* functions help you to visualize and understand the lidar, depth, and rgb data. 
	The "get_joint_index" function returns joint ID number.

"""""""""""""""""""""""""""""""""""""""""""""

# need to import djpeg
import numpy as np
import matplotlib.pyplot as plt
            
def replay_lidar(lidar_data):
	# lidar_data type: array where each array is a dictionary with a form of 't','pose','res','rpy','scan'
	theta = np.arange(0,270.25,0.25)*np.pi/float(180)

	for i in range(200,len(lidar_data),10):
		for (k,v) in enumerate(lidar_data[i]['scan'][0]):
			if v > 30:
				lidar_data[i]['scan'][0][k] = 0.0

		# Jinwook's plot
		ax = plt.subplot(111, projection='polar')
		ax.plot(theta, lidar_data[i]['scan'][0])
		ax.set_rmax(10)
		ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
		ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
		ax.grid(True)
		ax.set_title("Lidar scan data", va='bottom')

		plt.draw()
		plt.pause(0.001)



def replay_depth(depth_data):
	DEPTH_MAX = 4500
	DEPTH_MIN = 400	
	for k in range(len(depth_data)):
		D = depth_data[k]['depth']
		D = np.flip(D,1)
		for r in range(len(D)):
			for (c,v) in enumerate(D[r]):
				if (v<=DEPTH_MIN) or (v>=DEPTH_MAX):
					D[r][c] = 0.0

		plt.imshow(D)		
		plt.draw()
		plt.pause(0.001)

def replay_rgb(rgb_data):
	for k in range(len(rgb_data)):
		R = rgb_data[k]['image']
		R = np.flip(R,1)

		plt.imshow(R)		
		plt.draw()
		plt.pause(0.001)



def get_joint_index(joint):
    jointNames = ['Neck','Head','ShoulderL', 'ArmUpperL', 'LeftShoulderYaw','ArmLowerL','LeftWristYaw','LeftWristRoll','LeftWristYaw2','PelvYL','PelvL','LegUpperL','LegLowerL','AnkleL','FootL','PelvYR','PelvR','LegUpperR','LegLowerR','AnkleR','FootR','ShoulderR', 'ArmUpperR', 'RightShoulderYaw','ArmLowerR','RightWristYaw','RightWristRoll','RightWristYaw2','TorsoPitch','TorsoYaw','l_wrist_grip1','l_wrist_grip2','l_wrist_grip3','r_wrist_grip1','r_wrist_grip2','r_wrist_grip3','ChestLidarPan']
    joint_idx = 1
    for (i,jnames) in enumerate(joint):
        if jnames in jointNames:
            joint_idx = i
            break
    return joint_idx





