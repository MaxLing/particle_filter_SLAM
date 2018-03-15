"""""""""""""""""""""""""""""""""""""""""""""

Author: Heejin Chloe Jeong (heejinj@seas.upenn.edu)
Affiliation: University of Pennsylvania
Date: Feb 2017

DESCRIPTION
: In this file, you can load .mat file data in python dictionary format.
  The output of the "get_lidar" function is an array with dictionary elements. The length of the array is the length of data. 
  The output of the "get_joint" function is a dictionary with eight different data (read data description for details). Each dictionary is an array with the same length.
  The output of the "get_rgb" function is an array with dictionary elements. The length of the array is the length of data.
  The output of the "get_depth" function is an array with dictionary elements. The length of the array is the lenght of data.

"""""""""""""""""""""""""""""""""""""""""""""

import pickle
from scipy import io


def get_lidar(file_name):
	data = io.loadmat(file_name+".mat")
	lidar = []
	for m in data['lidar'][0]:
		x = {}
		x['t']= m[0][0][0]
		n = len(m[0][0])
		if (n != 5) and (n != 6):			
			raise ValueError("different length!")
		x['pose'] = m[0][0][n-4]
		x['res'] = m[0][0][n-3]
		x['rpy'] = m[0][0][n-2]
		x['scan'] = m[0][0][n-1]
		
		lidar.append(x)
	return lidar

def get_joint(file_name):
	key_names_joint = ['acc', 'ts', 'rpy', 'gyro', 'pos', 'ft_l', 'ft_r', 'head_angles']
	data = io.loadmat(file_name+".mat")
	joint = {kn: data[kn] for kn in key_names_joint}
	return joint


def get_rgb(file_name):
	key_names_rgb = ['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','image']
	# image size: 1080x1920x3 uint8
	data = io.loadmat(file_name+".mat")
	data = data['RGB'][0]
	rgb = []
	for m in data:
		tmp = {v:m[0][0][i] for (i,v) in enumerate(key_names_rgb)}
		rgb.append(tmp)
	return rgb

def get_depth(file_name):
	key_names_depth = ['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','depth']
	data = io.loadmat(file_name+".mat")
	data = data['DEPTH'][0]
	depth = []
	for m in data:
		tmp = {v:m[0][0][i] for (i,v) in enumerate(key_names_depth)}
		depth.append(tmp)
	return depth





