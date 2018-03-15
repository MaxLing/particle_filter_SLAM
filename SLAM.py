import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import p4_util as util
from SLAM_functions import *

def main():
    ''' modify this part accordingly '''
    data_idx = '0'
    joint_dir = 'train/data/train_joint'+data_idx
    lidar_dir = 'train/data/train_lidar'+data_idx

    # load and process data
    joint_data, lidar_data, lidar_angles = data_preprocess(joint_dir, lidar_dir)

    # init
    Map, Particles = init_SLAM()

    for t in range(100):
        # Mapping
        lidar_scan = lidar_data[t]['scan']
        good_range = np.logical_and(lidar_scan>0.1, lidar_scan<30) # lidar spec
        lidar_xyz = polar2cart(lidar_scan[good_range], lidar_angles[good_range]) # in lidar cart coordinate 3*n

        print(0)
        # Localization



def data_preprocess(joint_dir, lidar_dir):
    joint_data = ld.get_joint(joint_dir)
    lidar_data = ld.get_lidar(lidar_dir)

    #util.replay_lidar(lidar_data)

    # get lidar angles
    num_beams = lidar_data[0]['scan'].shape[1]
    lidar_angles = np.linspace(start=-135*np.pi/180, stop=135*np.pi/180, num=num_beams).reshape(1,-1)

    return joint_data, lidar_data, lidar_angles

def init_SLAM():
    Map = {}
    Map['res'] = 20 # cells / m
    Map['size'] = 50 # m
    Map['map'] = np.zeros((Map['res']*Map['size'], Map['res']*Map['size'])) # log odds
    # TODO: set a bound for log odds

    Particles = {}
    Particles['nums'] = 10
    Particles['weights'] = np.ones(Particles['nums']) / Particles['nums']

    return Map, Particles



if __name__ == '__main__':
    main()