import cv2
import numpy as np
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

    # init SLAM
    Map, Particles, Trajectory = init_SLAM()

    # init plot
    h, w = Map['map'].shape
    Plot = np.zeros((h,w,3),np.uint8)

    for lidar_idx in range(10):
        Pose = Particles['poses'][:, Particles['best_idx']] # TODO: now only the best particle is used

        # Mapping
        # extract lidar scan in good range and transform to lidar's cart coordinate
        lidar_scan = lidar_data[lidar_idx]['scan']
        good_range = np.logical_and(lidar_scan>0.1, lidar_scan<30) # lidar spec
        lidar_hit = polar2cart(lidar_scan[good_range], lidar_angles[good_range]) # 3*n

        # find closest joint data
        joint_idx = np.argmin(np.abs(joint_data['ts']-lidar_data[lidar_idx]['t']))
        joint_angles = joint_data['head_angles'][:,joint_idx]

        # transform hit from lidar to world coordinate, remove ground hitting
        world_hit = lidar2world(lidar_hit, joint_angles, Pose)
        not_floor = world_hit[2,:]>0.1
        world_hit = world_hit[:,not_floor]

        # update map according to hit
        update_map(world_hit[:2], Pose[:2], Map)

        # Localization



        # Plot
        plot_all(Map, Trajectory, Plot)

        print(0)



def data_preprocess(joint_dir, lidar_dir):
    joint_data = ld.get_joint(joint_dir)
    lidar_data = ld.get_lidar(lidar_dir)

    #util.replay_lidar(lidar_data)

    # get lidar angles
    num_beams = lidar_data[0]['scan'].shape[1]
    lidar_angles = np.linspace(start=-135*np.pi/180, stop=135*np.pi/180, num=num_beams).reshape(1,-1)

    return joint_data, lidar_data, lidar_angles



if __name__ == '__main__':
    main()