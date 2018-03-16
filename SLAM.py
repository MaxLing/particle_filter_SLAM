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
    Trajectory = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(color='gray', linestyle='--', linewidth='0.5')
    occ, = ax.plot([], [], 'k')
    free, = ax.plot([], [], 'w')
    pose, = ax.plot([], [], 'g*', markersize=10)
    # traj, = ax.plot([], [], 'b')

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
        not_floor = world_hit[2,:]>0.2
        world_hit = world_hit[:,not_floor]

        # update map according to hit
        update_map(world_hit[:2], Pose[:2], Map)


        # Localization




        # update plot
        occ_, free_ = map2plot(Map)
        occ.set_data(occ_[0], occ_[1])
        free.set_data(free_[0], free_[1])
        pose.set_data(Pose[0], Pose[1])

        # resize map
        plt.xlim(np.min(occ_[0]), np.max(occ_[0]))
        plt.ylim(np.min(occ_[1]), np.max(occ_[1]))

        fig.canvas.draw()
        plt.pause(0.001)

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
    belief = 0.9 # prob of lidar hit if the grid is occupied
    Map['occ_d'] = np.log(belief/(1-belief))
    Map['free_d'] = np.log((1-belief)/belief)*.5
    # TODO: set a bound for log odds

    Particles = {}
    Particles['nums'] = 10
    Particles['weights'] = np.ones(Particles['nums']) / Particles['nums']
    Particles['poses'] = np.zeros((3, Particles['nums']))
    Particles['best_idx'] = 0

    return Map, Particles



if __name__ == '__main__':
    main()