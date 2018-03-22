import numpy as np
import load_data as ld
from SLAM_functions import *

def main():
    ''' modify this part accordingly '''
    data_idx = 1
    joint_dir = 'data/train_joint'+str(data_idx)
    lidar_dir = 'data/train_lidar'+str(data_idx)

    # load and process data
    joint_data, lidar_data, lidar_angles = data_preprocess(joint_dir, lidar_dir)

    # init SLAM
    Map, Particles, Trajectory = init_SLAM()

    # init plot
    h, w = Map['map'].shape
    Plot = np.zeros((h,w,3),np.uint8)

    interval = 5 # interval is a trade-off of accuracy for speed
    for lidar_idx in range(0, len(lidar_data), interval):
        Pose = Particles['poses'][:, np.argmax(Particles['weights'])]
        Trajectory.append(np.copy(Pose))

        # Mapping
        # extract lidar scan in good range and transform to lidar's cart coordinate
        lidar_scan = lidar_data[lidar_idx]['scan']
        good_range = np.logical_and(lidar_scan>0.1, lidar_scan<30) # lidar spec
        lidar_hit = polar2cart(lidar_scan[good_range], lidar_angles[good_range]) # 3*n

        # find closest joint data
        joint_idx = np.argmin(np.abs(joint_data['ts']-lidar_data[lidar_idx]['t']))
        joint_angles = joint_data['head_angles'][:,joint_idx]

        # transform hit from lidar to world coordinate, this step also remove ground hitting
        world_hit = lidar2world(lidar_hit, joint_angles, lidar_data[lidar_idx]['rpy'][0,:], pose=Pose)

        # update map according to hit
        update_map(world_hit[:2], Pose[:2], Map)


        # Localization
        # no predict or update if no prev odometry
        if lidar_idx == 0:
            continue

        # predict particles pose by odom
        odom_predict(Particles, lidar_data[lidar_idx]['pose'][0,:2], lidar_data[lidar_idx]['rpy'][0,2],
                     lidar_data[lidar_idx-interval]['pose'][0,:2], lidar_data[lidar_idx-interval]['rpy'][0,2])

        # update particles by lidar
        particle_update(Particles, Map, lidar_hit, joint_angles, lidar_data[lidar_idx]['rpy'][0,:])


        # Plot
        # last frame
        if lidar_idx >= len(lidar_data)-interval:
            plot_all(Map, Trajectory, world_hit, Plot, idx = data_idx)
        # update normal frame in a frequency
        if lidar_idx%100==0:
            plot_all(Map, Trajectory, world_hit, Plot)


def data_preprocess(joint_dir, lidar_dir):
    joint_data = ld.get_joint(joint_dir)
    lidar_data = ld.get_lidar(lidar_dir)

    # get lidar angles
    num_beams = lidar_data[0]['scan'].shape[1]
    lidar_angles = np.linspace(start=-135*np.pi/180, stop=135*np.pi/180, num=num_beams).reshape(1,-1)

    # remove bias for odometry, init pose is (0,0,0)
    yaw_bias = lidar_data[0]['rpy'][0,2]
    pose_bias = lidar_data[0]['pose'][0,:2]
    for i in range(len(lidar_data)):
        lidar_data[i]['rpy'][0,2] -= yaw_bias
        lidar_data[i]['pose'][0,:2] -= pose_bias

    return joint_data, lidar_data, lidar_angles

def init_SLAM():
    Map = {}
    Map['res'] = 20 # cells / m
    Map['size'] = 50 # m
    Map['map'] = np.zeros((Map['res']*Map['size'], Map['res']*Map['size'])) # log odds
    belief = 0.7 # prob of lidar hit if the grid is occupied
    Map['occ_d'] = np.log(belief/(1-belief))
    Map['free_d'] = np.log((1-belief)/belief)*.5
    occ_thres = 0.9
    free_thres = 0.2
    Map['occ_thres'] = np.log(occ_thres / (1 - occ_thres))
    Map['free_thres'] = np.log(free_thres/(1-free_thres))
    Map['bound'] = 100 # allow log odds recovery

    Particles = {}
    Particles['nums'] = 100
    Particles['weights'] = np.ones(Particles['nums']) / Particles['nums']
    Particles['poses'] = np.zeros((3, Particles['nums']))
    Particles['noise_cov'] = [0.005, 0.005, 0.005] # [0.005, 0.005, 0.005]
    Particles['n_eff'] = .1*Particles['nums']

    Trajectory = []

    return Map, Particles, Trajectory

if __name__ == '__main__':
    main()