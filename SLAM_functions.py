import numpy as np
import cv2
from scipy.special import logsumexp

def polar2cart(scan, angles):
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    z = np.zeros(len(scan))
    return np.vstack((x, y, z))

def lidar2world(lidar_hit, joint_angles, body_angles, pose=None, Particles=None):
    neck_angle = joint_angles[0] # yaw wrt body frame
    head_angle = joint_angles[1] # pitch wrt body frame
    roll_gb = body_angles[0]
    pitch_gb = body_angles[1]
    yaw_gb = body_angles[2] # using imu's yaw has better performance than pose's yaw

    # lidar wrt head
    z_hl = 0.15
    H_hl = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z_hl],[0,0,0,1]]) # no rotation

    # head wrt body
    z_bh = 0.33
    T_bh = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z_bh],[0,0,0,1]])
    Rz = np.array([[np.cos(neck_angle), -np.sin(neck_angle), 0, 0],
                    [np.sin(neck_angle), np.cos(neck_angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    Ry = np.array([[np.cos(head_angle), 0, np.sin(head_angle), 0],
                   [0, 1, 0, 0],
                   [-np.sin(head_angle), 0, np.cos(head_angle), 0],
                   [0, 0, 0, 1]])
    R_bh = np.dot(Rz,Ry)
    H_bh = np.dot(T_bh, R_bh)

    if Particles is None: # for mapping
        # body wrt world
        x_gb = pose[0]
        y_gb = pose[1]
        z_gb = 0.93
        T_gb = np.array([[1, 0, 0, x_gb], [0, 1, 0, y_gb], [0, 0, 1, z_gb], [0, 0, 0, 1]])
        # yaw_gb = pose[2]
        R_gb = np.array([[np.cos(yaw_gb), -np.sin(yaw_gb), 0, 0],
                        [np.sin(yaw_gb), np.cos(yaw_gb), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])\
            .dot(np.array([[np.cos(pitch_gb), 0, np.sin(pitch_gb), 0],
                        [0, 1, 0, 0],
                        [-np.sin(pitch_gb), 0, np.cos(pitch_gb), 0],
                        [0, 0, 0, 1]]))\
            .dot(np.array([[1, 0, 0, 0],
                        [0, np.cos(roll_gb), -np.sin(roll_gb), 0],
                        [0, np.sin(roll_gb), np.cos(roll_gb), 0],
                        [0, 0, 0, 1]]))
        H_gb = np.dot(T_gb, R_gb)

        # lidar wrt world
        H_gl = H_gb.dot(H_bh).dot(H_hl)
        lidar_hit = np.vstack((lidar_hit,np.ones((1,lidar_hit.shape[1])))) # 4*n
        world_hit = np.dot(H_gl, lidar_hit)

        # ground check, keep hits not on ground
        not_floor = world_hit[2]>0.1
        world_hit = world_hit[:,not_floor]

        return world_hit[:3,:]

    else: # for particles update
        nums = Particles['nums']
        poses = Particles['poses']
        particles_hit = []
        lidar_hit = np.vstack((lidar_hit, np.ones((1, lidar_hit.shape[1]))))

        for i in range(nums):
            # body wrt world
            T_gb = np.array([[1, 0, 0, poses[0,i]], [0, 1, 0, poses[1,i]], [0, 0, 1, 0.93], [0, 0, 0, 1]])
            # yaw_gb = poses[2,i]
            R_gb = np.array([[np.cos(yaw_gb), -np.sin(yaw_gb), 0, 0],
                            [np.sin(yaw_gb), np.cos(yaw_gb), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]) \
                .dot(np.array([[np.cos(pitch_gb), 0, np.sin(pitch_gb), 0],
                               [0, 1, 0, 0],
                               [-np.sin(pitch_gb), 0, np.cos(pitch_gb), 0],
                               [0, 0, 0, 1]])) \
                .dot(np.array([[1, 0, 0, 0],
                               [0, np.cos(roll_gb), -np.sin(roll_gb), 0],
                               [0, np.sin(roll_gb), np.cos(roll_gb), 0],
                               [0, 0, 0, 1]]))
            H_gb = np.dot(T_gb, R_gb)

            # lidar wrt world
            H_gl = H_gb.dot(H_bh).dot(H_hl)
            world_hit = np.dot(H_gl, lidar_hit)[:3,:]

            # ground check, keep hits not on ground
            not_floor = world_hit[2] > 0.1
            particles_hit.append(world_hit[:, not_floor])

        return np.transpose(np.asarray(particles_hit), (1,2,0))


def world2map(xy, Map):
    # transform origin from center to upper left, meter to pixel
    pixels = np.zeros(xy.shape, dtype=int)
    pixels[0] = ((xy[0] + Map['size']/2)*Map['res']).astype(np.int)
    pixels[1] = ((-xy[1] + Map['size']/2)*Map['res']).astype(np.int) # y direction changes

    # check boundary and keep pixels within
    center = Map['size']*Map['res']/2
    in_bound = np.logical_and(np.abs(pixels[0]-center) < center, np.abs(pixels[1]-center) < center)
    pixels = pixels[:,in_bound]

    return pixels

def update_map(hit, pose, Map):
    # transform hit to occ grid and check boundary
    occ = world2map(hit, Map)

    # update log odds for occupied grid, Note: pixels access should be (column, row)
    Map['map'][occ[1], occ[0]] += Map['occ_d']-Map['free_d'] # will add back later

    # update log odds for free grid, using contours to mask region between pose and hit
    mask = np.zeros(Map['map'].shape)
    contour = np.hstack((world2map(pose, Map).reshape(-1,1), occ))
    cv2.drawContours(image=mask, contours = [contour.T], contourIdx = -1, color = Map['free_d'], thickness=-1)
    Map['map'] += mask

    # keep log odds within boundary, to allow recovery
    Map['map'][Map['map']>Map['bound']] = Map['bound']
    Map['map'][Map['map']<-Map['bound']] = -Map['bound']

def odom_predict(Particles, curr_xy, curr_theta, prev_xy, prev_theta):
    # relative movement in local frame (odom is in global frame)
    d_theta = curr_theta - prev_theta
    R_local = np.array([[np.cos(prev_theta), -np.sin(prev_theta)],
                       [np.sin(prev_theta), np.cos(prev_theta)]])
    d_xy = np.dot(R_local.T, (curr_xy-prev_xy).reshape((-1,1)))

    # apply relative movement and convert to global frame
    R_global = np.array([[np.cos(Particles['poses'][2]), -np.sin(Particles['poses'][2])],
                        [np.sin(Particles['poses'][2]), np.cos(Particles['poses'][2])]])
    Particles['poses'][:2] += np.squeeze(np.einsum('ijk,il->ilk', R_global, d_xy))
    Particles['poses'][2] += d_theta

    # apply noise
    # noise = np.random.normal([0,0,0],Particles['noise_cov'], size=(Particles['nums'],3))
    noise = np.random.multivariate_normal([0,0,0], np.diag(Particles['noise_cov']), size=Particles['nums'])
    Particles['poses'] += noise.T # slightly incorrect but faster
    # R_global = np.array([[np.cos(Particles['poses'][2]), -np.sin(Particles['poses'][2])],
    #                      [np.sin(Particles['poses'][2]), np.cos(Particles['poses'][2])]])
    # Particles['poses'][:2] += np.squeeze(np.einsum('ijk,ik->jk', R_global, noise.T[:2]))
    # Particles['poses'][2] += noise.T[2]


def particle_update(Particles, Map, lidar_hit, joint_angles, body_angles):
    # hit for each particles (particle num,3,beam num)
    particles_hit = lidar2world(lidar_hit, joint_angles, body_angles, Particles=Particles)

    # get matching between map and particle lidar reading
    corr = np.zeros(Particles['nums'])
    for i in range(Particles['nums']):
        occ = world2map(particles_hit[:2,:,i], Map)
        corr[i] = np.sum(Map['map'][occ[1],occ[0]]>Map['occ_thres'])
    corr /= 10 # by divide, adding a temperature to the softmax function

    # update particle weights
    log_weights = np.log(Particles['weights']) + corr
    log_weights -= np.max(log_weights) + logsumexp(log_weights - np.max(log_weights))
    Particles['weights'] = np.exp(log_weights)

    # resampling if necessary
    # Note: there is a trade-off between particle accuracy and n_eff
    n_eff = np.sum(Particles['weights'])**2/np.sum(Particles['weights']**2)
    if n_eff<= Particles['n_eff']:
        particle_resample(Particles)

def particle_resample(Particles):
    # Stratified resampling reference: http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
    nums = Particles['nums']
    # normalize weight and get cum sum
    weight_sum = np.cumsum(Particles['weights'])
    weight_sum /= weight_sum[-1]

    # Generate N ordered random numbers
    random = (np.linspace(0, nums-1, nums) + np.random.uniform(size=nums))/nums

    # multinomial distribution
    new_sample = np.zeros(Particles['poses'].shape)
    sample = 0
    index = 0
    while(sample<nums):
        while (weight_sum[index]<random[sample]):
            index += 1
        new_sample[:,sample] = Particles['poses'][:,index]
        sample += 1
    Particles['poses'] = new_sample
    Particles['weights'] = np.ones(nums) / nums

    # same speed
    # sums, randoms = np.meshgrid(weight_sum, random)
    # diff = sums-randoms
    # diff[diff<0]=1
    # new_sample = Particles['poses'][:,np.argmin(diff,axis=1)]


def plot_all(Map, Trajectory, Lidar, Plot, idx = None):
    # paint occ, free and und
    occ_mask = Map['map']>Map['occ_thres']
    free_mask = Map['map']<Map['free_thres']
    und_mask = np.logical_not(np.logical_or(occ_mask, free_mask))
    Plot[occ_mask] = [0,0,0] # black for occ
    Plot[free_mask] = [255,255,255] # white for free
    Plot[und_mask] = [128,128,128] # gray for und

    # paint trajectory
    traj = np.asarray(Trajectory)[:,:2]
    traj_pixel = world2map(traj.T, Map)
    Plot[traj_pixel[1],traj_pixel[0]] = [255,0,0] # blue for trajectory

    # paint lidar
    lidar_pixel = world2map(Lidar[:2], Map)
    Plot[lidar_pixel[1], lidar_pixel[0]] = [0, 255, 0]  # green for lidar

    if idx is None:
        # show the plot
        cv2.imshow('SLAM', Plot)
        cv2.waitKey(10)
    else:
        # save the last plot
        cv2.imwrite('SLAM_'+str(idx)+'.png', Plot)

