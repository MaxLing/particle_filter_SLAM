import numpy as np
import cv2

def polar2cart(scan, angles):
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    z = np.zeros(len(scan))
    return np.vstack((x, y, z))

def lidar2world(lidar_hit, joint_angles, pose=None, Particles=None):
    neck_angle = joint_angles[0] # yaw wrt body frame
    head_angle = joint_angles[1] # pitch wrt body frame

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
        yaw_gb = pose[2]
        R_gb = np.array([[np.cos(yaw_gb), -np.sin(yaw_gb), 0, 0],
                        [np.sin(yaw_gb), np.cos(yaw_gb), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
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
        particles_hit = np.zeros((nums,3,lidar_hit.shape[1]))
        lidar_hit = np.vstack((lidar_hit, np.ones((1, lidar_hit.shape[1]))))

        for i in range(nums): # TODO: vectorize
            # body wrt world
            T_gb = np.array([[1, 0, 0, poses[0,i]], [0, 1, 0, poses[1,i]], [0, 0, 1, 0.93], [0, 0, 0, 1]])
            R_gb = np.array([[np.cos(poses[2,i]), -np.sin(poses[2,i]), 0, 0],
                            [np.sin(poses[2,i]), np.cos(poses[2,i]), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            H_gb = np.dot(T_gb, R_gb)

            # lidar wrt world
            H_gl = H_gb.dot(H_bh).dot(H_hl)
            world_hit = np.dot(H_gl, lidar_hit)[:3,:]

            # ground check, keep hits not on ground
            not_floor = world_hit[2] > 0.1
            particles_hit[i, :, :] = world_hit[:, not_floor]

        return particles_hit


def world2map(xy, Map):
    # transform origin from center to upper left, meter to pixel
    # xy[1] *= -1 # not sure why, but world y is already downward
    pixels = ((xy + Map['size']/2)*Map['res']).astype(np.int)

    # check boundary and keep pixels within
    center = Map['size']*Map['res']/2
    in_bound = np.logical_and(np.abs(pixels[0]-center) <= center, np.abs(pixels[1]-center) <= center)
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
    noise = np.random.normal([0,0,0],Particles['noise_cov'], size=(Particles['nums'],3))
    Particles['poses'] += noise.T

def particle_update(Particles, Map, lidar_hit, joint_angles):
    # hit for each particles (particle num,3,beam num)
    particles_hit = lidar2world(lidar_hit, joint_angles, Particles=Particles)
    


def plot_all(Map, Trajectory, Lidar, Plot):
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

    # show the plot
    cv2.imshow('SLAM', Plot)
    cv2.waitKey(10)

