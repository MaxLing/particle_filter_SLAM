import numpy as np
import cv2

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

    Trajectory = []

    return Map, Particles, Trajectory

def polar2cart(scan, angles):
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    z = np.zeros(len(scan))
    return np.vstack((x, y, z))

def lidar2world(lidar_hit, joint_angles, pose):
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

    # body wrt world
    x_gb = pose[0]
    y_gb = pose[1]
    z_gb = 0.93
    yaw_gb = pose[2]
    R_gb = np.array([[np.cos(yaw_gb), -np.sin(yaw_gb), 0, 0],
                    [np.sin(yaw_gb), np.cos(yaw_gb), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    T_gb = np.array([[1,0,0,x_gb],[0,1,0,y_gb],[0,0,1,z_gb],[0,0,0,1]])
    H_gb = np.dot(T_gb, R_gb)

    # lidar wrt world
    H_gl = H_gb.dot(H_bh).dot(H_hl)
    lidar_hit = np.vstack((lidar_hit,np.ones((1,lidar_hit.shape[1])))) # 4*n
    world_hit = np.dot(H_gl, lidar_hit)

    return world_hit[:3,:]

def world2map(xy, Map):
    # transform origin from center to upper left, meter to pixel
    xy_map = np.copy(xy)
    xy_map[1] *= -1
    pixels = ((xy_map + Map['size']/2)*Map['res']).astype(np.int)

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

    # # for debug purpose
    # occ_grid = 1 - 1 / (1 + np.exp(Map['map']))
    # cv2.imshow('test', occ_grid)
    # cv2.waitKey(0)








def plot_all(Map, Trajectory, Plot):
    # paint occ, free and und
    prob_map = 1-1/(1+np.exp(Map['map']))
    occ_mask = prob_map>0.7
    free_mask = prob_map<0.3
    und_mask = np.logical_not(np.logical_or(occ_mask, free_mask))
    Plot[occ_mask] = [0,0,0] # black for occ
    Plot[free_mask] = [255,255,255] # white for free
    Plot[und_mask] = [128,128,128] # gray for und

    # show the plot
    cv2.imshow('SLAM', Plot)
    cv2.waitKey(10)

