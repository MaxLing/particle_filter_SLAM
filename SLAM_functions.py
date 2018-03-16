import numpy as np
import cv2

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

    # debug purpose
    occ_grid = 1 - 1 / (1 + np.exp(Map['map']))
    cv2.imshow('test', occ_grid)
    cv2.waitKey(0)


def map2plot(Map):
    occ_grid = 1-1/(1+np.exp(Map['map']))
    # find occupied and free grid
    occ = np.where(occ_grid>0.75)
    free = np.where(occ_grid<0.25)

    # transform origin from upper left to center, pixel to meter
    center = Map['size'] * Map['res'] / 2
    occ_x = (occ[0]-center)/Map['res']
    occ_y = (center-occ[1])/Map['res']
    free_x = (free[0]-center)/Map['res']
    free_y = (center-free[1])/Map['res']

    return np.vstack((occ_x, occ_y)), np.vstack((free_x, free_y))
