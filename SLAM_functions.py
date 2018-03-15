import numpy as np

def polar2cart(scan, angles):
    # good_scan = lidar_scan[good_range]
    # good_angles = lidar_angles[good_range]

    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    z = np.zeros(len(scan))
    return np.vstack((x, y, z))