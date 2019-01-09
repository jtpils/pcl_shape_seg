#!/usr/bin/env python3

import os

import numpy as np
import open3d as o3

# Colors
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
BROWN = (91, 68, 62)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
ORANGE = (255, 140, 0)
PINK = (255, 105, 180)
RED = (255, 0, 0)

# Axes
X_AXIS = (1, 0, 0)
Y_AXIS = (0, 1, 0)
Z_AXIS = (0, 0, 1)

sem_seg_dir = os.path.dirname(os.path.abspath('.'))
data_dir = 'data'


def flatten(lst):
    for elt in lst:
        try:
            yield from flatten(elt)
        except TypeError:
            yield elt


def draw(clouds):
    try:
        o3.draw_geometries(list(flatten(clouds)))
    except TypeError:
        o3.draw_geometries([clouds])


def make_cloud(points, color=BLUE):
    cloud = o3.PointCloud()
    cloud.points = o3.Vector3dVector(np.asarray(points, dtype=np.float32))
    colors = [color for _ in points]
    cloud.colors = o3.Vector3dVector(np.asarray(colors, dtype=np.float32))
    return cloud


def set_colors(cloud, color=BLUE):
    colors = [color for _ in cloud.points]
    cloud.colors = o3.Vector3dVector(np.asarray(colors))


def rotation_matrix(axis, theta):
    """ Return the rotation matrix associated with counterclockwise
    rotation about the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate(cloud, axis, theta):
    points = np.copy(np.asarray(cloud.points))
    points = [np.dot(rotation_matrix(axis, theta), x) for x in points]
    cloud.points = o3.Vector3dVector(np.asarray(points, dtype=np.float32))


def translate(cloud, translation):
    points = np.asarray(cloud.points)
    points = points + translation
    cloud.points = o3.Vector3dVector(np.asarray(points, dtype=np.float32))


def create_eighth_sphere_utils(r=1, NUM_POINTS=4096):
    u = np.random.uniform(0, 0.25, NUM_POINTS)
    v = np.random.uniform(0.5, 1, NUM_POINTS)
    thetas = 2 * np.pi * u
    phis = np.arccos(2 * v - 1)

    x = r * np.cos(thetas) * np.sin(phis)
    y = r * np.sin(thetas) * np.sin(phis)
    z = r * np.cos(phis)
    points = list(zip(x, y, z))
    cloud = make_cloud(points, color=RED)
    return cloud


def create_coordinate_system(NUM_POINTS=4096):
    x = np.random.uniform(0, 1, NUM_POINTS)
    y = [0 for _ in range(NUM_POINTS)]
    z = [0 for _ in range(NUM_POINTS)]
    points = list(zip(x, y, z))
    x_axis = make_cloud(points, color=RED)

    x = [0 for _ in range(NUM_POINTS)]
    y = np.random.uniform(0, 1, NUM_POINTS)
    z = [0 for _ in range(NUM_POINTS)]
    points = list(zip(x, y, z))
    y_axis = make_cloud(points, color=BLUE)

    x = [0 for _ in range(NUM_POINTS)]
    y = [0 for _ in range(NUM_POINTS)]
    z = np.random.uniform(0, -1, NUM_POINTS)
    points = list(zip(x, y, z))
    z_axis = make_cloud(points, color=GREEN)

    points = np.concatenate([x_axis.points, y_axis.points, z_axis.points])
    points = points + (-3, 3, 0)
    colors = np.concatenate([x_axis.colors, y_axis.colors, z_axis.colors])
    cloud = o3.PointCloud()
    cloud.points = o3.Vector3dVector(np.asarray(points, dtype=np.float32))
    cloud.colors = o3.Vector3dVector(np.asarray(colors, dtype=np.float32))

    return cloud


def viz_eighth_sphere_rotate_x_axis():
    coordinate_system = create_coordinate_system()

    zero = create_eighth_sphere_utils()
    set_colors(zero, RED)
    ninety = create_eighth_sphere_utils()
    set_colors(ninety, BLUE)
    rotate(ninety, X_AXIS, np.pi / 2)
    one_eighty = create_eighth_sphere_utils()
    set_colors(one_eighty, ORANGE)
    rotate(one_eighty, X_AXIS, np.pi)
    two_seventy = create_eighth_sphere_utils()
    set_colors(two_seventy, MAGENTA)
    rotate(two_seventy, X_AXIS, 1.5 * np.pi)
    draw([coordinate_system, zero, ninety, one_eighty, two_seventy])


def combine_clouds(clouds):
    points = []
    colors = []
    for cloud in list(flatten(clouds)):
        points.append(cloud.points)
        colors.append(cloud.colors)
    points = np.concatenate(points)
    colors = np.concatenate(colors)
    cloud.points = o3.Vector3dVector(np.asarray(points, dtype=np.float32))
    cloud.colors = o3.Vector3dVector(np.asarray(colors, dtype=np.float32))
    return cloud


def extract_min_max(cloud):
    min_x = min(cloud.points, key=(lambda point: point[0]))[0]
    max_x = max(cloud.points, key=(lambda point: point[0]))[0]
    min_y = min(cloud.points, key=(lambda point: point[1]))[1]
    max_y = max(cloud.points, key=(lambda point: point[1]))[1]
    min_z = min(cloud.points, key=(lambda point: point[2]))[2]
    max_z = max(cloud.points, key=(lambda point: point[2]))[2] 
    
    return (min_x, min_y, min_z), (max_x, max_y, max_z)
