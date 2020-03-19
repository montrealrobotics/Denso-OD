import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def TwoDtoThreeD(samples, matrix):
    Threedboxes = []
    CAM_HEIGHT = 1.72
    # CAM_HEIGHT = 1.65
    for bottom_xy in samples:
        bottom_xy = np.append(bottom_xy, 1.0).T
        n = np.asarray([0, -1., 0])
        numerator = np.linalg.solve(matrix, bottom_xy)
        denominator = n.dot(numerator)
        bboxBottomIn3D = np.reshape(-CAM_HEIGHT * (numerator/denominator), (3,1))
        Threedboxes.append(bboxBottomIn3D[[0,2]])
    return np.array(Threedboxes)

def read_matrix(path):
    file_path = "/home/dishank/denso-ws/src/denso/datasets/kitti_tracking/training/calib/0001.txt"
    file = open(file_path)
    lines = file.read().splitlines()
    p_matrix = lines[2].split(':', 1)[1].strip()
    matrix = np.array(p_matrix.split(' '), dtype='float').reshape((3,4))
    matrix = matrix[:,:-1]

    return matrix
            
def ground_project(instances, path="/home/dishank/denso-ws/src/denso/datasets/kitti_tracking/training/calib/0001.txt"):
    means = instances.pred_boxes
    means = [[(x[0]+x[2])/2,x[3]] for x in means]
    sigmas = instances.pred_variance
    sigmas = [[(y[0]+y[2])/4, y[3]] for y in sigmas]
    K_matrix = read_matrix(path)
    gd_means=[]
    gd_sigmas = []

    for mean, sigma in zip(means, sigmas): #iterating gthrough each of the box
        samples = np.random.normal(np.full((10,2), mean), sigma) #Draw 10 samples
        ground_points = TwoDtoThreeD(samples, K_matrix)
        gd_mean = ground_points.mean(axis=0, keepdims=False)
        gd_std = ground_points.std(axis=0, keepdims=False)
        gd_means.append(np.squeeze(gd_mean))
        gd_sigmas.append(np.squeeze(gd_std))

    return np.asarray(gd_means), np.asarray(gd_sigmas)
