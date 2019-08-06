import numpy as np

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from pyntcloud import PyntCloud

import os
from tqdm import tqdm
from pprint import pprint

def load_calib_cam2cam(filename, debug=False):
    """
    Only load R_rect & P_rect for neeed
    Parameters: filename of the calib file
    Return: 
        R_rect: a list of r_rect(shape:3*3)
        P_rect: a list of p_rect(shape:3*4)
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()
    
    R_rect = []
    P_rect = []

    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-4] == "R_rect":
            r_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r_r = np.reshape(r_r, (3,3))
            R_rect.append(r_r)
        elif title[:-4] == "P_rect":
            p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            p_r = np.reshape(p_r, (3,4))
            P_rect.append(p_r)
    
    if debug:
        print ("R_rect:")
        pprint (R_rect)

        print ()
        print ("P_rect:")
        pprint (P_rect)
    
    return R_rect, P_rect

def load_calib_lidar2cam(filename, debug=False):
    """
    Load calib
    Parameters: filename of the calib file
    Return:
        tr: shape(4*4)
            [  r   t
             0 0 0 1]

    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()
    
    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-1] == "R":
            r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r = np.reshape(r, (3,3))
        if title[:-1] == "T":
            t = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            t = np.reshape(t, (3,1))
    
    tr = np.hstack([r,t])
    tr = np.vstack([tr,np.array([0,0,0,1])])

    if debug:
        print ()
        print ("Tr:")
        print (tr)

    return tr

def cal_proj_matrix(filename_c2c, filename_l2c, camera_id, debug=False):
    """
    Compute the projection matrix from LiDAR to Img
    Parameters:
        filename_c2c: filename of the calib file for cam2cam
        filename_l2c: filename of the calib file for lidar2cam
        camera_id: the NO. of camera
    Return:
        P_lidar2img: the projection matrix from LiDAR to Img
    """
    R_rect, P_rect = load_calib_cam2cam(filename_c2c, debug)
    tr = load_calib_lidar2cam(filename_l2c, debug)

    R_cam2rect = np.hstack([np.array([[0],[0],[0]]),R_rect[0]])
    R_cam2rect = np.vstack([np.array([1,0,0,0]), R_cam2rect])
    
    P_lidar2img = np.matmul(P_rect[camera_id+1], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print ()
        print ("P_lidar2img:")
        print (P_lidar2img)

    return P_lidar2img

def show_img(name, img):
    """
    Show the image

    Parameters:    
        name: name of window    
        img: image
    """
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)

def load_img(filename, debug=False):
    """
    Load the image
    Parameter:
        filename: the filename of the image
    Return:
        img: image
    """
    img = cv2.imread(filename)
    
    if debug: show_img("Image", img)

    return img

def load_lidar(filename, debug=False):
    """
    Load the PointCloud
    Parameter:
        filename: the filename of the PointCloud
    Return:
        points: PointCloud associated with the image
    """
    # N*4 -> N*3
    points = np.fromfile(filename, dtype=np.float32)
    points = np.reshape(points, (-1,4))
    points = points[:, :3]
    points.tofile("./temp_pc.bin")

    # Remove all points behind image plane (approximation)
    cloud = PyntCloud.from_file("./temp_pc.bin")
    cloud.points = cloud.points[cloud.points["x"]>=5]
    points = np.array(cloud.points)

    if debug:
        print (points.shape)

    return points

def project_lidar2img(img, pc, p_matrix, debug=False):
    """
    Project the LiDAR PointCloud to Image
    Parameters:
        img: Image
        pc: PointCloud
        p_matrix: projection matrix
    """
    # Dimension of data & projection matrix
    dim_norm = p_matrix.shape[0]
    dim_proj = p_matrix.shape[1]

    # Do transformation in homogenuous coordinates
    pc_temp = pc.copy()
    if pc_temp.shape[1]<dim_proj:
        pc_temp = np.hstack([pc_temp, np.ones((pc_temp.shape[0],1))])
    points = np.matmul(p_matrix, pc_temp.T)
    points = points.T

    temp = np.reshape(points[:,dim_norm-1], (-1,1))
    points = points[:,:dim_norm]/(np.matmul(temp, np.ones([1,dim_norm])))

    # Plot
    if debug:
        depth_max = np.max(pc[:,0])
        for idx,i in enumerate(points):
            color = int((pc[idx,0]/depth_max)*255)
            cv2.rectangle(img, (int(i[0]-1),int(i[1]-1)), (int(i[0]+1),int(i[1]+1)), (0, 0, color), -1)
        show_img("Test", img)
    
    print (np.max(points[:,0]),np.max(points[:,1]))

    return points

def generate_colorpc(img, pc, pcimg, debug=False):
    """
    Generate the PointCloud with color
    Parameters:
        img: image
        pc: PointCloud
        pcimg: PointCloud project to image
    """
    x = pcimg[:,1]
    y = pcimg[:,0]
    xy = np.hstack(x,y)
    return 0

    









if __name__ == '__main__':
    calib_cam2cam = "./calib/calib_cam_to_cam.txt"
    calib_lidar2camera = "./calib/calib_velo_to_cam.txt"
    camera_id = 1

    filepath_img = "./img/0000000000.png"
    filepath_lidar = "./lidar/0000000000.bin"

    debug = False

    p_matrix = cal_proj_matrix(calib_cam2cam, calib_lidar2camera, camera_id, debug)
    img = load_img(filepath_img, debug)
    pc = load_lidar(filepath_lidar, debug)
    project_lidar2img(img, pc, p_matrix, debug)

    if debug:
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

