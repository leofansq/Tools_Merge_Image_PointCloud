"""
 @leofansq

 Basic function:
    show_img(name, img): Show the image
    find_files(directory, pattern): Method to find target files in one directory, including subdirectory
 Load function:
    load_calib_cam2cam(filename, debug=False): Only load R_rect & P_rect for need
    load_calib_lidar2cam(filename, debug=False): Load calib parameters for LiDAR2Cam
    load_calib(filename, debug=False): Load the calib parameters which has R_rect & P_rect & Tr in the same file
    load_img(filename, debug=False): Load the image
    load_lidar(filename, debug=False): Load the PointCloud
 Process function:
    cal_proj_matrix_raw(filename_c2c, filename_l2c, camera_id, debug=False): Compute the projection matrix from LiDAR to Img
    cal_proj_matrix(filename, camera_id, debug=False): Compute the projection matrix from LiDAR to Image
    project_lidar2img(img, pc, p_matrix, debug=False): Project the LiDAR PointCloud to Image
    generate_colorpc(img, pc, pcimg, debug=False): Generate the PointCloud with color
    save_pcd(filename, pc_color): Save the PointCloud with color in the term of .pcd
"""
import cv2
import numpy as np
from pyntcloud import PyntCloud

import os
import fnmatch
from tqdm import tqdm
from pprint import pprint

#**********************************************************#
#                    Basic Function                        #
#**********************************************************#

def show_img(name, img):
    """
    Show the image

    Parameters:    
        name: name of window    
        img: image
    """
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)

def find_files(directory, pattern):
    """
    Method to find target files in one directory, including subdirectory
    :param directory: path
    :param pattern: filter pattern
    :return: target file path list
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                file_list.append(filename)
    
    return file_list

#**********************************************************#
#                     Load Function                        #
#**********************************************************#

def load_calib_cam2cam(filename, debug=False):
    """
    Only load R_rect & P_rect for need
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
    Load calib parameters for LiDAR2Cam
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

def load_calib(filename, debug=False):
    """
    Load the calib parameters which has R_rect & P_rect & Tr in the same file
    Parameters:
        filename: the filename of the calib file
    Return:
        R_rect, P_rect, Tr
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()
    
        P_rect = []    
    for line in lines:
        title = line.strip().split(' ')[0]
        if len(title):
            if title[0] == "R":
                R_rect = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                R_rect = np.reshape(R_rect, (3,3))
            elif title[0] == "P":
                p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                p_r = np.reshape(p_r, (3,4))
                P_rect.append(p_r)
            elif title[:-1] == "Tr_velo_to_cam":
                Tr = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                Tr = np.reshape(Tr, (3,4))
                Tr = np.vstack([Tr,np.array([0,0,0,1])])
    
    return R_rect, P_rect, Tr

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
    cloud.points = cloud.points[cloud.points["x"]>=0]
    points = np.array(cloud.points)

    if debug:
        print (points.shape)

    return points

#**********************************************************#
#                   Process Function                       #
#**********************************************************#

def cal_proj_matrix_raw(filename_c2c, filename_l2c, camera_id, debug=False):
    """
    Compute the projection matrix from LiDAR to Img
    Parameters:
        filename_c2c: filename of the calib file for cam2cam
        filename_l2c: filename of the calib file for lidar2cam
        camera_id: the NO. of camera
    Return:
        P_lidar2img: the projection matrix from LiDAR to Img
    """
    # Load Calib Parameters
    R_rect, P_rect = load_calib_cam2cam(filename_c2c, debug)
    tr = load_calib_lidar2cam(filename_l2c, debug)

    # Calculation
    R_cam2rect = np.hstack([np.array([[0],[0],[0]]),R_rect[0]])
    R_cam2rect = np.vstack([np.array([1,0,0,0]), R_cam2rect])
    
    P_lidar2img = np.matmul(P_rect[camera_id], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print ()
        print ("P_lidar2img:")
        print (P_lidar2img)

    return P_lidar2img

def cal_proj_matrix(filename, camera_id, debug=False):
    """
    Compute the projection matrix from LiDAR to Image
    Parameters:
        filename: filename of the calib file
        camera_id: the NO. of camera
    Return:
        P_lidar2img: the projection matrix from LiDAR to Image
    """
    # Load Calib Parameters
    R_rect, P_rect, tr = load_calib(filename, debug)

    # Calculation
    R_cam2rect = np.hstack([np.array([[0],[0],[0]]),R_rect])
    R_cam2rect = np.vstack([np.array([1,0,0,0]), R_cam2rect])
    
    P_lidar2img = np.matmul(P_rect[camera_id], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print ()
        print ("P_lidar2img:")
        print (P_lidar2img)

    return P_lidar2img

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

    return points

def generate_colorpc(img, pc, pcimg, debug=False):
    """
    Generate the PointCloud with color
    Parameters:
        img: image
        pc: PointCloud
        pcimg: PointCloud project to image
    Return:
        pc_color: PointCloud with color e.g. X Y Z R G B
    """
    x = np.reshape(pcimg[:,0], (-1,1))
    y = np.reshape(pcimg[:,1], (-1,1))
    xy = np.hstack([x,y])

    pc_color = []
    for idx, i in enumerate(xy):
        if (i[0]>1 and i[0]<img.shape[1]) and (i[1]>1 and i[1]<img.shape[0]):            
            p_color = [pc[idx][0], pc[idx][1], pc[idx][2], img[i[1],i[0]][2], img[i[1],i[0]][1], img[i[1],i[0]][0]]
            pc_color.append(p_color)
    pc_color = np.array(pc_color)

    return pc_color

def save_pcd(filename, pc_color):
    """
    Save the PointCloud with color in the term of .pcd
    Parameter:
        filename: filename of the pcd file
        pc_color: PointCloud with color
    """
    f = open(filename, "w")

    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z rgb\n")
    f.write("SIZE 4 4 4 4\n")
    f.write("TYPE F F F F\n")
    f.write("COUNT 1 1 1 1\n")
    f.write("WIDTH {}\n".format(pc_color.shape[0]))
    f.write("HEIGHT 1\n")
    f.write("POINTS {}\n".format(pc_color.shape[0]))
    f.write("DATA ascii\n")

    for i in pc_color:
        rgb = (int(i[3])<<16) | (int(i[4])<<8) | (int(i[5]))
        f.write("{:.6f} {:.6f} {:.6f} {}\n".format(i[0],i[1],i[2],rgb))
    
    f.close()


if __name__ == '__main__':  
    # Option
    calib_cam2cam = "./calib/calib_cam_to_cam.txt"
    calib_lidar2camera = "./calib/calib_velo_to_cam.txt"
    camera_id = 1

    # filepath_img = "./img/0000000000.png"
    filepath_img = "./new.png"
    filepath_lidar = "./lidar/0000000022.bin"
    filename_save = "./test.pcd"

    debug = False

    # Process
    p_matrix = cal_proj_matrix_raw(calib_cam2cam, calib_lidar2camera, camera_id, debug)
    img = load_img(filepath_img, debug)
    # img = img[0:150,0:500]
    pc = load_lidar(filepath_lidar, debug)
    pcimg = project_lidar2img(img, pc, p_matrix, debug)
    pc_color = generate_colorpc(img, pc, pcimg)
    save_pcd(filename_save, pc_color)

    if debug:
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

