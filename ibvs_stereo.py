from RVC3 import *
import pickle
import pandas as pd
from machinevisiontoolbox import *
from machinevisiontoolbox.base import *
import cv2 as cv
from glob import glob
from matplotlib import pyplot as plt

def build_camera_model(intrinsic_matrix, image_shape, pose = None):
    u0 = intrinsic_matrix[0, 2]
    v0 = intrinsic_matrix[1, 2]
    px = intrinsic_matrix[0, 0]
    py = intrinsic_matrix[1, 1]
    camera = CentralCamera(f = [px, py], pp = [u0, v0], imagesize = image_shape, pose = pose);
    return camera

# class IBVS():
#     def __init__(self, camera):
#         self.camera = camera
#     def run(self, n_runs = 50, lmbda = 0.1):
#         camera_poses = []
#         for _ in range(n_runs):
#             p = self.camera.project_point(points3D_c)
#             pt = self.camera.pose.inv() * points3D_c
#             J = self.camera.visjac_p(p, pt[2, :])
#             e = points2D_d - p
#             v = lmbda * np.linalg.pinv(J) @ e.flatten(order="F")
#             self.camera.pose = self.camera.pose @ SE3.Delta(v)
#             camera_poses.append(self.camera.pose.A)
#         return camera_poses
from scipy.spatial.transform import Rotation as R

def homogeneous_transform(rotmat,vect):

    '''
    :param rotmat: 3x3 matrix
    :param vect: list x,y,z
    :return:homogeneous transformation 4x4 matrix using R and vect
    '''
    if not isinstance(vect, list):
        vect = list(vect)
    H = np.zeros((4,4))
    H[0:3,0:3] = rotmat
    frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1,4)
    H[:,3] = D
    return H

def inverse_homogeneous_transform(H):

    '''
    :param H: homogeneous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    '''


    rotmat = H[0:3,0:3]
    origin = H[:-1,3]
    origin.shape = (3,1)

    rotmat = rotmat.T
    origin = -rotmat.dot(origin)
    return homogeneous_transform(rotmat,list(origin.flatten()))

def vec2homo(vec):
    rotmat = R.from_rotvec(vec[3:]).as_matrix()
    t = vec[:3]
    return homogeneous_transform(rotmat, t)

def homo2vec(H):
    rotmat = H[:3, :3]
    rotvec = R.from_matrix(rotmat).as_rotvec()
    t = H[:3, 2]
    return np.concatenate([t, rotvec])

def homogeneous_position(vect):
    '''
    This function will make sure the coordinates in the right shape(4 by N) that could be multiplied by a
    homogeneous transformation(4 by 4).

    Parameters
    ----------
    vect: list or np.array
        A N by 3 array. x, y, z coordinates of the N points

    Return
    ------
    A 4 by N array.
    '''
    temp = np.array(vect)
    if temp.ndim == 1:
        ones = np.ones(1)
        return np.r_[temp, ones].reshape(-1, 1)
    elif temp.ndim == 2:
        num_rows, num_cols = temp.shape
        if num_cols != 3:
            raise Exception(f"vect is not N by 3, it is {num_rows}x{num_cols}")

        ones = np.ones(num_rows).reshape(-1, 1)
        return np.c_[temp, ones].T

def load_data(current_dir, target_dir, individuals = ['bolt1']):
    ### Load data ###
    scale = 1000
    # f_3d_c = f'C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/{id_t}/3d_combined/markers_trajectory_{cam}_3d.h5'
    # f_2d_d = f'C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/{id_d}/2d_combined/markers_trajectory_{cam}_2d.h5'
    f_3d_c = os.path.join(current_dir, '3d_combined/markers_trajectory_wrist_3d.h5')
    f_2d_d = os.path.join(target_dir, '2d_combined/markers_trajectory_wrist_2d.h5')
    df_2D_d = pd.read_hdf(f_2d_d)
    df_3D_c = pd.read_hdf(f_3d_c)

    points3D_c = []
    points2D_d_left = []  ## d for desired point for left camera
    points2D_d_right = []  ## d for desired point for left camera

    for individual in individuals:
        bps = df_2D_d[individual].columns.get_level_values(level='bodyparts').unique()
        for bp in bps:
            if not np.isnan(df_3D_c[(individual, bp)].iloc[0]).any() :
                points2D_d_left.append([df_2D_d[(individual, bp, 'x')].iloc[0], df_2D_d[(individual, bp, 'y')].iloc[0]])
                points2D_d_right.append([df_2D_d[(individual, bp, 'x')].iloc[1], df_2D_d[(individual, bp, 'y')].iloc[1]])
                points3D_c.append([df_3D_c[(individual, bp, 'x')].iloc[0], df_3D_c[(individual, bp, 'y')].iloc[0],
                                   df_3D_c[(individual, bp, 'z')].iloc[0]])

    points2D_d_left = np.array(points2D_d_left).T
    points2D_d_right = np.array(points2D_d_right).T
    points3D_c = np.array(points3D_c).T / scale

    return points2D_d_left, points2D_d_right, points3D_c

def main(current_dir, target_dir, individuals, lmbda, thresh):
    ### Get camera model ###
    calibration_date = '2024-08-20'
    camera_params_file = f'C:/Users/xyao0/Desktop/project/calibrate_camera/camera_matrix/{calibration_date}/stereo_params.pickle'
    # with open('C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/current/current_tcp_in_base.pickle',
    #           'rb') as f:
    #     vec_tcp_in_base = pickle.load(f)
    # with open('C:/Users/xyao0/Desktop/project/assembly/transformations/2024-08-28/wrist_cam_in_tcp.pickle', 'rb') as f:
    #     H_cam_in_tcp = pickle.load(f)
    ### Get tcp poses

    #
    # with open('C:/Users/xyao0/Desktop/project/assembly/transformations/2024-08-28/wrist_cam_in_tcp.pickle', 'rb') as f:
    #     H_cam_in_tcp = pickle.load(f)
    # with open('C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/current/current_tcp_in_base.pickle',
    #           'rb') as f:
    #     vec_tcp_in_base = pickle.load(f)
    #
    # H_tcp_in_base = vec2homo(vec_tcp_in_base)
    # H_cam_in_tcp[:-1, -1] /= 1000
    # H_tcp_in_cam = inverse_homogeneous_transform(H_cam_in_tcp)

    with open(camera_params_file, 'rb') as f:
        camera_params = pickle.load(f)
    intrinsicMatrix1 = camera_params['left-right']['cameraMatrix1']
    intrinsicMatrix2 = camera_params['left-right']['cameraMatrix2']
    intrinsicMatrix1_rectified = camera_params['left-right']['P1'][:3, :3]
    intrinsicMatrix2_rectified = camera_params['left-right']['P2'][:3, :3]
    image_shape = camera_params['left-right']['image_shape'][0]
    P1 = camera_params['left-right']['P1']
    R1 = camera_params['left-right']['R1']
    dist1 = camera_params['left-right']['distCoeffs1']
    P2 = camera_params['left-right']['P2']
    R2 = camera_params['left-right']['R2']
    dist2 = camera_params['left-right']['distCoeffs2']
    translation_rectified = P2[0, -1] / 1000 / P2[0, 0]  ### Translation is scaled by the focus, so undo it by dividing by P2[0, 0]. Divide by 1000 so that unit is meter.
    right_T_left_rectified = SE3.Tx(translation_rectified)  ### This is the homogeneous transfomation that represents rectified left camera in rectified right camera
    camera_left = build_camera_model(intrinsicMatrix1_rectified, image_shape)
    camera_right = build_camera_model(intrinsicMatrix2_rectified, image_shape, pose = right_T_left_rectified.inv())


    # leftMapX = camera_params['left-right']['MapX1']
    # leftMapY = camera_params['left-right']['MapY1']
    # rightMapX = camera_params['left-right']['MapX2']
    # rightMapY = camera_params['left-right']['MapY2']

    points2D_d_left, points2D_d_right, points3D_c = load_data(current_dir, target_dir, individuals = individuals)
    points2D_d_rectified_left = cv.undistortPoints(src=points2D_d_left, cameraMatrix=intrinsicMatrix1, distCoeffs=dist1, P=P1, R=R1).squeeze().T
    points2D_d_rectified_right = cv.undistortPoints(src=points2D_d_right, cameraMatrix=intrinsicMatrix2, distCoeffs=dist2, P=P2, R=R2).squeeze().T

    # img_left = glob(f'C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/target/2d_combined/*left*.png')[0]
    #
    # imgL = cv.imread(img_left)
    # Left_rectified = cv.remap(imgL, leftMapX, leftMapY, cv.INTER_LINEAR)
    # # plt.figure()
    # camera_left.plot_point(points2D_d_rectified_left.T);
    # plt.imshow(Left_rectified)
    # plt.show()

    # img_right = glob(f'C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/target/2d_combined/*right*.png')[0]
    #
    # imgR = cv.imread(img_right)
    # Right_rectified = cv.remap(imgR, rightMapX, rightMapY, cv.INTER_LINEAR)
    # # plt.figure()
    # camera_right.plot_point(points2D_d_rectified_right.T);
    # plt.imshow(Right_rectified)
    # plt.show()
    # raise

    n_run = 20

    # points3D_c_right = points3D_c_left.copy()
    # points3D_c_right[0, :] = points3D_c_right[0, :] + translation_rectified
    # print(points3D_c_right)
    # raise

    adj_matrix = right_T_left_rectified.Ad() ### adjoint matrix used to transform velocity in right camera to left camera
    cam_poses = []
    finished = False
    for i in range(n_run):
        p_left = camera_left.project_point(points3D_c)
        p_right = camera_right.project_point(points3D_c)

        depth = points3D_c[-1, :]
        J_left = camera_left.visjac_p(p_left, depth)
        J_right = camera_right.visjac_p(p_right, depth)
        J_right_to_left = J_right @ adj_matrix
        # e_left = points2D_d_rectified_left - p_left
        # e_right = points2D_d_rectified_right - p_right
        p = np.concatenate([p_left, p_right], axis = 1)
        J = np.concatenate([J_left, J_right_to_left], axis = 0)
        points2D_d_rectified = np.concatenate([points2D_d_rectified_left, points2D_d_rectified_right], axis = 1)
        e = points2D_d_rectified - p
        print(e.T)
        print('ffffffffff')
        if np.max(np.abs(e)) < thresh:
            cam_poses.append(camera_left.pose.A)
            finished = True
            break
        v = lmbda * np.linalg.pinv(J) @ e.flatten(order="F")
        # v_left = lmbda * np.linalg.pinv(J_left) @ e_left.flatten(order="F")
        # v_right = lmbda * np.linalg.pinv(J_right) @ e_right.flatten(order="F")
        # v = v_right
        # v = 0.5 * (v_left + v_right)
        camera_right.plot_point(points3D_c)
        camera_left.plot_point(points3D_c)

        # camera_right.plot_point(points3D_c)
        # cam_pose  = camera_left.pose.A @ SE3.Delta(v).A
        # tcp_pose = H_tcp_in_base @ H_cam_in_tcp @ cam_pose @ H_tcp_in_cam
        # tcp_poses.append(tcp_pose)
        # tcp_poses_vec.append(homo2vec(tcp_pose))

        cam_poses.append(camera_left.pose.A @ SE3.Delta(v).A)
        camera_left.pose = camera_left.pose @ SE3.Delta(v)
        camera_right.pose = camera_right.pose @ right_T_left_rectified @ SE3.Delta(v) @ right_T_left_rectified.inv()
    # tcp_poses_vec = np.array(tcp_poses_vec)

    plt.show()
    # cam_poses = []
    # for hist in ibvs.history:
    #     cam_poses.append(hist.pose.A)
    destdir = 'C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/plan'
    os.makedirs(destdir, exist_ok=True)
    cam_poses.append(finished)
    with open(os.path.join(destdir, 'cam_poses.pickle'), 'wb') as f:
        pickle.dump(cam_poses, f)
if __name__ == '__main__':
    import sys
    import ast
    current_dir = sys.argv[1]
    target_dir = sys.argv[2]
    individuals = ast.literal_eval(sys.argv[3])
    lmbda = float(sys.argv[4])
    thresh = float(sys.argv[5])
    main(current_dir, target_dir, individuals, lmbda, thresh)


