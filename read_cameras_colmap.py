import numpy as np
import os
import matplotlib.pyplot as plt
import collections
import open3d as o3d

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_images_txt(images_path):
    if not os.path.exists(images_path):
        raise Exception(f"No such file : {images_path}")

    with open(images_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {images_path}")

    comments = lines[:4]
    contents = lines[4:]

    img_ids = []
    img_names = []
    t_poses = []
    R_poses = []
    

    for img_idx, content in enumerate(contents[::2]):
        content_items = content.split(' ')
        img_id = content_items[0]
        q_wxyz = np.array(content_items[1:5], dtype=np.float32) # colmap uses wxyz
        t_xyz = np.array(content_items[5:8], dtype=np.float32)
        #Transform a quaternion into a rotation matrix following Hamilton convention
        R = qvec2rotmat(q_wxyz)
        t = -R.T @ t_xyz
        R = R.T
        img_name = content_items[9]

        img_ids.append(img_id)
        img_names.append(img_name)
        t_poses.append(t)
        R_poses.append(R)

    return img_ids, img_names, t_poses, R_poses

def plot_cameras_colmap(img_names, R_poses, t_poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(img_names)):
        T = np.column_stack((R_poses[i], t_poses[i]))
        T = np.vstack((T, (0, 0, 0, 1)))
        cam_pos = T[:3, 3]
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='r', marker='o')
        #Add a text in each point of the scatter plot
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], str(i), size=10, zorder=1, color='k')
    plt.show()



camera = read_cameras_text('/home/lmur/Documents/Monodepth/sequences/P02_101_colmap/cameras.txt')
cam = camera[1]

if cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
    fx = cam.params[0]
    fy = cam.params[1]
    cx = cam.params[2]
    cy = cam.params[3]

# intrinsics
K_int = np.identity(3)
K_int[0, 0] = fx
K_int[1, 1] = fy
K_int[0, 2] = cx
K_int[1, 2] = cy
K_inv = np.linalg.inv(K_int)

img_ids, img_names, t_poses, R_poses = read_images_txt('/home/lmur/Desktop/EPIC_KITCHENS_100_SEQUENCES/P02_101_part_1/sparse/images.txt')
plot_cameras_colmap(img_names, R_poses, t_poses)

"""
visor = o3d.visualization.Visualizer()
visor.create_window()
for i in range(len(img_names)):
    T = np.column_stack((R_poses[i], t_poses[i]))
    T = np.vstack((T, (0, 0, 0, 1)))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    axis.transform(T)
    visor.add_geometry(axis)
visor.poll_events()
visor.update_renderer()
visor.run()
"""


