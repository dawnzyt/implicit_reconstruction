import json
import os
import sys

import cv2
import imageio
import numpy as np
import pyrender
import trimesh
from scipy.linalg import expm
from pyrender import OffscreenRenderer, PerspectiveCamera, Mesh, Scene
from tqdm import tqdm

def rot2quat(R):
    batch_size, _,_ = R.shape
    q = np.ones((batch_size, 4))

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=np.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q
def quat2rot(q):
    batch_size, _ = q.shape
    q = q/np.linalg.norm(q, axis=1, keepdims=True)
    R = np.ones((batch_size, 3,3))
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def pose_interpolation(p1, p2, t):
    # p1: (n, 4, 4), 批量左侧位姿
    # p2: (n, 4, 4), 批量右侧位姿
    # t: (m,), 线性插值参数

    # 改shape
    n,m=p1.shape[0],t.shape[0]
    p1=p1[:,None].repeat(m,axis=1).reshape(n*m,4,4)
    p2=p2[:,None].repeat(m,axis=1).reshape(n*m,4,4)
    t=t[None].repeat(n,axis=0).reshape(n*m)
    # 提取姿势中的平移向量和旋转矩阵
    T1, R1 = p1[:,:3, 3], p1[:,:3, :3]
    T2, R2 = p2[:,:3, 3], p2[:,:3, :3]

    # 线性插值平移向量和旋转矩阵
    inter_T = T1 * (1 - t[:,None]) + T2 * t[:,None]
    inter_R = np.array([R1[i]@expm(t[i]*(R1[i].T@R2[i])) for i in tqdm(range(n*m),desc='Interpolating poses...')]) # 旋转矩阵的线性插值：R1@(R1.T@R2)^t

    # 构建插值后的姿势矩阵
    inter_poses = np.eye(4, dtype=np.float32)[None].repeat(n*m,axis=0)
    inter_poses[:,:3, :3] = inter_R
    inter_poses[:,:3, 3] = inter_T

    return inter_poses

def interpolate_poses(poses, num_interpolations=5):
    # Interpolate between the poses to get more frames
    # poses: (N, 4, 4), camera-to-world poses
    # num_interpolations: int, number of frames to interpolate between each pair of poses
    poses_l=poses[:-1]
    poses_r=poses[1:]
    ts= np.linspace(0, 1, num_interpolations + 2)[:-1]
    inter_poses=pose_interpolation(poses_l, poses_r, ts)
    inter_poses = np.concatenate([inter_poses,poses[-1:,...]],axis=0) # 最后一个位姿
    return inter_poses


def get_poses(data_dir):
    poses = []
    with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
        data = json.load(f)
    for frame in data["frames"]:
        pose = np.array(frame["camtoworld"], dtype=np.float32)
        poses.append(pose)
    scale_mat = np.array(data["worldtogt"], dtype=np.float32)
    return np.array(poses),scale_mat

mesh_dir = '/data/projects/implicit_reconstruction/runs/scannet_normal_bg_0616_00/scannet_normal_bg_0616_00_2048_epoch_540.ply'
data_dir = '/data/scannet/scans/scene0616_00'
if_interpolate = True
gt_space = True
textured = True
num_interpolations = 11
fps = 30
width= 640
height= 480

mesh = trimesh.load_mesh(mesh_dir)
poses,scale_mat = get_poses(data_dir)
if if_interpolate:
    print('Will interpolate {} frames between each pair of poses'.format(num_interpolations))
    poses = interpolate_poses(poses, num_interpolations=num_interpolations) # 实际上是c2n

if gt_space:
    poses = np.matmul(scale_mat[None].repeat(poses.shape[0],axis=0),poses) # c2n/c2w-> c2gt

material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.8, 0.8, 0.8, 1])
if textured:
    m = pyrender.Mesh.from_trimesh(mesh)
else:
    m = pyrender.Mesh.from_trimesh(mesh, material=material)


# 创建一个场景
scene = pyrender.Scene()
scene.add(m) # 添加网格

# 相机
pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414) # 创建一个透视相机
oc= pyrender.OrthographicCamera(xmag=1.0, ymag=1.0) # 创建一个正交相机

camera_node = pyrender.Node(camera=pc, matrix=np.eye(4)) # 创建一个相机节点
scene.add_node(camera_node) # 将相机节点添加到场景中

# 创建光
pl = pyrender.PointLight(color=[0.9, 0.9, 0.9], intensity=0.3) # 创建点光源
dl = pyrender.DirectionalLight(color=[0.9, 0.9, 0.9], intensity=6) # 创建平行直线光源

# 创建光节点
dl_node = pyrender.Node(light=dl, matrix=np.eye(4))
pl_node = pyrender.Node(light=pl, matrix=np.eye(4))
# 将光源添加到场景中
scene.add_node(dl_node)
scene.add_node(pl_node)

# 创建offscreen渲染器
renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)


# 设置视频和帧率
fourcc = cv2.VideoWriter_fourcc(*'MPEG') # 保存视频的编码: XVID, MPEG
color_out = cv2.VideoWriter(os.path.join('.',os.path.basename(mesh_dir).split('.')[0]+'.avi'), fourcc, fps, (width, height))
depths = []

# 遍历所有相机位姿，渲染每一帧并写入视频
for pose in tqdm(poses,total=len(poses),desc='Rendering rgb',unit='frames',colour='blue'):
    # pose由OPENCV->OPENGL
    pose[:3,1:3] = -pose[:3,1:3]
    # 设置相机的位置和方向
    scene.set_pose(camera_node, pose=pose)
    scene.set_pose(dl_node, pose=pose)


    # 渲染场景
    color, depth = renderer.render(scene)
    depths.append(depth)

    # 将渲染的帧添加到视频
    color_out.write(color)

# TODO: depth这里有问题，不知道什么原因一直在闪。
# depth_min = np.min(np.array(depths))
# depth_max = np.max(np.array(depths))
# depth_out = cv2.VideoWriter(os.path.join('.','mesh_depth.avi'), fourcc, fps, (width, height), isColor=True) # isColor=True: 彩色视频, isColor=False: 灰度视频
# for depth in tqdm(depths,total=len(depths),desc='Rendering depth',unit='frames',colour='green'):
#     # depth: float[0,1]单位m -> 16位无符号整数[0,65535]单位mm
#     # depth = (depth * 1000).astype(np.uint16)
#     depth = (depth-depth_min)/(depth_max-depth_min)
#     depth_image_color = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     depth_out.write(depth_image_color)

# 完成视频写入
# writer.close()
color_out.release()
# depth_out.release()

# 清理渲染器资源
renderer.delete()