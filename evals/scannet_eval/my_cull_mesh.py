import argparse
import json

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import glob
import os
import pyrender
import os
from tqdm import tqdm
from pathlib import Path

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# hard-coded image size
H, W = 968, 1296
data_dir = '/data/scannet/scans'
# load pose

def load_poses(scan_id):
    pose_path = os.path.join(data_dir,f'scene{scan_id}', 'pose')
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for pose_path in pose_paths[::10]:
        c2w = np.loadtxt(pose_path)
        if np.isfinite(c2w).any():
            poses.append(c2w)
    poses = np.array(poses)

    return poses
class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


# refuse func: 将mesh render到每个pose上，然后用TSDF算法进行refuse，得到视角过滤后的mesh
# 类似cull mesh过程
def cull_mesh(mesh, poses, K, h, w):
    pass
    out_scene_mask = np.ones(shape=mesh.vertices.shape[0], dtype=bool)

    for i,c2w in tqdm(enumerate(poses),total=len(poses)):
        w2c = np.linalg.inv(c2w)
        verts = mesh.vertices.T
        homo_verts = np.concatenate([verts,np.ones(shape=(1,verts.shape[1]))],axis=0)
        pix_verts = K@(w2c[:3]@homo_verts)
        pix = pix_verts[:2,:] / (pix_verts[2:,:]+1e-6)
        # ~mask: 在视锥外或在图像外
        mask = (pix_verts[-1,:]>=0)& ((pix[0,:]>=0)&(pix[0,:]<=w)&(pix[1,:]>=0)&(pix[1,:]<=h))
        out_scene_mask  =out_scene_mask&(~mask)
    verts,faces = mesh.vertices,mesh.faces
    face_mask = (~out_scene_mask)[faces].all(axis=1)
    mesh.update_faces(face_mask)
    return mesh

def main(args):
    # load mesh
    mesh = trimesh.load(args.mesh)

    idx=args.scan_id
    with open(os.path.join(args.data_dir,f'scene{idx}','meta_data.json'),'r') as f:
        data=json.load(f)

    # transform to world coordinate
    scale_mat = data['worldtogt']

    if args.gt_space:
        poses = load_poses(idx)
    else :
        # load pose and intrinsic for render depth
        poses=[]
        for frame in data['frames']:
            pose = np.array(frame["camtoworld"], dtype=np.float32)
            poses.append(pose)
    K = np.array(data['frames'][0]['intrinsics'], dtype=np.float32)[:3, :3]

    # intrinsic_path = os.path.join(f'../data/scannet/scan{idx}/intrinsic/intrinsic_color.txt')
    # K = np.loadtxt(intrinsic_path)[:3, :3]

    mesh = cull_mesh(mesh, poses, K,data['height'],data['width'])

    if not args.gt_space:
        mesh.apply_transform(scale_mat)

    # save mesh
    os.makedirs(os.path.join(args.out_dir,args.exp_name+f"_{idx}"),exist_ok=True)
    out_mesh_path = os.path.join(args.out_dir,args.exp_name+f"_{idx}", os.path.basename(args.mesh))
    mesh.export(out_mesh_path)

if __name__=='__main__':

    args=argparse.ArgumentParser()
    args.add_argument('--mesh', type=str, default='/data/projects/implicit_reconstruction/runs/scannet_normal_bg2-2_0616_00/2024-03-10_00-04-34/plots/bg2-2_mesh_1000.ply')
    args.add_argument('--data_dir',type=str, default='/data/scannet/scans')
    args.add_argument('--scan_id', type=str, default='0616_00')
    args.add_argument('--gt_space',action='store_true',help='输入mesh是否在gt space, 该文件保证输出为gt_space')
    args.add_argument('--exp_name', type=str, default='scannet_normal_bg2-2')
    args.add_argument('--out_dir', type=str, default='culled_mesh')
    args.set_defaults(gt_space=False)

    args=args.parse_args()

    main(args)
