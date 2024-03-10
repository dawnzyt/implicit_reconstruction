# adapted from https://github.com/zju3dv/manhattan_sdf
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

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics

# hard-coded image size
H, W = 968, 1296

# load pose
def load_poses(scan_id):
    pose_path = os.path.join(f'../data/scannet/scan{scan_id}', 'pose')
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

# TODO:目前是对实验中途跑出的mesh_epoch.ply进行evaluation。
data_dir = '/data/scannet/scans'
root_dir = "../../runs"
exp_name = "scannet_normal_bg2-2"
out_dir = os.path.join('evaluation', exp_name)
Path(out_dir).mkdir(parents=True, exist_ok=True)
scenes = ["0616_00"]
all_results = []
for id in scenes:
    cur_exp = f"{exp_name}_{id}"
    cur_root = os.path.join(root_dir, cur_exp)
    # use first timestamps
    dirs = sorted(os.listdir(cur_root))
    cur_root = os.path.join(cur_root, dirs[0])
    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "mesh", "*.ply"))))
    
    # evalute the latest mesh
    files.sort(key=lambda x:os.path.getmtime(x))
    ply_file = files[-1]
    print("Evaluating ", ply_file)
    
    mesh = trimesh.load(ply_file)

    # 加载json文件
    with open(os.path.join(data_dir,f'scene{id}','meta_data.json'),'r') as f:
        data=json.load(f)
    h, w = data["height"], data["width"]
    scale_mat = data['worldtogt']
    # load pose
    poses = []
    for frame in data['frames']:
        pose = np.array(frame["camtoworld"], dtype=np.float32)
        poses.append(pose)
    K = np.array(data['frames'][0]['intrinsics'], dtype=np.float32)[:3, :3]

    # intrinsic_path = os.path.join(f'/data/scannet/scans/scene{id}/intrinsic/intrinsic_color.txt')
    # K = np.loadtxt(intrinsic_path)[:3, :3]

    mesh = cull_mesh(mesh, poses, K,h,w)
    mesh.apply_transform(scale_mat)

    
    
    #gt_mesh = os.path.join("../data/scannet/GTmesh", f"{scan}_vh_clean_2.ply")
    gt_mesh = trimesh.load(f"/data/scannet/scans/scene{id}/scene{id}_vh_clean_2.ply")
    mesh.export('eval_mesh.ply')
    gt_mesh.export('gt_mesh.ply')
    metrics = evaluate(mesh, gt_mesh)
    print(metrics)
    all_results.append(metrics)
    
# print all results
for scan, metric in zip(scenes, all_results):
    values = [scan] + [str(metric[k]) for k in metric.keys()]
    out = ",".join(values)
    print(out)
