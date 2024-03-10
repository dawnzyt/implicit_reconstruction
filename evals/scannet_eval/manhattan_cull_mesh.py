# # adapted from https://github.com/zju3dv/manhattan_sdf
# import argparse
#
# import numpy as np
# import open3d as o3d
# from sklearn.neighbors import KDTree
# import trimesh
# import torch
# import glob
# import os
# import pyrender
# import os
# from tqdm import tqdm
# from pathlib import Path
#
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
#
# # hard-coded image size
# H, W = 968, 1296
# scenes = ["scene0050_00", "scene0084_00", "scene0580_00", "scene0616_00"]
#
# # load pose
# def load_poses(scan_id):
#     pose_path = os.path.join(f'../data/scannet/scan{scan_id}', 'pose')
#     poses = []
#     pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
#                         key=lambda x: int(os.path.basename(x)[:-4]))
#     for pose_path in pose_paths[::10]:
#         c2w = np.loadtxt(pose_path)
#         if np.isfinite(c2w).any():
#             poses.append(c2w)
#     poses = np.array(poses)
#
#     return poses
#
#
# class Renderer():
#     def __init__(self, height=480, width=640):
#         self.renderer = pyrender.OffscreenRenderer(width, height)
#         self.scene = pyrender.Scene()
#         # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES
#
#     def __call__(self, height, width, intrinsics, pose, mesh):
#         self.renderer.viewport_height = height
#         self.renderer.viewport_width = width
#         self.scene.clear()
#         self.scene.add(mesh)
#         cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
#                                         fx=intrinsics[0, 0], fy=intrinsics[1, 1])
#         self.scene.add(cam, pose=self.fix_pose(pose))
#         return self.renderer.render(self.scene)  # , self.render_flags)
#
#     def fix_pose(self, pose):
#         # 3D Rotation about the x-axis.
#         t = np.pi
#         c = np.cos(t)
#         s = np.sin(t)
#         R = np.array([[1, 0, 0],
#                       [0, c, -s],
#                       [0, s, c]])
#         axis_transform = np.eye(4)
#         axis_transform[:3, :3] = R
#         return pose @ axis_transform
#
#     def mesh_opengl(self, mesh):
#         return pyrender.Mesh.from_trimesh(mesh)
#
#     def delete(self):
#         self.renderer.delete()
#
#
# # refuse func: 将mesh render到每个pose上，然后用TSDF算法进行refuse，得到视角过滤后的mesh
# # 类似cull mesh过程
# def refuse(mesh, poses, K):
#     renderer = Renderer()
#     mesh_opengl = renderer.mesh_opengl(mesh)
#     volume = o3d.pipelines.integration.ScalableTSDFVolume(
#         voxel_length=0.01,
#         sdf_trunc=3 * 0.01,
#         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
#     )
#
#     for pose in tqdm(poses):
#         intrinsic = np.eye(4)
#         intrinsic[:3, :3] = K
#
#         rgb = np.ones((H, W, 3))
#         rgb = (rgb * 255).astype(np.uint8)
#         rgb = o3d.geometry.Image(rgb)
#         _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
#         depth_pred = o3d.geometry.Image(depth_pred)
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
#         )
#         fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)
#         extrinsic = np.linalg.inv(pose)
#         volume.integrate(rgbd, intrinsic, extrinsic)
#
#     return volume.extract_triangle_mesh()
#
# def main(args):
#     # load mesh
#     mesh = trimesh.load(args.mesh)
#
#     idx=args.scan_id
#     # transform to world coordinate
#     cam_file = os.path.join(args.data_dir,f"scan{idx}/cameras.npz")
#     scale_mat = np.load(cam_file)['scale_mat_0']
#     mesh.vertices = (scale_mat[:3, :3] @ mesh.vertices.T + scale_mat[:3, 3:]).T
#
#     # load pose and intrinsic for render depth
#     poses = load_poses(idx)
#
#     intrinsic_path = os.path.join(args.data_dir,f'scan{idx}/intrinsic/intrinsic_color.txt')
#     K = np.loadtxt(intrinsic_path)[:3, :3]
#
#     mesh = refuse(mesh, poses, K)
#
#     # save mesh
#     out_mesh_path = os.path.join(args.out_dir,args.exp_name+f"_{idx}",f"{scenes[idx-1]}.ply")
#     o3d.io.write_triangle_mesh(out_mesh_path, mesh)
#
# if __name__=='__main__':
#
#     args=argparse.ArgumentParser()
#     args.add_argument('--mesh', type=str, default='/data/projects/implicit_reconstruction/runs/scannet_monosdf_4/2024-03-06_01-30-51/plots/bg2-2_mesh_1000.ply')
#     args.add_argument('--data_dir',type=str, default='/data/monosdf/scannet')
#     args.add_argument('--scan_id', type=int, default=4)
#     args.add_argument('--exp_name', type=str, default='scannet_grids')
#     args.add_argument('--out_dir', type=str, default='culled_mesh')
#
#     args=args.parse_args()
#
#     main(args)
