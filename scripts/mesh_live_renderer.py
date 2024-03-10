import pyrender
import trimesh
import numpy as np

mesh_path = '/runs/scannet_normal_bg1_0616_00/scannet_normal_bg1_0616_00_512_latest.ply'

tm = trimesh.load(mesh_path)
# 创建一个红色的材质，RGBA值为[1, 0, 0, 1]
material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.6, 0.6, 0.6, 1])
m = pyrender.Mesh.from_trimesh(tm, material=material)

# 创建一个场景并添加网格
scene = pyrender.Scene()
scene.add(m)

# 创建一个透视相机
pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
# 创建一个正交相机
oc = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
# 创建一个相机节点
camera_node = pyrender.Node(camera=oc)
# 把相机节点添加到场景中
scene.add_node(camera_node)
# 创建一个变换矩阵，用于设置相机的位置和方向
# 这里我们把相机移动到(0, 0, 1)的位置，你可以根据你的需要修改这些值
camera_pose = np.eye(4)
camera_pose[:3, 3] = [0, 0, 3]
# 设置相机的位置和方向
scene.set_pose(camera_node, pose=camera_pose)

# pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1000.0)
light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=10)
# 创建一个变换矩阵，用于移动光源的位置
# 这里我们把光源移动到(0.5, 0.5, 0.5)的位置，你可以根据你的需要修改这些值
light_pose = np.eye(4)
light_pose[:3, 3] = [0, 0, 0]
scene.add(light, pose=light_pose)

# 用pyrender.Viewer()函数来可视化网格
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)