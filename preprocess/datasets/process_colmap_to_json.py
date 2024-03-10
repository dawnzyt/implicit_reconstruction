import json

from preprocess.colmap.colmap_utils import *
import cv2,os
import numpy as np
import trimesh
import argparse
import tqdm
from PIL import Image
from torchvision import transforms

def get_scale_mat(pcd, poses, bound=1.0, method='points', if_align=False):
    """

    Args:
        pcd: sparse 3d points的trimesh.PointCloud
        poses: 所有相机位姿c2ws
        bound: normalized sphere的半径或cube：[-bound, bound]^3
        method: 'points' or 'poses'
        if_align: 是否将bounding box对齐到坐标轴，用trimesh.bounds.oriented_bounds实现
    Returns:
        scale_mat: new-w->gt, 物体中心坐标系到世界坐标系的变换，带缩放的非欧式变换。
        normalized_pcd: 去除离群点后的物体中心系点云
        poses: c->new-w
    """
    # 去除离群点
    # TODO： 改进离群点去除方法，这里仅考虑了距离异常点。
    verts = pcd.vertices
    colors = pcd.colors
    bbox_center = np.sum(verts, axis=0) / verts.shape[0]
    dist = np.linalg.norm(verts - bbox_center, axis=1)
    threshold = np.percentile(dist, 99.5)  # 距离过滤阈值
    mask = dist < threshold
    to_align = np.eye(4) # to_align: 定义的世界坐标的对齐变换（场景平移到中心且与坐标轴对齐），∈SE(3)，即是欧式变换没有缩放。默认to_align=I。
    # 获得to_align，以及align后的世界场景scale和shift
    if method == 'points': # 根据3d points
        if if_align:
            # FIXME: 有问题，缩放变换，不∈欧式变换空间，不能用c2n=scale_mat^-1@pose代表 变换后pose，SE(3)空间要求R是正交阵，这样得到的c2n，T分量就不是相机坐标[本质也在于：这样做这个n也不是世界坐标]。
            #        因此，需直接改变世界坐标，即对world进行对齐、缩放平移，即只需要作用于原c2w的T即可。[√] 已解决，
            #      总结：对齐、缩放平移后pose必须保持是欧式变换，因此对pose进行align，以及align后仅对T进行缩放平移。scale
            filtered_pcd = trimesh.PointCloud(vertices=verts[mask], colors=colors[mask])
            # TODO： 改进to_align的计算方法，这里仅考虑了bounding box对齐到坐标轴。
            to_align, extents = trimesh.bounds.oriented_bounds(filtered_pcd) # to_align: 世界坐标的对齐变换，∈SE(3)，即是欧式变换没有缩放。
            radius = np.linalg.norm(extents)/2 * 1.2
            shift = np.zeros(3)
        else: # 不对齐：即简单的平移和缩放
            bbox_center = np.sum(verts[mask], axis=0) / verts[mask].shape[0]
            radius = np.percentile(np.linalg.norm(verts[mask] - bbox_center[None], axis=1), 99) * 1.2
            shift = bbox_center
    elif method == 'poses': # 根据相机位姿
        if if_align:
            centers_pcd = trimesh.PointCloud(vertices=poses[:, :3, -1])
            to_align, extents = trimesh.bounds.oriented_bounds(centers_pcd)
            radius = np.linalg.norm(extents) / 2 * 1.5 # 1.5是为了保证包围盒比点云大
            shift = np.zeros(3)
        else:
            centers = poses[:, :3, -1]
            bbox_min = np.min(centers, axis=0)
            bbox_max = np.max(centers, axis=0)
            bbox_center = (bbox_min + bbox_max) / 2
            radius = np.linalg.norm(bbox_max - bbox_center) * 1.5
            shift = bbox_center
    # 将c2w转换为c2alignedw   = w->alignedw @ c2w
    poses = np.matmul(to_align[None].repeat(poses.shape[0], axis=0), poses)
    tmp = (poses[:,:3,:3]**2).reshape(-1,9).sum(1)
    # 对align后的相机center进行缩放平移，这里变换得到的就是json中的camtoworld了。
    poses[:, :3, -1] -= shift
    poses[:, :3, -1] = poses[:, :3, -1]/radius*bound # 新的c2w, world坐标∈[-bound, bound]^3
    # 接下来构建scale_mat，即对齐缩放平移后物体中心坐标系到GT原世界坐标系的变换。 ！！！注意：不∈SE(3)！！！
    # 很简单，缩放回去再做对齐逆变换即可
    scale_mat = np.eye(4)
    scale_mat[range(3), range(3)] = radius/bound
    scale_mat[:3, -1] = shift
    scale_mat = np.linalg.inv(to_align) @ scale_mat
    # 将离群点即~mask的坐标和颜色设置为0
    normalized_pcd = pcd.copy()
    normalized_pcd.apply_transform(np.linalg.inv(scale_mat))
    normalized_pcd.vertices[~mask] = 0
    normalized_pcd.colors[~mask] = 0
    print(f'max normalized_pcd dist: {np.max(np.linalg.norm(normalized_pcd.vertices, axis=1))}')

    return scale_mat, normalized_pcd, poses




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=False,default='/data/gaussian/myhome', help='colmap sparse reconstruction folder, containing cameras.bin, images.bin, points3D.bin')
    parser.add_argument('-o', '--output_dir', type=str, default='',  help='output folder')
    parser.add_argument( '--resize_h', type=int, default=480, help='resize height')
    parser.add_argument( '--resize_w', type=int, default=640, help='resize width')
    parser.add_argument('--radius', type=float, default=1.0, help='radius of the scene, or scene box bound of the scene')
    parser.add_argument("--has_mono_depth", action='store_true', help="monocular depth prior ")
    parser.add_argument("--has_mono_normal", action='store_true', help="monocular normal prior")
    parser.add_argument("--has_mask", action='store_true', help="2d mask")
    parser.add_argument("--has_uncertainty", action='store_true', help="2d uncertainty")
    parser.set_defaults(has_mono_depth=True, has_mono_normal=True, has_mask=True, has_uncertainty=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir else args.input_dir

    # 1. read cameras, images, _
    cameras, images, _ = read_model(os.path.join(args.input_dir, 'sparse/0'), ext=".bin")
    # 获取内参
    fx, fy, cx, cy = cameras[1].params
    intrinsic = np.eye(4)
    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = fx, fy, cx, cy

    # crop and resize使得resize前长宽比一致
    H, W = cameras[1].height, cameras[1].width
    h, w= args.resize_h, args.resize_w # resize to h,w
    min_ratio = int(min(H / h, W / w))
    crop_size = (h * min_ratio, w * min_ratio)
    trans = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.Resize((h, w), interpolation=Image.LANCZOS),
        ]
    )
    # 调整resize后的内参
    offset_x = (W - w * min_ratio) * 0.5
    offset_y = (H - h * min_ratio) * 0.5
    intrinsic[0, 2] -= offset_x
    intrinsic[1, 2] -= offset_y
    intrinsic[:2, :] /= min_ratio

    # 2. get poses
    poses = []  # c2w
    intrinsics = []
    filenames = []
    for i, image in enumerate(images.values()):
        # get pose
        qvec=image.qvec
        tvec=image.tvec
        R = qvec2rotmat(qvec)
        extrinsic=np.concatenate([R, tvec.reshape(3, 1)], 1)
        c2w = np.linalg.inv(np.concatenate([extrinsic, np.array([0, 0, 0, 1])[None]], 0))
        poses.append(c2w)

        intrinsics.append(intrinsic)
        filenames.append(image.name)
    poses = np.array(poses)

    # 3. get scale_mat
    ply_file = os.path.join(args.input_dir,'sparse','0', 'points3D.ply') # sparse点云
    pcd = trimesh.load(ply_file)
    scale_mat, normalized_pcd, poses = get_scale_mat(pcd, poses, bound=args.radius, method='points', if_align=True)

    normalized_pcd.export(os.path.join(args.input_dir, 'sparse/0', 'points3D_normalized.ply'))

    # TODO: 1. 读入图片然后trans进行crop和resize,; 2. 保存json文件
    scene_box = { # near、intersection_type由confs配置文件确定, far由运行时具体光线与cube/sphere的交点确定
        "aabb": [[-args.radius, -args.radius, -args.radius], [args.radius, args.radius, args.radius]],
        "near": 0.0,
        "far": 2.5,
        "radius": args.radius,
        "collider_type": "box",
    }
    data= {
        "camera_model": "OPENCV",
        "height": h,
        "width": w,
        "has_mono_depth": args.has_mono_depth,
        "has_mono_normal": args.has_mono_normal,
        "has_mask": args.has_mask,
        "has_uncertainty": args.has_uncertainty,
        "worldtogt": scale_mat.tolist(),
    }
    # frames
    frames = []
    out_index = 0
    # 创建rgb、mono_depth、mono_normal、mask、uncertainty等文件夹
    rgb_path = os.path.join(args.output_dir, "rgb")
    os.makedirs(rgb_path, exist_ok=True)
    mono_depth_path = os.path.join(args.output_dir, "mono_depth")
    os.makedirs(mono_depth_path, exist_ok=True)
    mono_normal_path = os.path.join(args.output_dir, "mono_normal")
    os.makedirs(mono_normal_path, exist_ok=True)
    mask_path = os.path.join(args.output_dir, "mask")
    os.makedirs(mask_path, exist_ok=True)
    uncertainty_path = os.path.join(args.output_dir, "uncertainty")
    os.makedirs(uncertainty_path, exist_ok=True)

    frames = []
    for i, (pose,intrinsic,filename) in tqdm.tqdm(enumerate(zip(poses, intrinsics, filenames))):
        img = Image.open(os.path.join(args.input_dir,'images',filename))
        img = trans(img)
        img.save(os.path.join(rgb_path, filename))

        frame = {
            "rgb_path": os.path.join("rgb", filename),
            "camtoworld": pose.tolist(),
            "intrinsics": intrinsic.tolist(),
            "mono_depth_path": os.path.join("mono_depth", 'res', filename[:-4] + ".npy"),
            "mono_normal_path": os.path.join("mono_normal", 'res', filename[:-4] + ".npy"),
            # "sensor_depth_path": rgb_path.replace("_rgb.png", "_sensor_depth.npy"),
            "mask_path": os.path.join("mask", 'res', filename[:-4] + ".npy"),
            "uncertainty_path": os.path.join("uncertainty", 'res', filename[:-4] + ".npy"),
        }

        frames.append(frame)
    data["frames"] = frames

    # 保存meta_data.json
    with open(os.path.join(args.output_dir, 'meta_data.json'), 'w') as f:
        json.dump(data, f, indent=4)