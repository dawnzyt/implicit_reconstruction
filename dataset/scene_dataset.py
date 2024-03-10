import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random


fewshot_idx_replica_6_3views = [9, 13, 16]
fewshot_idx_replica_6_10views = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 scan_id,
                 split,
                 num_rays,
                 img_res=[384, 384],
                 center_crop_type='no_crop',
                 use_mask=False,
                 num_views=-1,
                 fewshot_idx=[],
                 save_training_img=True
                 ):

        self.instance_dir = os.path.join(data_dir, 'scan{0}'.format(scan_id))
        self.fewshot_idx = fewshot_idx
        self.total_pixels = img_res[0] * img_res[1]
        self.h = img_res[0]
        self.w = img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        self.split = split
        self.num_rays = num_rays
        self.has_mono_prior = True
        self.has_mask = use_mask
        assert num_views < 0 or (num_views >= 0 and num_views == len(
            fewshot_idx)), "not satisfying num_views<0 means all views, or num_views>=0 means fewshot "
        # assert num_views in [-1, 3, 6, 9]

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))

        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError

            self.intrinsics_all.append(intrinsics)
            self.pose_all.append(pose)

        self.rgb_images = []
        if save_training_img and os.path.exists(os.path.join(self.instance_dir, 'training_img')):
            rm_cmd = f"rm -rf {os.path.join(self.instance_dir, 'training_img')}"
            os.system(os.path.join(self.instance_dir, 'training_img'))
        for i, path in enumerate(image_paths):
            rgb = rend_util.load_rgb(path)
            if save_training_img and (num_views < 0 or (num_views >= 0 and i in self.fewshot_idx)):
                utils.mkdir_ifnotexists(os.path.join(self.instance_dir, 'training_img'))
                cv2.imwrite(self.instance_dir + '/training_img/' + os.path.basename(path),
                            cv2.cvtColor((rgb * 255).astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())

            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

    def __len__(self):
        if self.num_views < 0:
            return self.n_images
        return self.num_views

    def load_data(self, idx):
        rgb = self.rgb_images[idx]
        if self.has_mono_prior:
            depth = self.depth_images[idx]
            normal = self.normal_images[idx]
        else:
            depth = torch.zeros_like(rgb[:, :1])
            normal = torch.zeros_like(rgb)
        if self.has_mask:
            mask = torch.from_numpy(np.load(self.mask_paths[idx])).long().reshape(-1, 1)
        else:
            mask = torch.ones_like(depth)
        return rgb, depth, normal, mask

    def get_rays(self, K, pose, x=None, y=None):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        sk = K[0, 1]  # skew: 坐标轴倾斜参数, 理想情况下应该为0
        if x is None:
            x, y = np.meshgrid(np.arange(0, self.w, 1), np.arange(0, self.h, 1))
        y_cam = (y - cy) / fy * 1
        x_cam = (x - cx - sk * y_cam) / fx * 1  # 把cam->pixel的转换写出来即可得到这个式子
        z_cam = np.ones_like(x_cam)  # 假设z=1
        d = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (h,w,3)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        depth_scale = d.reshape(-1, 3)[:, 2:]  # 即z轴余弦角cosθ，用于计算depth
        d = (pose[:3, :3] @ (d.reshape(-1, 3).T)).T  # c2w
        o = np.tile(pose[:3, -1][None, :], (d.shape[0], 1))

        return torch.from_numpy(o).float(), torch.from_numpy(d).float(), torch.from_numpy(depth_scale).float()

    def __getitem__(self, idx):
        sample = {
            "idx": idx,
            "K": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        rgb, depth, normal, mask = self.load_data(idx)
        x, y = np.meshgrid(np.arange(0, self.w, 1), np.arange(0, self.h, 1))
        if self.split == 'valid':
            rays_o, rays_d, depth_scale = self.get_rays(sample["K"], sample["pose"], x, y)
            sample["rays_o"] = rays_o
            sample["rays_d"] = rays_d
            sample["depth_scale"] = depth_scale
            sample["rgb"] = rgb
            sample["normal"] = normal
            sample["depth"] = depth
            sample["mask"] = mask
        elif self.split == 'train':
            sampling_idx = torch.randperm(self.total_pixels)[:self.num_rays]
            rays_o, rays_d, depth_scale = self.get_rays(sample["K"], sample["pose"], x.ravel()[sampling_idx],
                                                        y.ravel()[sampling_idx])
            sample["rays_o"] = rays_o
            sample["rays_d"] = rays_d
            sample["depth_scale"] = depth_scale
            sample["rgb"] = rgb[sampling_idx, :]
            sample["normal"] = normal[sampling_idx, :]
            sample["depth"] = depth[sampling_idx, :]
            sample["mask"] = mask[sampling_idx, :]
            # sample["full_rgb"] = rgb
            # sample["full_depth"] = depth
            # sample["full_mask"] = mask

        return sample

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
