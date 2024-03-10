import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import numpy as np

# import utils.general as utils
from glob import glob
import cv2
import random
import json
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, conf, split='train', num_rays=1024) -> None:
        super().__init__()
        self.data_dir = conf.data_dir if conf.scan_id == -1 else os.path.join(conf.data_dir, f'scan{conf.scan_id}')
        # self.data_dir = conf.data_dir if conf.scan_id == -1 else os.path.join(conf.data_dir, f'scene{conf.scan_id}')
        self.split=split
        self.num_rays=num_rays

        data_json = os.path.join(self.data_dir, 'meta_data.json')
        with open(data_json, 'r') as f:
            data = json.load(f)

        self.h = data["height"]
        self.w = data["width"]
        self.total_pixels = self.h * self.w
        self.img_res = [self.h, self.w]
        self.bound = data["scene_box"]["radius"]

        # self.has_mono_depth = data["has_mono_depth"] if not hasattr(conf, "use_mono_depth") else conf.use_mono_depth
        # self.has_mono_normal = data["has_mono_normal"] if not hasattr(conf, "use_mono_normal") else conf.use_mono_normal
        self.has_mono_depth = True
        self.has_mono_normal = True
        # TODO: 添加2d mask, guide osf
        self.has_mask = True if not hasattr(conf, "use_mask") else conf.use_mask
        # TODO: 添加2d uncertainty
        self.has_uncertainty = True if not hasattr(conf, "use_uncertainty") else conf.use_uncertainty

        self.importance_sampling = conf.importance_sampling
        self.scale_mat = data["worldtogt"]

        frames = data["frames"]
        self.n_images = len(frames)
        self.rgb_paths = []
        self.mono_depth_paths = []
        self.mono_normal_paths = []
        self.mask_paths = []
        self.uncertainty_paths = []
        self.poses = []
        self.intrinsics = []

        for frame in frames:
            rgb_path = os.path.join(self.data_dir, frame["rgb_path"])
            pose = np.array(frame["camtoworld"],dtype=np.float32)
            # pose[:3, -1]=pose[:3, -1]/1.1
            intrinsic = np.array(frame["intrinsics"],dtype=np.float32)

            self.rgb_paths.append(rgb_path)
            self.poses.append(pose)
            self.intrinsics.append(intrinsic)

            if self.has_mono_depth:
                self.mono_depth_paths.append(os.path.join(self.data_dir, frame["mono_depth_path"]))
            if self.has_mono_normal:
                self.mono_normal_paths.append(os.path.join(self.data_dir, frame["mono_normal_path"]))
            if self.has_mask:
                self.mask_paths.append(os.path.join(self.data_dir, frame["mask_path"]))
            if self.has_uncertainty:
                self.uncertainty_paths.append(os.path.join(self.data_dir, frame["uncertainty_path"]))
        print(f"Loaded {self.data_dir} with {self.n_images} images successfully, split: {split}")

    def __len__(self):
        return self.n_images

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

    def get_scale_mat(self):
        return self.scale_mat

    def sobel(self,x):
        gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
        gm = np.sqrt(gx ** 2 + gy ** 2)
        return gm

    def load_data(self, idx):
        rgb = np.asarray(Image.open(self.rgb_paths[idx])).astype(np.float32) / 255
        rgb = torch.from_numpy(rgb.reshape(-1, 3)).float()
        if self.has_mono_depth:
            depth = torch.from_numpy(np.load(self.mono_depth_paths[idx])).float().reshape(-1, 1)
        else:
            depth = torch.zeros_like(rgb[:, :1])
        if self.has_mono_normal:
            normal = torch.from_numpy(np.load(self.mono_normal_paths[idx]))
            normal = normal.reshape(-1, 3).float()
            # normal = normal.reshape(3, -1).float() # monosdf是(3,h*w)
            normal = normal * 2 - 1
            # normal = normal.T
        else:
            normal = torch.zeros_like(rgb)
        if self.has_mask:
            mask = torch.from_numpy(np.load(self.mask_paths[idx])).long().reshape(-1, 1)
        else:
            mask = torch.ones_like(depth)
        if self.has_uncertainty:
            uncertainty = np.load(self.uncertainty_paths[idx])
            uncertainty_grad = self.sobel(uncertainty)
            uncertainty, uncertainty_grad = torch.from_numpy(uncertainty).float().reshape(-1, 1), torch.from_numpy(uncertainty_grad).float().reshape(-1, 1)
        else:
            uncertainty = torch.zeros_like(depth)
            uncertainty_grad = torch.zeros_like(depth)
        return rgb, depth, normal, mask, uncertainty, uncertainty_grad

    def get_rays(self, K, pose, x=None, y=None):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        sk = K[0, 1] # skew: 坐标轴倾斜参数, 理想情况下应该为0
        if x is None:
            x, y = np.meshgrid(np.arange(0, self.w, 1), np.arange(0, self.h, 1))
        y_cam = (y - cy) / fy * 1
        x_cam = (x - cx -sk*y_cam) / fx * 1 # 把cam->pixel的转换写出来即可得到这个式子
        z_cam = np.ones_like(x_cam)  # 假设z=1
        d = np.stack([x_cam, y_cam, z_cam], axis=-1) # (h,w,3)
        d = d/np.linalg.norm(d,axis=-1,keepdims=True)
        depth_scale = d.reshape(-1,3)[:, 2:] # 即z轴余弦角cosθ，用于计算depth
        d = (pose[:3,:3]@(d.reshape(-1,3).T)).T # c2w
        o=np.tile(pose[:3,-1][None,:],(d.shape[0],1))

        return torch.from_numpy(o).float(), torch.from_numpy(d).float(), torch.from_numpy(depth_scale).float()

    def __getitem__(self, idx):
        sample = {
            "idx": idx,
            "K": self.intrinsics[idx],
            "pose": self.poses[idx]
        }
        rgb, depth, normal, mask, uncertainty, uncertainty_grad= self.load_data(idx)
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
            sample["uncertainty"] = uncertainty
            sample["uncertainty_grad"] = uncertainty_grad
        elif self.split == 'train':
            # TODO: 基于object mask、uncertainty等进行重要性采样
            if self.importance_sampling:
                prob_map=torch.empty_like(depth)
                prob_map[mask==1]=3.3
                prob_map[mask==0]=6.6
                prob_map[(mask==0)&(uncertainty>0.3)]*=1.2 # [MONO-5]
                # uncertainty_threshold=0.35
                # prob_map[uncertainty>0.35]*=1.4
                # prob_map[uncertainty<=0.3]/=1.4
                prob_map=prob_map.ravel()/prob_map.sum()
                sampling_idx = torch.multinomial(prob_map, self.num_rays, replacement=False)
            else:
                sampling_idx = torch.randperm(self.total_pixels)[:self.num_rays] # TODO: 根据uncertainty/mask进行importance sampling
            rays_o, rays_d, depth_scale = self.get_rays(sample["K"], sample["pose"], x.ravel()[sampling_idx], y.ravel()[sampling_idx])
            sample["rays_o"] = rays_o
            sample["rays_d"] = rays_d
            sample["depth_scale"] = depth_scale
            sample["rgb"] = rgb[sampling_idx, :]
            sample["normal"] = normal[sampling_idx, :]
            sample["depth"] = depth[sampling_idx, :]
            sample["mask"] = mask[sampling_idx, :]
            sample["uncertainty"] = uncertainty[sampling_idx, :]
            sample["uncertainty_grad"] = uncertainty_grad[sampling_idx, :]
            # sample["full_rgb"] = rgb
            # sample["full_depth"] = depth
            # sample["full_mask"] = mask

        return sample
