import os.path
import sys

import cv2
import numpy as np
import torch
import argparse

from tqdm import tqdm

from models.system import ImplicitReconSystem
from utils.mesh import my_extract_mesh, texture_function
from omegaconf import OmegaConf
from functools import partial
from dataset.base_dataset import BaseDataset
import utils.utils as utils

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='/data/projects/implicit_reconstruction/runs/scannet_normal_bg2_0616_00/2024-03-08_01-12-42/conf.yaml')
    parser.add_argument('--checkpoint', type=str, default='/data/projects/implicit_reconstruction/runs/scannet_normal_bg2_0616_00/2024-03-08_01-12-42/checkpoints/latest.pth')
    parser.add_argument('--output_dir', type=str, default='./rendering_results')
    args=parser.parse_args()

    # load model
    conf = OmegaConf.load(args.conf)
    bound = 1.0 if not hasattr(conf.model, 'bound') else conf.model.bound
    model = ImplicitReconSystem(conf, bound, device='cuda:0').cuda()
    ckpt=torch.load(args.checkpoint)
    cur_step = ckpt['step'] if ckpt.get('step') else 1e9
    if conf.model.object.sdf.enable_progressive:
        model.sdf.set_active_levels(cur_step)
        model.sdf.set_normal_epsilon()
    model.load_state_dict(ckpt['model'])
    model.eval()

    # rendering
    valid_dataset = BaseDataset(conf.dataset, split='valid',num_rays=conf.train.num_rays)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    h,w=valid_dataset.h,valid_dataset.w
    for i,sample in enumerate(valid_loader):
        sample = {k: v.cuda() for k, v in sample.items()} # to gpu
        split_sample = utils.split_input(sample, valid_dataset.total_pixels, n_pixels=1024)
        outputs = []
        for s in tqdm(split_sample, total=len(split_sample), desc=f'rendering batch {i}...', file=sys.stdout, position=0, leave=True):
            output = model(s)
            d = {'rgb': output['rgb'].detach(), 'depth': output['depth'].detach(), 'normal': output['normal'].detach()}
            outputs.append(d)
        outputs = utils.merge_output(outputs) # outputs: {'rgb': (batch_size, h*w, 3), 'depth': (batch_size, h*w, 1), 'normal': (batch_size, h*w, 3)}
        plot_outputs = utils.get_plot_data(outputs, sample, valid_dataset.h, valid_dataset.w)
        for plot_output in plot_outputs:
            idx=plot_output['idx']
            filename=valid_dataset.rgb_paths[idx].split('/')[-1]
            for k,v in plot_output.items():
                if k != 'idx':
                    os.makedirs(os.path.join(args.output_dir, k), exist_ok=True)
                    v.save(os.path.join(args.output_dir, k, filename))
            print(f'{filename} rendered and saved to {args.output_dir}', file=sys.stdout)

        # # 可视化normal_mask rgb
        # rgb = sample['rgb'][0].cpu().numpy().reshape(h, w, 3)
        # mask = sample['mask'][0].cpu().numpy().reshape(h, w)
        # uncertainty = sample['uncertainty'][0].cpu().numpy().reshape(h, w)
        # uncertainty_grad = sample['uncertainty_grad'][0].cpu().numpy().reshape(h, w)
        # normal_mask = ((mask == 1) | ((mask == 0) & (uncertainty < 0.3) & (uncertainty_grad < 0.03)))
        # rgb[~normal_mask] = 0
        # rgb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # os.makedirs(os.path.join(args.output_dir, 'normal_mask'), exist_ok=True)
        # cv2.imwrite(os.path.join(args.output_dir,'normal_mask', f'{i}.png'), rgb)


