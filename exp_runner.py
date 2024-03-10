import time
from glob import glob

import cv2
import torch
from PIL import Image
import argparse
import sys, os, datetime
from tqdm import tqdm

import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from models.system import ImplicitReconSystem
from models.loss import ImplicitReconLoss, get_psnr, compute_scale_and_shift
from dataset.base_dataset import BaseDataset
from dataset.scene_dataset import SceneDatasetDN
from utils.mesh import my_extract_mesh
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *

def init_processes():
    # 获取rank
    gpu = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])

    # local_rank用于初始化torch device
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, )
    print('device {}/{} started...'.format(rank, world_size))
    dist.barrier()
    return gpu


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='/data/projects/implicit_reconstruction/confs/scannet.yaml', help='conf')
    parser.add_argument('--epoches', type=int, default=1000)
    parser.add_argument('--root_dir', type=str, default='runs', help='实验根目录')
    parser.add_argument('--is_continue', action='store_true', help='continue')
    parser.add_argument('--checkpoint', default='latest', type=str, help='checkpoint')
    parser.set_defaults(is_continue=False)
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DDP')
    opt = parser.parse_args()
    return opt


class Trainer():
    def __init__(self, opt, gpu):
        self.conf = OmegaConf.load(opt.conf)
        self.root_dir = opt.root_dir
        self.exp_name = self.conf.train.exp_name if self.conf.dataset.scan_id == -1 else self.conf.train.exp_name + f'_{self.conf.dataset.scan_id}'
        self.epoches = opt.epoches
        self.last_epoch = 0
        self.cur_step = 0
        self.is_continue = opt.is_continue
        self.gpu = gpu
        self.chunk=self.conf.train.chunk

        # 实验相关目录
        self.exp_dir = os.path.join(self.root_dir, self.exp_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.exp_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.plot_dir = os.path.join(self.exp_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        # save conf
        with open(os.path.join(self.exp_dir, 'conf.yaml'), 'w') as f:
            OmegaConf.save(self.conf, f)

        # dataset
        self.train_dataset = BaseDataset(self.conf.dataset, split='train', num_rays=self.conf.train.num_rays)
        self.total_pixels, self.h, self.w = self.train_dataset.total_pixels, self.train_dataset.h, self.train_dataset.w
        self.img_res = [self.w, self.h]
        self.valid_dataset = BaseDataset(self.conf.dataset, split='valid', num_rays=self.conf.train.num_rays)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
        self.dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.conf.train.batch_size,
                                                      num_workers=20, sampler=self.train_sampler)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=1, shuffle=True)

        self.bound = self.train_dataset.bound # scene bound
        # model
        self.model = ImplicitReconSystem(self.conf, bound=self.bound, device=gpu).to(gpu)

        # loss
        self.loss = ImplicitReconLoss(**self.conf.loss, warm_up_end=self.conf.optim.sched.warm_up_end)

        # optimizer
        sdf_conf = self.conf.model.object.sdf
        rgb_conf = self.conf.model.object.rgb
        osf_conf = self.conf.model.object.osf
        bg_conf = self.conf.model.background
        optim_conf = self.conf.optim
        if optim_conf.type == 'AdamW':
            optim = torch.optim.AdamW
        elif optim_conf.type == 'Adam':
            optim = torch.optim.Adam
        params = []
        if sdf_conf.enable_hashgrid:  # multi-res hash encoder, 貌似需要高学习率
            params += [{'name': 'hash-encoder', 'params': self.model.sdf.get_grid_params(),'lr': optim_conf.lr * optim_conf.lr_scale_grid}]
        params += [{'name': 'sdf-mlp', 'params': self.model.sdf.get_mlp_params(), 'lr': optim_conf.lr},
                   {'name': 'radiance', 'params': self.model.rgb.parameters(), 'lr': optim_conf.lr}] # sdf mlp, radiance
        if bg_conf.enabled:  # bg nerf
            params += [{'name': 'background-nerf', 'params': self.model.bg_nerf.parameters(), 'lr': optim_conf.lr}]
        if osf_conf.enabled:  # osf
            params += [{'name': 'osf', 'params': self.model.osf.parameters(), 'lr': optim_conf.lr}]
        params += [{'name': 'density', 'params': self.model.density.parameters(), 'lr':optim_conf.lr}] # density beta
        if rgb_conf.enable_app:  # appearance object/scene
            params += [{'name': 'appearance', 'params': self.model.app.parameters(), 'lr': optim_conf.lr}]
        if bg_conf.nerf.enable_app and bg_conf.enabled:  # appearance bg
            params += [{'name': 'appearance-bg', 'params': self.model.app_bg.parameters(), 'lr': optim_conf.lr}]
        self.optimizer = optim(params=params, betas=(0.9, 0.999), eps=1e-15)

        # TODO: scheduler
        # 1. LambdaLR自定义two_steps; 2. exponential
        self.scheduler = self.get_scheduler(self.optimizer,self.conf.optim.sched)

        # load checkpoint
        if self.is_continue:
            self.load_checkpoint(opt.checkpoint)

        # init DDP
        self.model = DDP(self.model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

        # tensorboard
        self.loger = SummaryWriter(self.log_dir)

    def get_scheduler(self, optimizer, sched_conf):
        if self.conf.optim.sched.type == 'two_steps_lr':
            def lr_lambda(step): # 即lr:[warm, 1, gamma, gamma^2]对应的step:[0, warm_up_end, two_steps[0], two_steps[1], end]
                if step < self.conf.optim.sched.warm_up_end:
                    return step / self.conf.optim.sched.warm_up_end
                elif step < sched_conf.two_steps[0]:
                    return 1.
                elif step < sched_conf.two_steps[1]:
                    return sched_conf.gamma
                else:
                    return sched_conf.gamma ** 2

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.conf.optim.sched.type == 'exponential_lr':
            total_steps = self.epoches * len(self.dataloader)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_conf.gamma**(1/total_steps))
        else:
            raise ValueError('Unknown scheduler type')
        return scheduler

    def load_checkpoint(self, checkpoint):
        if checkpoint == 'latest':
            timestamp_dir = os.path.join(self.root_dir, self.exp_name)
            # 找到最新的timestamp dir
            timestamps = glob(os.path.join(timestamp_dir, '*'))
            timestamps.sort(key=os.path.getmtime)
            checkpoint = os.path.join(timestamps[-2], 'checkpoints', 'latest.pth') # -1是刚创建的
        ckpt = torch.load(checkpoint, map_location='cuda:{}'.format(self.gpu))
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.last_epoch = ckpt['epoch']
        if self.gpu == 0:
            print(f'Continue training! Last epoch: {self.last_epoch}')
            print('Loaded checkpoint from {}'.format(dist.get_rank(), dist.get_world_size(),self.last_epoch, checkpoint))
            print(self.model)
            print(self.optimizer)

    def save_checkpoint(self, epoch, save_epoch=False):
        ckpt = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'step': self.cur_step,
            'scale_mat': self.train_dataset.scale_mat,
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, 'latest.pth'))
        if save_epoch:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))

    def plot(self, epoch, if_rendering=True):
        print('plotting...')
        self.model.eval()
        # 1. plot rgb、depth、normal
        if if_rendering:
            sample = next(iter(self.valid_dataloader))
            sample = {k: v.to(self.gpu) for k, v in sample.items()}
            split_sample = split_input(sample,self.total_pixels, self.chunk)
            outputs = []
            for s in tqdm(split_sample, total=len(split_sample), desc=f'rendering valid...', file=sys.stdout):
                output = self.model(s)
                d= {'rgb': output['rgb'].detach(), 'depth': output['depth'].detach(), 'normal': output['normal'].detach()}
                outputs.append(d)
            outputs = merge_output(outputs) # plot rgb、depth、normal
            plot_outputs = get_plot_data(outputs, sample, self.h,self.w)
            for i, plot_output in enumerate(plot_outputs):
                idx = plot_output['idx']
                for k, v in plot_output.items(): # v是PIL.Image,k∈['idx', 'rgb', 'depth', 'normal', 'merge']
                    if k!='idx':
                        v.save(os.path.join(self.plot_dir, f'{k}_epoch{epoch}_view{idx}.png'))

        # 2. extract mesh
        mesh=my_extract_mesh(sdf_func=self.model.module.sdf.get_sdf, bounds=np.array([[-self.bound,self.bound],[-self.bound,self.bound],[-self.bound,self.bound]]),
                             res=self.conf.train.mesh_resolution, block_res = self.conf.train.block_resolution)
        mesh.export(os.path.join(self.plot_dir, f'mesh_{epoch}.ply'), 'ply')
        self.model.train()

    def train(self):
        self.cur_step = self.last_epoch * len(self.dataloader)
        if self.gpu == 0:
            pass
            # self.plot(self.last_epoch) # plot before training
            # exit(0)
        for epoch in range(self.last_epoch + 1, self.epoches + 1):
            self.train_sampler.set_epoch(epoch)
            for i, sample in tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout, desc=f"Epoch{epoch}"):
                self.model.module.sdf.set_active_levels(self.cur_step)
                self.model.module.sdf.set_normal_epsilon()
                self.loss.set_curvature_weight(self.cur_step, self.model.module.sdf.anneal_levels)
                sample = {k: v.to(self.gpu) for k, v in sample.items()} # to gpu
                output = self.model(sample)
                losses = self.loss(output, sample)
                loss = losses['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.cur_step += 1


                # print and log
                psnr = get_psnr(output['rgb'], sample['rgb'], mask=~output['outside'])
                print(f'[loss]: total:{loss.item():.4f}, eik:{losses["loss_eik"].item():.4f}, rgb:{losses["loss_rgb"].item():.4f}, smooth:{losses["loss_smooth"].item():.4f}, '
                      f'normal_l1:{losses["loss_normal_l1"].item():.4f}, normal_cos:{losses["loss_normal_cos"].item():.4f}, depth:{losses["loss_depth"].item():.4f}, '
                      f'curvature:{losses["loss_curvature"].item():.4f}, [psnr]: {psnr.item():.4f}, [α]: {1/self.model.module.density.beta.item():.4f}, '
                      f'[active_levels]: {self.model.module.sdf.active_levels}/{self.model.module.sdf.num_levels}')

                if self.gpu == 0:
                    for key, value in losses.items(): # log loss
                        self.loger.add_scalar(tag="loss" + '/' + key, scalar_value=value, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+ '/psnr', scalar_value=psnr, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+ '/alpha', scalar_value=1/self.model.module.density.beta, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+ '/active_levels', scalar_value=self.model.module.sdf.active_levels, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+'/epoch', scalar_value=epoch, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+'/lr', scalar_value=self.optimizer.param_groups[1]['lr'], global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+'/w_curvature', scalar_value=self.loss.lambda_curvature, global_step=self.cur_step)
                
                # scheduler step
                self.scheduler.step()
            # save checkpoint
            self.save_checkpoint(epoch,save_epoch=epoch%self.conf.train.save_freq==0 and self.gpu==0)
            # plot
            if self.gpu==0 and epoch % self.conf.train.plot_freq == 0:
                self.plot(epoch)  # plot


if __name__ == '__main__':
    opt = get_args()
    gpu = init_processes()

    ti=time.time()
    trainer = Trainer(opt, gpu)
    trainer.train()
    h,m,s=convert_seconds(time.time()-ti)
    print('successful!, total time: {}h {}m {}s'.format(h,m,s))
