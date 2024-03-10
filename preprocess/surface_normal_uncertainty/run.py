import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data.dataloader_custom import CustomLoader
from models.NNET import NNET
import utils.utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def test(model, test_loader, device, results_dir, imgs_dir,normal_dir, uncertainty_dir):
    alpha_max = 60
    kappa_max = 30
    A=[normal_dir,uncertainty_dir]
    B=['res','vis']
    for a in A:
        for b in B:
            os.makedirs(os.path.join(a,b),exist_ok=True)
    os.makedirs(os.path.join(normal_dir,'vis_concat'),exist_ok=True)
    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # save results
            img_name = data_dict['img_name'][0]

            # 1. save input image
            img = utils.unnormalize(img[0, ...])

            target_path = '%s/%s_img.png' % (results_dir, img_name)
            # plt.imsave(target_path, img)

            # 2. predicted normal
            pred_norm[:,:,:,:]=-pred_norm[:,:,:,:] # 这里不知道为什么要乘以-1（才和OPENCV相机坐标系的local normal是一致的），待研究
            pred_norm = pred_norm / np.linalg.norm(pred_norm, axis=-1, keepdims=True) # 归一化
            pred_norm =(pred_norm + 1) * 0.5 # [-1,1]^3 -> [0,1]^3 # 统一规定存储时的范围
            pred_norm_rgb = pred_norm * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)
            normal_rgb=Image.fromarray(pred_norm_rgb[0])

            target_path = '%s/%s_pred_norm.png' % (results_dir, img_name)
            normal_rgb.save(os.path.join(normal_dir,'vis',img_name+'.png'))
            # plt.imsave(os.path.join(normal_dir,'vis',img_name+'.png'), pred_norm_rgb[0, :, :, :]) # save normal rgb
            np.save(os.path.join(normal_dir,'res',img_name+'.npy'),pred_norm[0]) # save normal data .npy

            # 3. predicted kappa (concentration parameter)
            target_path = '%s/%s_pred_kappa.png' % (results_dir, img_name)
            # plt.imsave(target_path, pred_kappa[0, :, :, 0], vmin=0.0, vmax=kappa_max, cmap='gray')

            # 4. predicted uncertainty
            pred_alpha = utils.kappa_to_alpha(pred_kappa)
            target_path = '%s/%s_pred_alpha.png' % (results_dir, img_name)
            plt.imsave(os.path.join(uncertainty_dir,'vis',img_name+'.png'), pred_alpha[0, :, :, 0], vmin=0.0, vmax=alpha_max, cmap='jet') # save uncertainty visual jet rgb
            np.save(os.path.join(uncertainty_dir,'res',img_name+'.npy'), pred_alpha[0,:,:,0]/alpha_max)


            # 5. concatenated results
            # image_path_list = [imgs_dir,normal_dir+'/vis',uncertainty_dir+'/vis']
            # image_path_list = ['%s/%s.png' % (i, img_name) for i in image_path_list]
            # target_path = '%s/vis_concat/%s.png' % (normal_dir, img_name)
            # utils.concat_image(image_path_list, target_path)


if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args
    parser.add_argument('-i','--imgs_dir', default='./examples', type=str)
    parser.add_argument('-n','--normal_dir',default='./results/normal',type=str)
    parser.add_argument('-u','--uncertainty_dir',default='./results/uncertainty',type=str)

    parser.add_argument('--architecture', required=True, type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", required=True, type=str, help="{nyu, scannet}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    device = torch.device('cuda:0')

    # load checkpoint
    checkpoint = './checkpoints/%s.pt' % args.pretrained
    print('loading checkpoint... {}'.format(checkpoint))
    model = NNET(args).to(device)
    model = utils.load_checkpoint(checkpoint, model)
    model.eval()
    print('loading checkpoint... / done')

    # test the model
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    test_loader = CustomLoader(args, args.imgs_dir).data
    test(model, test_loader, device, results_dir, args.imgs_dir, args.normal_dir, args.uncertainty_dir)

