import torch
from models.loss import compute_scale_and_shift
import numpy as np
from PIL import Image
import cv2

def convert_seconds(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return [int(h), int(m), int(s)]


def split_input(sample, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    split_keys = ['rays_o', 'rays_d', 'depth_scale']
    for i in range(0, total_pixels, n_pixels):
        data = {}
        data['idx'] = sample['idx']
        data['K'] = sample['K']
        data['pose'] = sample['pose']
        for key in split_keys:
            data[key] = sample[key][:, i:i + n_pixels, :] # (B, chunk, -1)
        split.append(data)
    return split

def merge_output(outputs):
    # Merge the split output.
    merge = {}
    for entry in outputs[0]:
        if outputs[0][entry] is None:
            continue
        merge[entry] = torch.cat([r[entry] for r in outputs],1) # r[entry] shape: (batch_size, chunk, -1)
    return merge


def get_plot_data(outputs, sample, h, w):
    # outputs: {'rgb': (B, h*w, 3), 'depth': (B, h*w, 1), 'normal': (B, h*w, 3)}
    batch_size = outputs['rgb'].shape[0]
    scale_shifts = compute_scale_and_shift(outputs['depth'], sample['depth'])
    plot_outputs = []
    for b in range(batch_size):
        plot_output = {}
        idx = sample['idx'][b].item()  # image index
        rgb = outputs['rgb'][b].cpu().numpy()
        depth = outputs['depth'][b].cpu().numpy()
        normal = outputs['normal'][b].cpu().numpy()
        gt_rgb = sample['rgb'][b].cpu().numpy()
        gt_depth = sample['depth'][b].cpu().numpy()
        gt_normal = sample['normal'][b].cpu().numpy()
        # 处理可视化
        # rgb
        cat_rgb = np.concatenate([rgb.reshape(h, w, 3), gt_rgb.reshape(h, w, 3)], axis=0)
        plot_rgb = Image.fromarray((cat_rgb * 255).astype(np.uint8))
        # normal
        normal, gt_normal = (normal + 1) / 2, (gt_normal + 1) / 2
        cat_normal = np.concatenate([normal.reshape(h, w, 3), gt_normal.reshape(h, w, 3)], axis=0)
        plot_normal = Image.fromarray((cat_normal * 255).astype(np.uint8))
        # depth
        scale, shift = scale_shifts[b, 0, 0].item(), scale_shifts[b, 1, 0].item()
        depth = scale * depth + shift
        depth, gt_depth = (depth - depth.min()) / (depth.max() - depth.min()), (gt_depth - gt_depth.min()) / (
                    gt_depth.max() - gt_depth.min())
        depth_bgr = cv2.applyColorMap((depth.reshape(h, w) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        gt_depth_bgr = cv2.applyColorMap((gt_depth.reshape(h, w) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cat_depth = np.concatenate([depth_bgr, gt_depth_bgr], axis=0)
        plot_depth = Image.fromarray(cv2.cvtColor(cat_depth, cv2.COLOR_BGR2RGB))
        # merge
        plot_merge = Image.fromarray(
            np.concatenate([np.array(plot_rgb), np.array(plot_normal), np.array(plot_depth)], axis=1))

        plot_output['idx'] = idx
        plot_output['rgb'] = plot_rgb
        plot_output['depth'] = plot_depth
        plot_output['normal'] = plot_normal
        plot_output['merge'] = plot_merge
        plot_outputs.append(plot_output)
    return plot_outputs