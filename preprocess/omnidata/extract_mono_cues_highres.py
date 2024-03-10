# omnidata要求输入图像分辨率必须是384×384
# TODO：在不不损失信息的前提下对high-res任意分辨率的图像输入进行切割对齐拼接，提取先验。
import numpy as np
import torch
import os
import argparse
import PIL 
from PIL import Image
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import transforms
from align_utils import align_normal_y, align_y, best_fit_transform
from data.transforms import get_transform
from modules.midas.dpt_depth import DPTDepthModel


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input_dir', type=str, default=None, help="输入图像分辨率宽高比不一致且分辨率远大于384时, 这时可以对图像进行拼拼乐了")
    args.add_argument('--task', choices=['normal', 'depth'])
    args.add_argument('--output_dir', type=str, default=None)

    return args.parse_args()

def get_img_transform(task):

    if task == 'depth':
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(384, interpolation=PIL.Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
    elif task == 'normal':
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(384, interpolation=PIL.Image.BILINEAR), 
                get_transform('rgb', image_size=None),
            ]
        )
    
    return trans_totensor

def get_img_slices(img):
    # 384/3=128
    h,w = img.shape[:2]
    slice_map = []
    for i in range(0,h,256):
        for j in range(0,w,256):
            img_slice = img[i:i+256,j:j+256]
            if img_slice.shape[0]==256 and img_slice.shape[1]==256:
                img_slices.append(img_slice)

def get_depth_output(img, model, trans_totensor, device):
    img_slices,a,offset1,offset2= get_img_slices(img)
    with torch.no_grad():
        input_tensor = torch.stack(list(trans_totensor(Image.fromarray(img)) for img in img_slices),dim = 0).to(device)
        output = model(input_tensor).clamp(min=0, max=1)
        output = torch.nn.functional.interpolate(output.unsqueeze(1), size=640, mode='bilinear')
        output = output.squeeze(1).cpu() # (3, 640, 640), 3就是slice 0,mid,1
        if w>h:
            output = output.permute(0,2,1) # 以便都在y轴上对齐

        depth = align_y(output[0][None], output[1][None], offset1, a, 0, a-offset1)
        depth = align_y(depth, output[2][None], depth.shape[1]-(a-offset2), depth.shape[1], 0, a-offset2)
        depth = depth.squeeze(0).float()
        if w>h:
            depth = depth.permute(1,0)
        # print(depth.max(), depth.min(), depth.mean())
        return depth

def save_depth_output(img, depth, output_dir, img_name):
    output_path = os.path.join(output_dir, 'vis','{}.png'.format(img_name))
    plt.imsave(output_path, depth)
    depth_img = np.asarray(Image.open(output_path))[..., :3]
    depth_img = np.concatenate([img, depth_img], axis = 1)
    Image.fromarray(depth_img).save(output_path)
    np.save(os.path.join(output_dir, 'res','{}.npy'.format(img_name)), depth)

def get_normal_output(img, model, trans_totensor, device):
    h,w = img.shape[:2]
    img_slices,a,offset1,offset2 = get_img_slices(img)
    with torch.no_grad():
        input_tensor = torch.stack(list(trans_totensor(Image.fromarray(img)) for img in img_slices),dim = 0).to(device)
        output = model(input_tensor).clamp(min=0, max=1)
        output = torch.nn.functional.interpolate(output, size=640, mode='bilinear') # (3, 3, 640, 640)
        # output = output.clamp(0, 1)
        if w>h:
            output = output.permute(0,1,3,2)

        output = output * 2 - 1 # [0,1] -> [-1,1]
        output = output / (output.norm(dim = 1, keepdim = True) + 1e-15)
        output = output.float().detach().cpu().numpy()
        normal = align_normal_y(output[0], output[1], offset1, a, 0, a-offset1)
        normal = align_normal_y(normal, output[2],  normal.shape[2]-(a-offset2), normal.shape[2], 0, a-offset2)

        R = best_fit_transform(normal[:, offset1:-offset2, :].reshape(3, -1).T, output[1].reshape(3, -1).T)
        normal = (R @ normal.reshape(3, -1)).reshape(normal.shape)
        if w>h:
            normal = normal.permute(0,2,1)
    return normal

def save_normal_output(img, normal, output_dir, img_name):
    normal = (normal + 1) / 2
    output_path = os.path.join(output_dir, 'vis','{}.png'.format(img_name))
    # print(output_path)
    normal_img = (normal * 255)
    normal_img = normal_img.astype(np.uint8)
    normal_img = np.concatenate([img, normal_img], axis = 1)
    Image.fromarray(normal_img).save(output_path)
    np.save(os.path.join(output_dir, 'res','{}.npy'.format(img_name)), normal.transpose(1, 2, 0))

def main(args): 
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrained_weights_path = os.path.join('./pretrained_models', f"omnidata_dpt_{args.task}_v2.ckpt")  # 'omnidata_dpt_depth_v1.ckpt'
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3 if args.task == 'normal' else 1)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    trans_totensor = get_img_transform(args.task)

    img_paths = glob.glob(os.path.join(args.input_dir, '*.jpg'))+glob.glob(os.path.join(args.input_dir, '*.png'))
    img_paths.sort()

    output_dir = os.path.join(args.output_dir, 'mono_'+args.task)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'res'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)

    for idx, ipath in enumerate(tqdm(img_paths)):
        img = np.asarray(Image.open(ipath))
        img_name = os.path.basename(ipath).split('.')[0]
        if args.task == 'depth':
            depth = get_depth_output(img, model, trans_totensor, device)
            save_depth_output(img, depth, output_dir, img_name)
        elif args.task == 'normal':
            normal = get_normal_output(img, model, trans_totensor, device)
            save_normal_output(img, normal, output_dir, img_name)

if __name__ == "__main__":
    args = get_args()
    main(args)