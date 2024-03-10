import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import matplotlib.pyplot as plt
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def turn_disp_to_depth(disp, rate=50):
    # rate设置表示最远距离最多是最近距离（最大视差）的100倍，超过百倍距离clamp掉。
    # 参考issue：https://github.com/LiheYoung/Depth-Anything/issues/47
    max_disp = disp.max()
    range=np.minimum(rate,max_disp/(disp.min()+1e-5))
    minn = max_disp/range
    disp=np.maximum(disp,minn) # clamp
    depth=1/disp
    print(f'estimated depth range: {depth.min():.4f}m - {depth.max():.4f}m, mean: {depth.mean():.4f}m')
    return depth.min(),depth.max(),depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir,'res'),exist_ok=True)
    os.makedirs(os.path.join(args.outdir,'vis'),exist_ok=True)
    min_depth = 1e5
    max_depth = 0
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        actual_disp = depth.cpu().numpy() # depth-anything默认输出是disp（虽然他写的depth），需要转换成depth
        minn,maxx,depth1= turn_disp_to_depth(actual_disp)
        min_depth = min(min_depth, minn)
        max_depth = max(max_depth, maxx)

        depth1_normalized = (depth1 - depth1.min()) / (depth1.max() - depth1.min()) * 255.0
        depth1_normalized = depth1_normalized.astype(np.uint8)

        depth_rgb = cv2.applyColorMap(depth1_normalized, cv2.COLORMAP_INFERNO)
        depth_gray = cv2.cvtColor(depth1_normalized, cv2.COLOR_GRAY2BGR)

        filename = os.path.basename(filename)
        np.save(os.path.join(args.outdir,'res',filename[:-4]+'.npy'),depth1)
        # cv2.imwrite(os.path.join(args.outdir,'vis',filename),depth_gray) # depth anything 映射到`INFERNO` color map
        plt.imsave(os.path.join(args.outdir,'vis',filename),depth1_normalized,cmap='viridis') # monosdf的mono depth rgb可视化到`viridis`
        
        # if args.grayscale:
        #     depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        # else:
        #     depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        # filename = os.path.basename(filename)
        
        # if args.pred_only:
        #     cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        # else:
        #     split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
        #     combined_results = cv2.hconcat([raw_image, split_region, depth])
            
        #     caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
        #     captions = ['Raw image', 'Depth Anything']
        #     segment_width = w + margin_width
            
        #     for i, caption in enumerate(captions):
        #         # Calculate text size
        #         text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        #         # Calculate x-coordinate to center the text
        #         text_x = int((segment_width * i) + (w - text_size[0]) / 2)

        #         # Add text caption
        #         cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
        #     final_result = cv2.vconcat([caption_space, combined_results])
            
        #     cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)
    print(f'Total, min_depth: {min_depth:.4f}m, max_depth: {max_depth:.4f}m')