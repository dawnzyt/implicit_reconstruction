import os

import cv2
import numpy as np

def gradient(x):
    gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, gm

# 不确定性阈值
U='/data/scannet/scans/scene0616_00/uncertainty/res/000{}.npy'
U_vis='/data/scannet/scans/scene0616_00/uncertainty/vis/000{}.png'
NORMAL='/data/scannet/scans/scene0616_00/mono_normal/res/000{}.npy'
RGB='/data/scannet/scans/scene0616_00/rgb/000{}.png'
MASK = '/data/scannet/scans/scene0616_00/mask/res/000{}.npy'
outdir='./normal_mask_vis'
u_threshold = 0.3 # 越大flat区域越多
u_grad_threshold = 0.03 # 越大flat区域越多
os.makedirs(outdir, exist_ok=True)
for i in range(303):
    u1=np.load(U.format(str(i).zfill(3)))
    u_vis=cv2.imread(U_vis.format(str(i).zfill(3)))
    normal = np.load(NORMAL.format(str(i).zfill(3)))
    rgb=cv2.imread(RGB.format(str(i).zfill(3)))
    mask = np.load(MASK.format(str(i).zfill(3)))

    normal = cv2.cvtColor((normal*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # u_img = (u1*255).astype(np.uint8)
    _,_,grad_u = gradient(u1)
    normal_mask = (mask==1)|((mask==0)&(u1<u_threshold)&(grad_u<u_grad_threshold))
    rgb[~normal_mask]=0

    # u2=u1.copy()
    # threshold2 = 0.3
    u1[u1 > u_threshold] = 255
    u1[u1 <= u_threshold] = 0
    # u2[u2 > threshold2] = 255
    # u2[u2 <= threshold2] = 0

    u_flat=(255-u1).astype(np.uint8)
    u_flat=cv2.cvtColor(u_flat, cv2.COLOR_GRAY2BGR)
    u_cat=np.concatenate([np.concatenate([u_flat,u_vis], axis = 1),np.concatenate([normal,rgb], axis = 1)], axis = 0)
    cv2.imshow(f'u{i}', u_cat)
    cv2.imwrite(f'{outdir}/u{i}.png', u_cat)
    cv2.waitKey(1)
    cv2.destroyWindow(f'u{i}')
print(1)

# for i in range(303):
#     img=cv2.imread('/data/scannet/scans/scene0616_00/rgb/000{}.png'.format(str(i).zfill(3)))
#     mask = (~np.load('/data/scannet/scans/scene0616_00/mask/res/000{}.npy'.format(str(i).zfill(3))))[...,None]
#     mask = mask.astype(np.uint8)*255
#     # 创建限制对比度的自适应直方图均衡化对象
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#
#     # 转换空间到YCrCb
#     ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#
#     # 分离通道
#     y, cr, cb = cv2.split(ycc)
#
#     # 获取前景和背景的亮度图像
#     foreground = cv2.bitwise_and(y, y, mask=mask)
#     background = cv2.bitwise_and(y, y, mask=cv2.bitwise_not(mask))
#
#     # 创建限制对比度的自适应直方图均衡化对象
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#
#     # 对前景和背景的亮度图像进行对比度增强
#     y=clahe.apply(y)
#     foreground = clahe.apply(foreground)
#     background = clahe.apply(background)
#
#     # 将对比度增强后的前景和背景的亮度图像进行加权叠加，得到最终的Y通道
#     # y = cv2.addWeighted(foreground+background, 0.8, y, 0.2, 0)
#
#     # 合并Y通道和原始的Cr和Cb通道，得到最终的contrast_img
#     contrast_img = cv2.merge([y, cr, cb])
#
#     # 转换回RGB空间
#     contrast_img = cv2.cvtColor(contrast_img, cv2.COLOR_YCrCb2BGR)
#
#     # 显示和保存结果
#     cv2.imshow('img', img)
#     cv2.imshow('mask', mask)
#     cv2.imshow('contrast_img', contrast_img)
#     cv2.imwrite('/data/scannet/scans/scene0616_00/rgb/000{}.png'.format(str(i).zfill(3)), contrast_img)
#     # cv2.waitKey(0)
#     cv2.destroyAllWindows()


# de_omni= '/data/monosdf/scannet/scan4/000000_depth.npy'
# de_de_any= '/data/scannet/scans/scene0616_00/mono_depth/res/000000.npy'
# depth_omni=np.load(de_omni)
# depth_de_any=np.load(de_de_any)
# depth_omni = (depth_omni - depth_omni.min()) / (depth_omni.max() - depth_omni.min())*255
# depth_omni = depth_omni.astype(np.uint8)
# depth_de_any = (depth_de_any - depth_de_any.min()) / (depth_de_any.max() - depth_de_any.min())*255
# depth_de_any = depth_de_any.astype(np.uint8)
# depth_omni_gray = cv2.cvtColor(depth_omni, cv2.COLOR_GRAY2BGR)
# depth_de_any_gray = cv2.cvtColor(depth_de_any, cv2.COLOR_GRAY2BGR)
# cv2.imshow('depth_omni', depth_omni_gray)
# cv2.imshow('depth_de_any', depth_de_any_gray)
# cv2.waitKey(0)
