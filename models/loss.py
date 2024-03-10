import torch
import torch.nn as nn

# TODO: 这里计算scale和shift应该分batch各自计算，只考虑了单个batch的情况 [√]
def compute_scale_and_shift(pred, target, mask=None):
    # 标量之间求解线性回归最小二乘解系数即：min||s*pred+d-target||
    # pred、target: (B, ...), (B, ...)
    # 构造最小二乘：min L=∑|Aix-bi|^2, A=[pred, 1], x=[s,t]^T为求解系数,b则为[target]
    # 得A^TAx=A^Tb,求解即可
    if mask is not None:
        pred, target = pred * mask.float(), target * mask.float()
    else:
        mask = torch.ones_like(pred)
    dim = tuple(range(1, pred.dim()))
    a00 = (pred ** 2).sum(dim)
    a01 = pred.sum(dim)
    a10 = a01
    a11 = mask.float().sum(dim)

    b00 = (pred * target).sum(dim)
    b10 = target.sum(dim)
    b = torch.stack([b00, b10], dim=-1)[..., None]  # (B, 2, 1)
    # 手解线性方程组，对于Ax=b, 直接x=A^-1b, A^-1=A*/det(A)
    det = a00 * a11 - a01 * a10 # (B, )
    det=det.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
    adjoint = torch.stack([a11, -a10, -a01, a00], dim=-1).reshape(pred.shape[0], 2, 2)
    valid = torch.nonzero(det.ravel()) # (n, 1), 第二维指det有多少维，标量因此为1
    valid = valid.ravel()
    x=torch.zeros_like(b)
    x[valid] = torch.bmm(adjoint[valid] / det[valid], b[valid]) # (B, 2, 1)
    return x


def get_psnr(pred, gt, mask=None):
    mse_signal = mse_loss(pred, gt, mask)
    return -10 * torch.log10(mse_signal)

def gradient_loss(prediction, target, mask=None):
    # prediction，target: (B, H, W, 1)，估计图像和真实图像的patch（但实际做的时候H,W的patch像素点都是随机、乱序的，互相没有关系，计算grad意义不明确，但确实有效。）
    # mask: (B, H, W, 1)
    if mask is None:
        mask = torch.ones_like(prediction)
    M = mask.float().sum()

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return image_loss.sum() / M

def mse_loss(pred, gt, mask=None):
    # mask: (B, R, 1)
    error = (pred - gt) ** 2  # [B,R,C]
    if mask is not None:
        return (error * mask.float()).sum()/torch.sum(mask)/error.shape[-1] # ÷C
    else:
        return error.mean()


def l1_loss(pred, gt, mask=None, uncertainty=None):
    error = (pred - gt).abs()  # [B,R,C]
    if uncertainty is not None: # uncertainty越大, normal越不可信, rgb权重越大
        weight = torch.minimum(0.5 + uncertainty, torch.ones_like(uncertainty))
        error = error * weight
    if mask is not None:
        return (error * mask.float()).sum()/torch.sum(mask)/error.shape[-1]
    return error.mean()


def eikonal_loss(gradients, mask=None):
    # eikonal-loss规范sdf
    # gradients: (..., 3), mask一般为None
    error = (gradients.norm(dim=-1, keepdim=True) - 1.0) ** 2
    error = error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        return (error * mask.float()).sum()/torch.sum(mask)
    else:
        return error.mean()


# angelo提出的曲率loss：最小化hessian矩阵和的绝对值。
def curvature_loss(hessian, mask=None):
    # hessian: (B, R, N_hessian, 3), mask: (B, R, 1)
    if hessian is None:
        return torch.tensor(0.0, device=mask.device)
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N_hessian]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N_hessian]
    if mask is not None:
        M=mask.float().sum()*laplacian.shape[-1]
        if M==0:
            return torch.tensor(0.0, device=mask.device)
        return (laplacian * mask.float()).sum()/M
    else:
        return laplacian.mean()


def smooth_loss(g1, g2, mask=None):
    # g1、g2分别是场景中邻居点的sdf gradient:(...,3), 一般无需mask
    if g2 is None:
        return torch.tensor(0.0, device=g1.device)
    normals_1 = g1 / (g1.norm(2, dim=-1,keepdim=True) + 1e-6)
    normals_2 = g2 / (g2.norm(2, dim=-1,keepdim=True) + 1e-6)
    smooth_error = torch.norm(normals_1 - normals_2, dim=-1,keepdim=True)
    if mask is not None:
        M = mask.float().sum()
        if M == 0:
            return torch.tensor(0.0, device=mask.device)
        return (smooth_error * mask.float()).sum()/torch.sum(mask)
    return smooth_error.mean()


def normal_loss(normal_pred, normal_gt, mask=None, uncertainty=None):
    # 单目normal先验监督，一般来说l1足够
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1, keepdim=True)
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1, keepdim=True))
    if uncertainty is not None:
        weight = torch.maximum(2 - uncertainty, torch.zeros_like(uncertainty)) # uncertainty越大，normal越不可信，权重越小
        l1, cos = l1 * weight, cos * weight
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device), torch.tensor(0.0, device=mask.device)
        return (l1 * mask.float()).sum()/M, (cos * mask.float()).sum()/M
    return l1.mean(), cos.mean()


def depth_loss(depth_pred, depth_gt, mask=None, monocular=True, cal_grad_loss=True):
    # 单目/真实深度监督
    # depth_pred, depth_gt: (B, R, 1), mask: (B, R, 1)
    if monocular:
        scale_shift = compute_scale_and_shift(depth_pred, depth_gt, mask=mask) # (B, 2, 1)
        scale, shift = scale_shift[:, :1, :], scale_shift[:, 1:, :]
        depth_pred = scale * depth_pred + shift
    loss= mse_loss(depth_pred, depth_gt, mask=mask)
    if cal_grad_loss: # depth gradient loss
        depth_pred = depth_pred.reshape(-1, 32, 32,1)
        depth_gt = depth_gt.reshape(-1, 32, 32,1)
        if mask is not None:
            mask = mask.reshape(-1, 32, 32,1)
        loss = loss+ 0.5*gradient_loss(depth_pred, depth_gt, mask=mask)
    return loss




# loss概览：
# 1. eikonal_loss: 用于约束梯度的模长为1
# 2. color_loss: 体渲染rgb loss(l1/l2)
# 3. smooth_loss/曲率loss:
# 4. normal_loss:
# *5. depth_loss:
# *6. l2 osf mask loss
# *7. l3 osf loss
# *8. mvs refine osf loss
class ImplicitReconLoss(nn.Module):
    def __init__(self,
                 lambda_rgb=1.0,
                 lambda_eik=0.05,
                 lambda_smooth=0.005,
                 lambda_normal=0.05,
                 lambda_depth=0.1,
                 lambda_curvature=0.1,
                 lambda_2d_osf=0.1,
                 lambda_3d_osf=0.1,
                 lambda_mvs_osf=0.1,
                 warm_up_end=0):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_eik = lambda_eik
        self.lambda_smooth = lambda_smooth
        self.lambda_normal = lambda_normal
        self.lambda_depth = lambda_depth
        self.init_lambda_curvature = lambda_curvature
        self.lambda_2d_osf = lambda_2d_osf
        self.lambda_3d_osf = lambda_3d_osf
        self.lambda_mvs_osf = lambda_mvs_osf

        self.warm_up_end = warm_up_end
        self.torch_l1=torch.nn.L1Loss()

    def set_curvature_weight(self, cur_step, anneal_levels):
        # 1.38098是grid res指数增长系数
        sched_weight = self.init_lambda_curvature
        if cur_step <= self.warm_up_end:
             sched_weight *= cur_step / self.warm_up_end
        else:
            decay_factor = 1.38098 ** (anneal_levels - 1)
            sched_weight /= decay_factor
        self.lambda_curvature = sched_weight

    def forward(self, output, sample):
        outside = output['outside']
        sdf=output['sdf']
        uncertainty = sample['uncertainty']
        uncertainty_grad = sample['uncertainty_grad']
        foreground_mask = (sdf > 0.).any(dim=-2) & (sdf < 0.).any(dim=-2) # 过滤掉射出场景的点
        mask = sample['mask'] # bg mask, bg包括ceiling.wall.floor.poster.window
        u_threshold = 0.3  # 越大flat区域越多
        u_grad_threshold = 0.03  # 越大flat区域越多
        normal_mask=((mask==1)|((mask==0)&(uncertainty<u_threshold)&(uncertainty_grad<u_grad_threshold)))

        # 1. eikonal_loss
        loss_eik = eikonal_loss(output['gradient_eik'],mask=None)
        # 2. color_loss
        loss_rgb = l1_loss(output['rgb'], sample['rgb'], mask=(~outside))
        # 3. smooth_loss
        loss_smooth = smooth_loss(output['gradient_eik'], output['gradient_eik_neighbor'], mask=None)
        # 4. normal_loss
        loss_normal_l1, loss_normal_cos = normal_loss(output['normal'], sample['normal'], mask=(~outside)&foreground_mask&normal_mask) # [MONO-3] bg或者object的平整区域
        # 5. depth_loss
        loss_depth = depth_loss(output['depth'], sample['depth']*5+0.05, mask=(~outside)&foreground_mask, monocular=True) # [MONO-4]
        # 6. curvature_loss
        loss_curvature = curvature_loss(output['hessian'], mask=(~outside))
        # loss_curvature = torch.tensor(0.0, device=mask.device)

        losses = {}
        losses['loss_eik'] = loss_eik
        losses['loss_rgb'] = loss_rgb
        losses['loss_smooth'] = loss_smooth
        losses['loss_normal_l1'] = loss_normal_l1
        losses['loss_normal_cos'] = loss_normal_cos
        losses['loss_depth'] = loss_depth
        losses['loss_curvature'] = loss_curvature

        loss= self.lambda_rgb * loss_rgb + self.lambda_eik * loss_eik + self.lambda_smooth * loss_smooth + \
              self.lambda_normal * (loss_normal_l1 + loss_normal_cos) + self.lambda_depth * loss_depth + \
                self.lambda_curvature * loss_curvature
        losses['loss'] = loss
        # print(f'sum outside: {(~outside).sum().item()}, sum outside&mask: {((~outside)&mask).sum().item()}, loss: {loss.item()}')
        # if torch.isnan(loss):
        #     raise ValueError('loss is nan')
        return losses