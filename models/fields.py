# fields including SDF, Radiance, etc.
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys

sys.path.append(os.getcwd())
from models.encoder import *
from models.hash_encoder.hash_encoder import HashEncoder
from models.nerf_util import MLPwithSkipConnection

# debsdf, sdf bias function。
class SDFTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = [0.01, 0.2]
        self.iter_steps = 0
        self.iter_end = 3000

    def get_ratio(self):
        self.iter_steps += 1
        if self.iter_steps > self.iter_end:
            return 1
        return self.iter_steps / self.iter_end

    def forward(self, pts, deltas, sdf, ray_d, normal, sdf_network):
        # print('delta max: {}, min: {}'.format(deltas.max().item(), deltas.min().item()))
        deltas_clamp = torch.clamp(torch.abs(sdf) / 4, min=self.delta[0], max=self.delta[1])
        # deltas_s= torch.clamp(deltas, min=0.01, max=0.01)
        pts_delta = pts + deltas_clamp * ray_d
        # pts_delta_s = pts + deltas_s * ray_d # ...
        _, _, grad_delta = sdf_network.get_outputs(pts_delta)
        normal_delta = grad_delta / grad_delta.norm(2, -1, keepdim=True)
        # _, _, grad_delta_s = sdf_network.get_outputs(pts_delta_s)
        # normal_delta_s = grad_delta_s / grad_delta_s.norm(2, -1, keepdim=True)

        # detach normal
        normal = normal.detach()
        normal_delta = normal_delta.detach()
        # normal_delta_s = normal_delta_s.detach()
        # 1. compute R
        cos_A = torch.sum(ray_d * normal, -1, keepdim=True)
        cos_B = torch.sum(ray_d * normal_delta, -1, keepdim=True)
        # cos_B_s = torch.sum(ray_d * normal_delta_s, -1, keepdim=True)
        concave_flag = cos_A > cos_B

        # warm up
        ratio = self.get_ratio()
        cos_A[cos_A > 0] = torch.pow(cos_A[cos_A > 0], ratio)  # warm up
        cos_A[cos_A < 0] = -torch.pow(-cos_A[cos_A < 0], ratio)
        cos_A = cos_A.clamp(min=-1, max=1)

        cos_B[cos_B > 0] = torch.pow(cos_B[cos_B > 0], ratio)  # warm up
        cos_B[cos_B < 0] = -torch.pow(-cos_B[cos_B < 0], ratio)
        cos_B = cos_B.clamp(min=-1, max=1)

        # cos_B_s[cos_B_s>0] = torch.pow(cos_B_s[cos_B_s>0], ratio) # warm up
        # cos_B_s[cos_B_s<0] = -torch.pow(-cos_B_s[cos_B_s<0], ratio)
        # cos_B_s=cos_B_s.clamp(min=-1,max=1)

        sin_A = torch.sqrt(1 - cos_A ** 2)
        sin_B = torch.sqrt(1 - cos_B ** 2)
        # sin_B_s = torch.sqrt(1 - cos_B_s ** 2)

        # angle |A-B|,
        angle = torch.acos(torch.clamp(cos_A * cos_B + sin_A * sin_B, min=-1, max=1)) / torch.pi * 180
        # angle_s = torch.acos(torch.clamp(cos_A * cos_B_s + sin_A * sin_B_s, min=-1, max=1))/ torch.pi * 180
        # print('mean angle: {}'.format(angle.mean().item() / torch.pi * 180))
        # rd_idx=torch.randint(0, pts.shape[0], (1,))

        sin_alpha = torch.abs(sin_A * cos_B - cos_A * sin_B)

        # DEBUG：有一个问题, 当R>0时, sdf>>R, 也就是明明不会相交判断为相交, 此时近似表面形状的圆a=R-sdf<0
        R = deltas_clamp * (sin_B + 1e-12) / (sin_alpha + 1e-12)

        R[concave_flag] *= -1  # A<B: 凹且R<0, A>B: 凸且R>0
        # print('delta: {}, angle: {}, R: {}'.format(deltas[rd_idx].item(), angle[rd_idx].item() / torch.pi * 180, R[rd_idx].item()))
        # planar_mask = R>23.3
        # print('sum of planar mask: {}'.format(planar_mask.sum()))
        R = torch.clamp(R, max=23.3, min=-23.3)

        # 暂时不考虑planar_mask
        # 2. compute y
        a = R - sdf
        l_square = a ** 2 - R ** 2 * sin_A ** 2  # l是ray与a交点在ray上的投影长度
        y = torch.zeros(a.shape).cuda()
        # 相交情况: 所有表面为凹时和 部分为凸相交时(圆心离直线的距离rsinA < a)
        # y=RcosA-sign(R)*sqrt(a^2-r^2sin^2A)
        intersect_mask = l_square >= 0
        # 若pts在表面内: sdf<0, 得到的y自然为负。
        y[intersect_mask] = R[intersect_mask] * cos_A.abs()[intersect_mask] - torch.sign(
            a[intersect_mask]) * torch.sqrt(l_square[intersect_mask])
        # 不相交情况:一定是凸
        y[~intersect_mask] = (sdf[~intersect_mask]) / (cos_A.abs()[~intersect_mask] + 1e-12) + sdf[~intersect_mask] * (
                    sin_A[~intersect_mask] / (cos_A[~intersect_mask] + 1e-12)).abs()

        # y*=torch.sign(sdf)
        # y_minus_sdf=y-sdf
        # d_max_idx=torch.argmax(y_minus_sdf.abs())
        # print('max abs(sdf-y): {}'.format((sdf-y).abs().max().item()))
        # nan->0
        y = torch.clamp(y, min=-23.3, max=23.3)
        y[torch.isnan(y)] = 0
        return y


# volsdf, sdf->density的laplace变换
class LaplaceDensity(nn.Module):
    def __init__(self, beta_init=0.1, beta_min=1e-4):
        super().__init__()
        self.beta_min = torch.tensor(beta_min)
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta # 为方便
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

# (x)->(sdf, z)
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in=3,
                 d_hidden=256,
                 n_layers=8,
                 skip=[4],  # 1-n_layers
                 geometric_init=True,
                 bias=0.5,  # 初始化球半径
                 norm_weight=True,
                 inside_outside=False,  # default outside for object
                 bound=1.0,
                 enable_fourier=True,
                 N_freqs=10, # 'fourier'
                 enable_hashgrid=True,
                 num_levels=16,
                 per_level_dim=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 max_resolution=2048,
                 resolution_list=None,
                 enable_progressive=False,
                 init_active_level=4,
                 active_step = 5000,
                 gradient_mode='analytical',
                 taps=6,
                 ):
        super(SDFNetwork, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.skip = skip
        self.geometric_init = geometric_init
        self.bias = bias
        self.norm_weight = norm_weight
        self.inside_outside = inside_outside
        self.bound = bound
        self.enable_fourier = enable_fourier
        self.N_freqs = N_freqs
        self.enable_hashgrid = enable_hashgrid
        self.num_levels = num_levels
        self.per_level_dim = per_level_dim

        ############### progressive grid ##################
        self.enable_progressive = enable_progressive
        self.init_active_level = init_active_level
        self.active_levels = 0
        self.active_step = active_step
        self.warm_up = 0 # [MONO-1]

        # epsilon for numerical gradient
        self.gradient_mode = gradient_mode  # 'numerical' or 'analytical'
        self.taps = taps  # 6 or 4
        self.normal_epsilon = 0

        # encoder
        if self.enable_hashgrid:
            self.grid_encoder = HashEncoder(x_dim=d_in, num_levels=num_levels, per_level_dim=per_level_dim,
                                       log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
                                       max_resolution=max_resolution, resolution_list=resolution_list)
            self.d_in = self.grid_encoder.encoded_length() + self.d_in
        if self.enable_fourier: # [DEBUG-1] input 同时包括fourier和grid [√]
            self.fourier_encoder = FourierEncoder(d_in=d_in, max_log_freq=N_freqs - 1, N_freqs=N_freqs, log_sampling=True)
            self.d_in = self.fourier_encoder.encoded_length() + self.d_in


        # net initialization
        self.linears = torch.nn.ModuleList()
        for l in range(1, n_layers + 2):
            out_features = self.d_hidden - (self.d_in if l in self.skip else 0)
            if l == 1:
                layer = torch.nn.Linear(self.d_in, out_features)
            elif l <= n_layers:
                layer = torch.nn.Linear(self.d_hidden, out_features)
            else:
                layer = torch.nn.Linear(self.d_hidden, self.d_hidden+1)
            # geometric initialization
            if geometric_init:  # 《SAL: Sign Agnostic Learning of Shapes from Raw Data》
                if l == n_layers + 1:  # 输出层
                    if inside_outside:  # inside
                        torch.nn.init.normal_(layer.weight, mean=-np.sqrt(np.pi) / np.sqrt(self.d_hidden), std=0.0001)
                        torch.nn.init.constant_(layer.bias, bias)  # 保证scene内中心sdf为正
                    else:  # outside
                        torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(self.d_hidden), std=0.0001)
                        torch.nn.init.constant_(layer.bias, -bias)  # 保证object内中心sdf为负
                elif l == 1:  # 第一层
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)  # 高频初始置0
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_features))
                elif l - 1 in self.skip:  # 上一层是res layer, 即该层input cat了embedding(xyz)
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_features))
                    torch.nn.init.constant_(layer.weight[:, -(self.d_in - 3):], 0.0) # 高频置0
                else: # 其他层
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_features))
            if norm_weight:
                layer = nn.utils.weight_norm(layer)
            self.linears.append(layer)

        self.activation = nn.Softplus(beta=100)
        # self.activation = nn.ReLU()

    def get_grid_params(self):
        return self.grid_encoder.parameters()

    def get_mlp_params(self):
        return self.linears.parameters()

    def get_feature_mask(self, feature):
        mask = torch.zeros_like(feature)
        if self.enable_progressive:
            mask[..., :(self.active_levels * self.per_level_dim)] = 1
        else:
            mask[...] = 1
        return mask

    def forward(self, x, if_cal_hessian_x=False):
        # if_cal_hessian_x: 传给my grid_encoder是否计算hessian
        x_enc = x
        if self.enable_fourier:
            x_enc = torch.cat([x, self.fourier_encoder(x)], dim=-1)
        if self.enable_hashgrid:
            grid_enc=self.grid_encoder(x/self.bound, if_cal_hessian_x) # [MONO-2]
            mask = self.get_feature_mask(grid_enc)
            x_enc = torch.cat([x_enc, grid_enc*mask], dim=-1)
        x=x_enc
        for l in range(1, self.n_layers + 2):
            layer = self.linears[l - 1]
            if l - 1 in self.skip:
                x = torch.cat([x, x_enc], dim=-1)
            x = layer(x)
            if l < self.n_layers + 1:
                x = self.activation(x)
        return x

    def get_sdf(self, x):
        return self.forward(x)[..., :1]

    def get_sdf_feat(self, x):
        output=self.forward(x)
        sdf, feat = output[..., :1], output[..., 1:]
        return sdf, feat

    # def gradient(self, x):
    #     if self.gradient_mode == 'analytical':
    #         x.requires_grad_(True)
    #         y = self.forward(x)[..., :1]
    #         d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    #         gradients = torch.autograd.grad(
    #             outputs=y,
    #             inputs=x,
    #             grad_outputs=d_output,
    #             create_graph=True,
    #             retain_graph=True,
    #             only_inputs=True)[0]
    #         # TODO: 应用hessian曲率loss进行场景平滑
    #
    #         return gradients
    #     elif self.gradient_mode == 'numerical':
    #         if self.taps == 6:
    #             eps = self.normal_epsilon
    #             # 1st-order gradient
    #             eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
    #             eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
    #             eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
    #             sdf_x_pos = self.get_sdf(x + eps_x).cpu()  # [...,1]
    #             sdf_x_neg = self.get_sdf(x - eps_x).cpu()  # [...,1]
    #             sdf_y_pos = self.get_sdf(x + eps_y).cpu()  # [...,1]
    #             sdf_y_neg = self.get_sdf(x - eps_y).cpu()  # [...,1]
    #             sdf_z_pos = self.get_sdf(x + eps_z).cpu()  # [...,1]
    #             sdf_z_neg = self.get_sdf(x - eps_z).cpu()  # [...,1]
    #             gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
    #             gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
    #             gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
    #             gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1).cuda()  # [...,3]
    #             # 2nd-order gradient (hessian)
    #             if self.training:
    #                 hessian=None
    #                 # assert sdf is not None  # computed when feed-forwarding through the network
    #                 # hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
    #                 # hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
    #                 # hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
    #                 # hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
    #             else:
    #                 hessian = None
    #             return gradient
    #         elif self.taps == 4:
    #             eps = self.normal_eps / np.sqrt(3)
    #             k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
    #             k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
    #             k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
    #             k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
    #             sdf1 = self.get_sdf(x + k1 * eps)  # [...,1]
    #             sdf2 = self.get_sdf(x + k2 * eps)  # [...,1]
    #             sdf3 = self.get_sdf(x + k3 * eps)  # [...,1]
    #             sdf4 = self.get_sdf(x + k4 * eps)  # [...,1]
    #             gradient = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)
    #             if self.training:
    #                 hessian=None
    #                 # assert sdf is not None  # computed when feed-forwarding through the network
    #                 # # the result of 4 taps is directly trace, but we assume they are individual components
    #                 # # so we use the same signature as 6 taps
    #                 # hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
    #                 # hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
    #             else:
    #                 hessian = None
    #             return gradient
    #         else:
    #             raise NotImplementedError

    def get_all(self, x, if_cal_hessian_x=False):
        # return sdf, feat, gradients
        # if if_cal_hessian_x: return sdf, feat, gradients, hessian
        # TODO: 注意这里的hessian,当为analytical时, hessian这里实际得到的是一个[...,3]的向量而不是[...,3,3]的矩阵，分别表示梯度三个分量对 x 的偏导数和(同理y、z)。
        x.requires_grad_(True)
        output = self.forward(x, False and self.gradient_mode == 'analytical')
        sdf, feat = output[..., :1], output[..., 1:]
        if self.gradient_mode == 'analytical':
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            if if_cal_hessian_x:
                hessian = torch.autograd.grad(outputs=gradients.sum(), inputs=x, create_graph=True)[0]
                return sdf, feat, gradients, hessian
            else:
                return sdf, feat, gradients
        elif self.gradient_mode == 'numerical':
            if self.taps == 6:
                eps = self.normal_epsilon
                # 1st-order gradient
                eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
                sdf_x_pos = self.get_sdf(x + eps_x)  # [...,1]
                sdf_x_neg = self.get_sdf(x - eps_x)  # [...,1]
                sdf_y_pos = self.get_sdf(x + eps_y)  # [...,1]
                sdf_y_neg = self.get_sdf(x - eps_y)  # [...,1]
                sdf_z_pos = self.get_sdf(x + eps_z)  # [...,1]
                sdf_z_neg = self.get_sdf(x - eps_z)  # [...,1]
                gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
                gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
                gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
                gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1) # [...,3]
                # 2nd-order gradient (hessian)
                if if_cal_hessian_x:
                    assert sdf is not None  # computed when feed-forwarding through the network
                    hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
                    return sdf, feat, gradient, hessian
                else:
                    return sdf, feat, gradient
            elif self.taps == 4:
                eps = self.normal_epsilon / np.sqrt(3)
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                sdf1 = self.get_sdf(x + k1 * eps)  # [...,1]
                sdf2 = self.get_sdf(x + k2 * eps)  # [...,1]
                sdf3 = self.get_sdf(x + k3 * eps)  # [...,1]
                sdf4 = self.get_sdf(x + k4 * eps)  # [...,1]
                gradient = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
                if if_cal_hessian_x:
                    # hessian = None
                    assert sdf is not None  # computed when feed-forwarding through the network
                    # the result of 4 taps is directly trace, but we assume they are individual components
                    # so we use the same signature as 6 taps
                    hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
                    hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
                    return sdf, feat, gradient, hessian
                else:
                    return sdf, feat, gradient
            else:
                raise NotImplementedError

    def set_active_levels(self, cur_step):
        self.anneal_levels = min(max((cur_step - self.warm_up) // self.active_step, 1), self.num_levels)
        self.active_levels = max(self.anneal_levels, self.init_active_level)

    def set_normal_epsilon(self):
        if self.enable_progressive: # normal_epsilon是grid Voxel边长的1/4
            self.normal_epsilon = 2.0 *self.bound/ (self.grid_encoder.resolution_list[self.anneal_levels - 1] - 1)/4
        else:
            self.normal_epsilon = 2.0 *self.bound / (self.grid_encoder.resolution_list[-1] - 1)/4

    def get_feature_mask(self, feature):
        mask = torch.zeros_like(feature)
        if self.enable_progressive:
            mask[..., :(self.active_levels * self.per_level_dim)] = 1
        else:
            mask[...] = 1
        return mask

# (x,v,n,z)->(color) 即idr模式
class ColorNetwork(nn.Module):
    def __init__(self,
                 feat_dim=256,  # sdf feature
                 d_hidden=256,
                 n_layers=4,
                 skip=[],
                 N_freqs=3,
                 encoding_view='spherical',  # 'fourier' 或者 'spherical'
                 weight_norm=True,
                 layer_norm=False,
                 enable_app=False,  # TODO: appearance embedding
                 app_dim=8):
        super(ColorNetwork, self).__init__()
        self.feat_dim = feat_dim
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.N_freqs = N_freqs
        self.encoding_view = encoding_view
        self.enable_app = enable_app
        self.app_dim = app_dim

        if self.encoding_view == 'fourier':
            self.view_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs - 1, N_freqs=N_freqs)
        elif self.encoding_view == 'spherical':
            self.view_encoder = SphericalEncoder(levels=N_freqs)
        else:
            raise NotImplementedError
        view_enc_dim = self.view_encoder.encoded_length()

        # build mlp
        layer_dims = [3 + 3+ view_enc_dim + 3 + feat_dim + (app_dim if enable_app else 0)] + [d_hidden] * n_layers + [3]
        self.mlp = MLPwithSkipConnection(layer_dims, skip_connection=skip, activ=nn.ReLU(), use_layernorm=layer_norm,
                                         use_weightnorm=weight_norm)

    def forward(self, x, v, n, z, app=None):
        # x: (..., 3)
        # v: (..., 3)
        # n: (..., 3)
        # z: (..., feat_dim)
        # app: (..., app_dim)
        # return: rgb(..., 3)

        # view encoding 无需cat方向自身
        view_encoding = self.view_encoder(v)
        view_encoding = torch.cat([v, view_encoding], dim=-1)
        if app is None and self.enable_app == True: # 添0
            app = torch.zeros_like(x[...,:1]).tile(*((x.ndim-1)*[1]),self.app_dim)
        input = torch.cat([x, view_encoding, n, z], dim=-1) if app is None else torch.cat([x, view_encoding, n, z, app],
                                                                                          dim=-1)
        rgb = self.mlp(input).sigmoid_()
        return rgb

class OSFNetwork(nn.Module):
    def __init__(self,
                 feat_dim=256,
                 d_hidden=256,
                 n_layers=4,
                 skip=[],
                 N_freqs=10,
                 encoding='fourier',  # 'fourier' 或者 'spherical'
                 weight_norm=True,
                 layer_norm=False):
        super().__init__()
        self.feature_dim = feat_dim
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        # encoding
        if encoding == 'fourier':
            self.encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs - 1, N_freqs=N_freqs)
        else:
            raise NotImplementedError
        # build mlp
        layer_dims=[3+self.encoder.encoded_length()+feat_dim]+[d_hidden]*n_layers+[1]
        self.mlp = MLPwithSkipConnection(layer_dims, skip_connection=skip, activ=nn.ReLU(), use_layernorm=layer_norm,
                                            use_weightnorm=weight_norm)
    def forward(self, x, z):
        # x: (..., 3)
        # z: (..., feat_dim)
        # return: 物体存在概率 osf(..., 1)
        x_encoding = torch.cat([x, self.encoder(x), z], dim=-1)
        osf = self.mlp(x_encoding).sigmoid_()
        return osf

# 对于室外的场景重建可以单独引入一个background nerf来encoding
class BGNeRF(nn.Module):
    def __init__(self,
                 d_hidden=256,
                 n_layers=8,
                 skip=[4],  # density network
                 d_hidden_rgb=128,
                 n_layers_rgb=2,
                 skip_rgb=[],
                 encoding='fourier',
                 N_freqs=10,
                 encoding_view='spherical',
                 N_freqs_view=3,
                 enable_app=False,
                 app_dim=8):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.d_hidden_rgb = d_hidden_rgb
        self.n_layers_rgb = n_layers_rgb
        self.encoding = encoding
        self.encoding_view = encoding_view
        self.enable_app=enable_app
        self.app_dim=app_dim

        if self.encoding == 'fourier':  # inverse sphere sampling即(x',y',z',1/r)，输入是4维。
            self.encoder = FourierEncoder(d_in=4, max_log_freq=N_freqs - 1, N_freqs=N_freqs)
        else:
            raise NotImplementedError

        if self.encoding_view == 'fourier':
            self.view_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs - 1, N_freqs=N_freqs)
        elif self.encoding_view == 'spherical':
            self.view_encoder = SphericalEncoder(levels=N_freqs_view)
        else:
            raise NotImplementedError

        activ_mlp = nn.ReLU()
        # density mlp
        layer_dims_density = [self.encoder.encoded_length() + 4] + [d_hidden] * (n_layers - 1) + [1 + d_hidden]
        self.density_mlp = MLPwithSkipConnection(layer_dims_density, skip, activ_mlp)
        self.activ_density = nn.Softplus()

        # rgb mlp
        # TODO: add app
        layer_dims_rgb = [d_hidden + self.view_encoder.encoded_length() + (app_dim if enable_app else 0)] + [
            d_hidden_rgb] * (n_layers_rgb - 1) + [3]
        self.rgb_mlp = MLPwithSkipConnection(layer_dims_rgb, skip_rgb, activ_mlp)

    def forward(self, x, v, app=None):
        # x: (..., 3)
        # v: (..., 3)
        # app: (..., app_dim)

        # encode三维点, nerf++式的背景重参数化
        dist = torch.norm(x, dim=-1, keepdim=True)
        norm_x = torch.cat([x / dist, 1 / dist], dim=-1)
        x_encoding = torch.cat([norm_x, self.encoder(norm_x)], dim=-1)  # （..., 4+encoded_length）
        view_encoding = self.view_encoder(v)

        density_output = self.density_mlp(x_encoding)
        sigma, feat = density_output[..., :1].sigmoid_(), density_output[..., 1:]

        rgb_input = torch.cat([feat, view_encoding, app], dim=-1) if app is not None else torch.cat(
            [feat, view_encoding], dim=-1)
        rgb = self.rgb_mlp(rgb_input).sigmoid_()
        return sigma, rgb

# test
# sdf_model = SDFNetwork().cuda()
# x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).cuda()
# z = torch.tensor([[1, 2, 3]])
# grad = sdf_model.gradient(x)
# print(grad)




# color_model = ColorNetwork().cuda()
# x = torch.tensor([[1, 2, 3]], dtype=torch.float32).cuda()
# v = torch.tensor([[1, 2, 3]], dtype=torch.float32).cuda()
# v = F.normalize(v, dim=-1)
# n = torch.tensor([[1, 2, 3]], dtype=torch.float32).cuda()
# # z feature: 256维度
# z = torch.randn(1, 256).cuda()
#
# color = color_model(x, v, n, z)
# print(color)
