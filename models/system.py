import torch
from models.fields import *
from models.ray_sampler import *
from models.nerf_util import *

class ImplicitReconSystem(torch.nn.Module):
    def __init__(self, conf, bound=None, device='cuda:0'):
        super().__init__()
        self.conf = conf
        self.bound = conf.model.bound if bound is None else bound
        self.device = device
        self.bg_enabled = conf.model.background.enabled
        self.osf_enabled = conf.model.object.osf.enabled
        self.white_bg = conf.model.white_bg # 是否使用白色背景

        density_conf = conf.model.density
        sdf_conf = conf.model.object.sdf
        rgb_conf = conf.model.object.rgb
        osf_conf = conf.model.object.osf
        bg_conf = conf.model.background
        sampler_conf = conf.model.sampler
        optim_conf = conf.optim

        self.lr = optim_conf.lr
        # initialize models
        self.density = LaplaceDensity(**density_conf)
        self.sdf = SDFNetwork(**sdf_conf,bound=self.bound)
        self.rgb = ColorNetwork(**rgb_conf)
        if osf_conf.enabled: # osf
            self.osf = OSFNetwork(**osf_conf.mlp)
        if bg_conf.enabled:  # 启用background nerf，即在单位球外采样一些点，然后用传统nerf建模辐射场。
            self.bg_nerf = BGNeRF(**bg_conf.nerf)
            if bg_conf.nerf.enable_app:
                self.app_bg = nn.Embedding(1000, bg_conf.nerf.app_dim)
                std=1e-4
                self.app_bg.weight.data.uniform_(-std, std)
        if rgb_conf.enable_app: # 启用appearance field
            self.app = nn.Embedding(1000, rgb_conf.app_dim)
            std=1e-4
            self.app.weight.data.uniform_(-std, std)
        # sampler
        self.sampler = ErrorBoundSampler(**sampler_conf,scene_bounding_sphere=self.bound)

    def forward(self, sample):
        # shape format: (B,R,N,D)
        # indices: (B ), rays_o: (B, R, 3), rays_d: (B, R, 3), depth_scale: (B, R, 1)
        indices, rays_o, rays_d, depth_scale = sample['idx'], sample['rays_o'], sample['rays_d'], sample['depth_scale']
        output={}

        # sample
        B, R = rays_o.shape[:2]
        rays_o, rays_d= rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        # 这个z_near是每条ray只从z（表面附近）随机采样了一个点。
        # TODO: 这里sampler启用osf应在全局场景学习收敛并且osf训练充分后才能启用，否则会导致训练不稳定。
        z, z_near, near, far, outside = self.sampler.get_z_vals(ray_dirs=rays_d, cam_loc=rays_o, model=self, osf_enabled=False) # z: (B*R,N), z_eik:(B*R,1), near:(B*R,1), far:(B*R,1), outside:(B*R,1)
        if outside.any():
            print('there are rays outside the scene')
        rays_o, rays_d = rays_o.reshape(B, R, 3), rays_d.reshape(B, R, 3)

        # object/scene渲染, 基于sdf
        z = z.reshape(B, R, -1, 1)
        points = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z
        sdf, feat, gradient = self.sdf.get_all(points)
        # TODO: osf应当在场景学习收敛后再启用
        if self.osf_enabled and False:
            osf = self.osf(points,feat)
        densities = self.density(sdf)
        app = None if not self.rgb.enable_app else self.app(indices)[:, None, None, :].tile(1, R, z.shape[2], 1)
        rgbs = self.rgb(points, rays_d[:, :, None, :].tile(1, 1, z.shape[2], 1), gradient/torch.norm(gradient,dim=-1,keepdim=True),feat, app)
        alphas = volume_rendering_alphas(densities=densities,dists=z)


        # background nerf rendering
        if self.bg_enabled:
            inverse_r = self.sampler.uniform_sampler.sample_dists(ray_size=(B, R), dist_range=(1, 0), intvs=self.sampler.N_samples_bg, stratified=self.training) # (B, R, N_samples_bg, 1)
            z_bg = far.reshape(B, R, 1, 1)/inverse_r # (B, R, N_samples_bg, 1)
            points_bg = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_bg # (B, R, N_samples_bg, 3)
            app_bg = None if not self.bg_nerf.enable_app else self.app_bg(indices)[:, None, None, :].tile(1, R, z_bg.shape[2], 1)
            densities_bg, rgbs_bg  = self.bg_nerf(points_bg, rays_d[:, :, None, :].tile(1, 1, self.sampler.N_samples_bg, 1), app_bg)
            alphas_bg = volume_rendering_alphas(densities=densities_bg,dists=z_bg) # 不透明度, 为了方便合并object和bg的结果, 我们使用基于alpha、T的离散体渲染。

            # merge
            alphas= torch.cat([alphas,alphas_bg],dim=2)
            rgbs = torch.cat([rgbs,rgbs_bg],dim=2)
            z = torch.cat([z,z_bg],dim=2)

        # compositing
        weights=alpha_compositing_weights(alphas)
        rgb = composite(rgbs, weights)
        if self.white_bg: # 白色背景
            opacity = composite(1, weights)
            rgb = rgb + (1 - opacity)
        gradient_normalized = gradient/(torch.norm(gradient,dim=-1,keepdim=True)+1e-6) # [DEBUG-5] ！先归一化gradient再计算ray normal, 否则不一致。
        if self.bg_enabled: # 体渲染场景normal
            normal = composite(gradient_normalized, weights[...,:gradient.shape[-2],:]/torch.sum(weights[...,:gradient.shape[-2],:],dim=-2,keepdim=True))
        else:
            normal = composite(gradient_normalized,weights)
        dist = composite(z, weights) # 体渲染距离
        depth=dist*depth_scale # depth
        if self.osf_enabled and False: # 体渲染osf得到的mask
            if self.bg_enabled:
                object_prob = composite(osf, weights[...,:osf.shape[-2],:]/torch.sum(weights[...,:osf.shape[-2],:],dim=-2,keepdim=True))
            else:
                object_prob = composite(osf, weights)

        # normal 世界坐标系->相机坐标系
        R_w2c=torch.transpose(sample['pose'][:,:3,:3],1,2)
        normal=torch.bmm(R_w2c,torch.transpose(normal,1,2))
        normal = torch.transpose(normal,1,2)

        # output
        output['outside'] = outside.reshape(B, R, 1)
        output['rgb'] = rgb
        output['depth'] = depth
        output['normal'] = normal
        output['gradient'] = gradient
        output['sdf'] = sdf
        if self.osf_enabled and False:
            output['osf'] = osf
            output['object_prob'] = object_prob

        # 这里采样eikonal points、curvature points等
        # TODO: 是否引入sparse points, 优化近邻曲率，instant-angelo做法。
        if self.training:
            volsdf_type=True # [MONO-6]
            if volsdf_type: # volsdf的采样方法：稀疏eik点，包括了z中的抽样点和场景均匀采样点
                z_near=z_near.reshape(B,R,-1,1)
                points_near = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_near # (B, R, N_near, 3)
                if self.conf.model.sampler.intersection_type == 'cube':
                    points_uniform=sample_points_in_cube(side_length=2.*self.bound, shape=z_near.shape[:-2]+(1,),device=self.device)
                elif self.conf.model.sampler.intersection_type == 'sphere':
                    points_uniform=sample_points_in_sphere(radius=self.bound, shape=z_near.shape[:-2]+(1,),device=self.device)
                else:
                    raise NotImplementedError
                points_eik=torch.cat([points_near,points_uniform],dim=-2)
                sdf_eik, _, gradient_eik = self.sdf.get_all(points_eik)
                # 1. 拓展：基于gradient_eik在切点切平面上采样neighbor points，以计算曲率
                points_eik_neighbor = sample_neighbours_near_plane(gradient_eik, points_eik, device=self.device)
                # 2. 在附近随机采样
                # points_eik_neighbor = points_eik + (torch.rand_like(points_eik) - 0.5) * 0.01
                sdf_eik_neighbor, _, gradient_eik_neighbor = self.sdf.get_all(points_eik_neighbor)

                output['gradient_eik'] = gradient_eik.reshape(-1,3)
                output['gradient_eik_neighbor'] = gradient_eik_neighbor.reshape(-1,3)

                # hessian
                _,_,_,hessian_near=self.sdf.get_all(points_near,if_cal_hessian_x=True)
                output['hessian']=hessian_near

                # output['hessian'] = hessian
            else: # neural-angelo选择所有点。
                output['gradient_eik'] = gradient
                output['gradient_eik_neighbor'] = None
                output['hessian'] = hessian


        # TODO: 引入mvs的稠密点云信息（mvs也可以考虑deep-mvs）：
        # 1. 基于patch match的稠密点云带深度和法向信息，可以用来监督体渲染depth和normal；
        # 2. 同时结合osf来筛选可用mvs点云（osf>0.5）。

        return output