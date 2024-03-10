# adapted from monosdf/models/ray_sampler.py

import abc
import torch


class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self, near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass


class UniformSampler(RaySampler):
    def __init__(self,
                 scene_bounding_sphere=1,
                 near=0,
                 far=-1,
                 N_samples=64,
                 take_intersection=False,
                 intersection_type='cube'):
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75 if far == -1 else far)
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_intersection = take_intersection
        self.intersection_type = intersection_type

    def near_far_from_sphere(self, rays_o, rays_d, bound):
        # sphere的bound指半径
        # intersect with sphere
        o_projection = torch.sum(rays_o * rays_d, dim=-1)
        h_square = torch.sum(rays_o * rays_o, dim=-1) - o_projection ** 2
        # (h_square - bound ** 2) >= 0, means no intersection or tangent
        outside = h_square - bound ** 2 >= 0

        far = torch.sqrt(bound ** 2 - h_square) - o_projection
        near = -torch.sqrt(bound ** 2 - h_square) - o_projection
        # restrict near to a minimal value
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far, outside

    def near_far_from_cube(self, rays_o, rays_d, bound):
        # cube的bound指半边长
        # intersect with cube
        tmin = (-bound - rays_o) / (rays_d + 1e-15)  # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection seg, set both near and far to inf (1e9 here)
        outside = far <= near
        near[outside] = 1e9
        far[outside] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far, outside

    # currently this is used for replica scannet and T&T
    def get_z_vals(self, ray_dirs, cam_loc, model):
        if not self.take_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0],
                                                                                                   1).cuda()
        else:
            if self.intersection_type == 'cube':
                _, far, outside = self.near_far_from_cube(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            elif self.intersection_type == 'sphere':
                _, far, outside = self.near_far_from_sphere(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals, near, far, outside

    @staticmethod
    def sample_dists(ray_size, dist_range, intvs, stratified, device="cuda"):
        """Sample points on ray shooting from pixels using distance.
        Args:
            ray_size (int [2]): Integers for [batch size, number of rays].
            range (float [2]): Range of distance (depth) [min, max] to be sampled on rays.
            intvs: (int): Number of points sampled on a ray.
            stratified: (bool): Use stratified sampling or constant 0.5 sampling.
        Returns:
            dists (tensor [batch_size, num_ray, intvs, 1]): Sampled distance for all rays in a batch.
        """
        batch_size, num_rays = ray_size
        dist_min, dist_max = dist_range
        if stratified:
            rands = torch.rand(batch_size, num_rays, intvs, 1, device=device)
        else:
            rands = torch.empty(batch_size, num_rays, intvs, 1, device=device).fill_(0.5)
        rands += torch.arange(intvs, dtype=torch.float, device=device)[None, None, :, None]  # [B,R,N,1]
        dists = rands / intvs * (dist_max - dist_min) + dist_min  # [B,R,N,1]
        return dists
class ErrorBoundSampler(RaySampler):
    def __init__(self,
                 scene_bounding_sphere=1,
                 near=0,
                 N_samples=64,  # 有效采样点数
                 N_samples_eval=128,  # 无偏估计评估每次迭代增加的采样点数
                 N_samples_extra=32,  # 无偏估计抽样点数
                 N_samples_bg=32,  # 背景nerf采样点数
                 N_near=4, # 表面附近采样点数
                 eps=0.1,  # opacity error上限
                 beta_iters=10,  # 二分次数
                 max_total_iters=5,  # 最大迭代次数
                 take_intersection=True,
                 intersection_type='cube'):
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75)

        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.N_samples_extra = N_samples_extra
        self.N_samples_bg = N_samples_bg
        self.N_near = N_near
        self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples=N_samples_eval,take_intersection=take_intersection,
                                              intersection_type=intersection_type)  # replica scannet and T&T courtroom

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere

    def get_z_vals(self, ray_dirs, cam_loc, model, osf_enabled=False):
        # ray_dirs: (N, 3)
        # return
        # N_samples个无偏估计逆采样点+N_extra个无偏估计样本中随机取的点
        # 1个从上面这些点中随机取的eikonal point
        # outside: (N, ) 1表示光线不与边界相交，0表示相交

        # osf：论文《H 2 O-SDF: TWO-PHASE LEARNING FOR 3D INDOOR RECONSTRUCTION USING OBJECT SURFACE FIELDS》提出
        # 空间概率场， 通过空间概率筛选该光路，来缓解volume-sdf导致的错误weight多峰问题。
        # 这里让weight=weight*osf, bound_error=bound_error*osf
        beta0 = model.density.get_beta().detach()

        # Start with uniform sampling
        z_vals, near, far, outside = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        # samples是每次迭代的新采样点
        # samples_idx是[z,samples]的sort后的index
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)  # beta_plus: opacity error最大上限<=eps时的beta

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            # 只计算new samples的sdf
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)

            # Calculating the SDF only for the new sampled points
            with torch.no_grad():
                samples_sdf, samples_feat = model.sdf.get_sdf_feat(points_flat)
                if osf_enabled:
                    samples_osf = model.osf(points_flat, samples_feat)
            if samples_idx is not None:  # merge new samples with old samples
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
                if osf_enabled:
                    osf_merge = torch.cat([osf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                           samples_osf.reshape(-1, samples.shape[1])], -1)
                    osf = torch.gather(osf_merge, 1, samples_idx)
            else:
                sdf = samples_sdf
                if osf_enabled:
                    osf = samples_osf.reshape(z_vals.shape)

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            min_d_star, max_d_star = d_star.min(), d_star.max()
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign：光线穿过surface的d star=0

            # 维护beta, s.t. 一定满足opacity error bound <= eps
            # Updating beta using line search
            # 加速操作: 可以直接缩小beta->beta0
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists,
                                              d_star)  # 计算缩小beta即beta0时得到的：当前每条ray的Opacity error上限
            beta[curr_error <= self.eps] = beta0  # 满足beta->beta0
            # 批处理二分:
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                # beta 增 -> error 增
                # error 小 R = mid
                # error 大 L = mid
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            # calculate the transmittance and weights
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))
            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0  # 收敛指beta尽可能小, 即结构解耦度高， <beta0, 虽 <eps但beta大认为不收敛

            if not_converge and total_iters < self.max_total_iters:
                ''' Sample more points proportional to the current error bound'''
                # d(cdf) = seg opacity error bound, 越大需要sample更多点。
                N = self.N_samples_eval

                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:, :-1] ** 2.) / (
                        4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * transmittance[:,
                                                                                           :-1]  # bound= (exp(error_integral)-1)*transmittance

                pdf = bound_opacity + 1e-6
                if osf_enabled:
                    pdf = pdf * (osf[:, :-1] + osf[:, 1:]) / 2
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # 未收敛按每段opacity error bound比例采样

            else:
                ''' Sample the final sample set to be used in the volume rendering integral '''
                # d(cdf) = weights = seg transmittance * opacity, 即coarse -> fine inverse sampling, 越大需要sample更多点。
                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-6  # prevent nans
                if osf_enabled:
                    pdf = pdf * (osf[:, :-1] + osf[:, 1:]) / 2
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # opacity error收敛后按体渲染权重比例采样

            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (
                    not model.training):  # 未收敛或eval时new samples按cdf比例采样
                u = torch.linspace(0., 1., steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
            else:  # train过程中收敛时random sample再按cdf采样增加随机性
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            u = u.contiguous()

            # inverse sampling from cdf
            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        # 最后的samples是bound error尽可能收敛(beta尽可能小)的光路估计，from weights的fine samples
        z_samples = samples
        # TODO Use near and far from intersection
        # near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0],
        #                                                                                        1).cuda()

        if self.N_samples_extra > 0:  # extra指从算法估计光路的采样点z_vals中再采样一些点
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # TODO: 这里是否应该只抽样z_samples得到表面附近点？
        # add some of the near surface points
        # eikonal point, 每个光路中随机抽样一点
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0], self.N_near)).cuda()
        z_samples_near = torch.gather(z_vals, 1, idx)

        return z_vals, z_samples_near, near, far, outside

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        # calculate max opacity error bound of each ray
        # bound = (exp(E)-1) exp(-R)
        density = model.density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]
