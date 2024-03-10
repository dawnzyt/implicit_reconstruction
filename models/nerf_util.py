import cv2
import numpy as np
import torch
import torch.nn.functional as torch_F
from torch.cuda.amp import autocast



# adapted from official implementing of neuralangelo
class MLPwithSkipConnection(torch.nn.Module):

    def __init__(self, layer_dims, skip_connection=[], activ=None, use_layernorm=False, use_weightnorm=False):
        """Initialize a multi-layer perceptron with skip connection.
        Args:
            layer_dims: A list of integers representing the number of channels in each layer.
            skip_connection: A list of integers representing the index of layers to add skip connection.
        """
        super().__init__()
        self.skip_connection = skip_connection
        self.use_layernorm = use_layernorm
        self.linears = torch.nn.ModuleList()
        if use_layernorm:
            self.layer_norm = torch.nn.ModuleList()
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            if li in self.skip_connection:
                k_in += layer_dims[0]
            linear = torch.nn.Linear(k_in, k_out)
            if use_weightnorm:
                linear = torch.nn.utils.weight_norm(linear)
            self.linears.append(linear)
            if use_layernorm and li != len(layer_dim_pairs) - 1:
                self.layer_norm.append(torch.nn.LayerNorm(k_out))
            if li == len(layer_dim_pairs) - 1:
                self.linears[-1].bias.data.fill_(0.0)
        self.activ = activ or torch_F.relu_

    def forward(self, input):
        feat = input
        for li, linear in enumerate(self.linears):
            if li in self.skip_connection:
                feat = torch.cat([feat, input], dim=-1)
            feat = linear(feat)
            if li != len(self.linears) - 1:
                if self.use_layernorm:
                    feat = self.layer_norm[li](feat)
                feat = self.activ(feat)
        return feat


def volume_rendering_alphas(densities, dists, dist_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        densities (tensor [batch,ray,samples,1]): The predicted volume density samples.
        dists (tensor [batch,ray,samples,1]): The corresponding distance samples.
        dist_far (tensor [batch,ray,1,1]): The farthest distance for computing the last interval.
    Returns:
        alphas (tensor [batch,ray,samples,1]): The occupancy of each sampled point along the ray (in [0,1]).
    """
    if dist_far is None:
        dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
    dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, :] - dists[..., :-1, :]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp_()  # [B,R,N]
    return alphas[..., 0]


def alpha_compositing_weights(alphas):
    """Alpha compositing of (sampled) MPIs given their RGBs and alphas.
    Args:
        alphas (tensor [batch,ray,samples]): The predicted opacity values.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each MPI (in [0,1]).
    """
    alphas_front = torch.cat([torch.zeros_like(alphas[..., :1]),
                              alphas[..., :-1]], dim=2)  # [B,R,N]
    with autocast(enabled=False):  # TODO: may be unstable in some cases.
        visibility = (1 - alphas_front).cumprod(dim=2)  # [B,R,N]
    weights = (alphas * visibility)[..., None]  # [B,R,N,1]
    return weights


def composite(quantities, weights):
    """Composite the samples to render the RGB/depth/opacity of the corresponding pixels.
    Args:
        quantities (tensor [batch,ray,samples,k]): The quantity to be weighted summed.
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray.
    Returns:
        quantity (tensor [batch,ray,k]): The expected (rendered) quantity.
    """
    # Integrate RGB and depth weighted by probability.
    quantity = (quantities * weights).sum(dim=2)  # [B,R,K]
    return quantity


def sample_points_in_sphere(radius, shape, device='cpu'):
    u = torch.empty(shape, device=device).uniform_(0, 1)
    theta = torch.empty(shape, device=device).uniform_(0, 2 * torch.pi)
    phi = torch.empty(shape, device=device).uniform_(0, torch.pi)

    # 立方根运算对应了球体体积增长的速率，因此使用 torch.pow(u, 1/3) 可以确保生成的点在球体内部均匀分布。
    x = radius * torch.pow(u, 1 / 3) * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.pow(u, 1 / 3) * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.pow(u, 1 / 3) * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1)
    return points


def sample_points_in_cube(side_length, shape, device='cpu'):
    # side_length: 边长
    # shape: 采样点数量
    points = torch.empty(list(shape)+[3], device=device).uniform_(-side_length / 2, side_length / 2)
    return points


def sample_neighbours_near_plane(n, p, device='cpu'):
    # n: (..., 3), p: (..., 3)
    # 在与法向垂直的切平面上采样邻近点
    pre_shape=p.shape[:-1]
    n,p=n.reshape(-1,3),p.reshape(-1,3)
    v = torch.randn_like(n, device=device)
    perpendicular = torch.cross(n, v,dim=1)  # 生成与n垂直的随机向量
    perpendicular = perpendicular / perpendicular.norm(dim=-1, keepdim=True)
    # 切点邻近点
    dist = torch.rand_like(n[:, :1], device=device) * 0.01
    neighbours = p + dist * perpendicular
    neighbours=neighbours.reshape(pre_shape+(3,))
    return neighbours

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose