import torch
import numpy as np
import pdb
import torch.nn as nn

def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame

    # pdb.set_trace()
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):

    with torch.no_grad():
        rays_o_all = torch.zeros(6, H, W, 3)
        rays_d_all = torch.zeros(6, H, W, 3)

        for i in range (6):
            # pdb.set_trace()
            rays_o, rays_d = get_rays(H, W, K[0, i, ...], c2w[0, i, ...], inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
            rays_o_all[i,...] = rays_o
            rays_d_all[i,...] = rays_d


    return rays_o_all, rays_d_all


def cumprod_exclusive(p):

    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)

def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)  #  accumulated transmittance
    weights = alpha * alphainv_cum[..., :-1]  # alpha*accumulated transmittance =  weights

    return weights, alphainv_cum


def sample_ray(self, rays_o, rays_d, near, far, stepsize, xyz_min, xyz_max, voxel_size, is_train=False):
    '''Sample query points on rays'''

    N_samples = int(
        np.linalg.norm(
            np.array(self.density.get_dense_grid().shape[2:]) + 1) / stepsize) + 1

    # 2. determine the two end-points of ray bbox intersection
    vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)

    rate_a = (xyz_max - rays_o) / vec
    rate_b = (xyz_min - rays_o) / vec

    t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
    t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)

    # 3. check whether a raw intersect the bbox or not
    mask_outbbox = (t_max <= t_min)

    # 4. sample points on each ray
    rng = torch.arange(N_samples)[None].float()
    if is_train == 'train':
        rng = rng.repeat(rays_d.shape[-2], 1)
        rng += torch.rand_like(rng[:, [0]])  # add some noise to the sample

    step = stepsize * voxel_size * rng
    interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

    # 5. update mask for query points outside bbox
    mask_outbbox = mask_outbbox[..., None] | ((xyz_min > rays_pts) | (rays_pts > xyz_max)).any(dim=-1)

    return rays_pts, mask_outbbox




class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


# params_init = {'beta':0.1}, beta_min=0.0001
class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={'beta': 1.0}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()
            # print('---------------------------------beta:', beta)

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))


    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
