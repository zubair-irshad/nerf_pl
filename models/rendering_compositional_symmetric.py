import torch
from einops import rearrange, reduce, repeat
from typing import List, Dict, Any, Optional
from models.nerf import ObjectNeRF
import torch.nn.functional as F
import torch.nn as nn

# __all__ = ['render_rays']
__all__ = ["render_rays", "sample_pdf"]

def apply_symmetry_transform(rays_o, rays_d, reflection_dim):
    """ Applies symmetry transformation to a batch of 3D points.
    Args:
        x: point cloud with shape (..., 3)
    Returns:
        transformed point cloud with shape (..., 3)
    """
    # rays_o = torch.FloatTensor(rays_o)
    # rays_d = torch.FloatTensor(rays_d)
    canonical_symmetry_matrix = torch.eye(4).to(rays_o.device)
    canonical_symmetry_matrix[reflection_dim, reflection_dim] *= -1
    canonical_symmetry_matrix = canonical_symmetry_matrix.unsqueeze(0) # add batch dim

    rays_o = F.pad(rays_o, (0, 1), mode='constant', value=0.0) # homogenise points
    rays_d = F.pad(rays_d, (0, 1), mode='constant', value=1.0) # homogenise points

    rays_o = torch.einsum('...ij,...j->...i', canonical_symmetry_matrix, rays_o)[..., :3]
    rays_d = torch.einsum('...ij,...j->...i', canonical_symmetry_matrix, rays_d)[..., :3]
    return rays_o, rays_d


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

def inference_model(
    results: Dict[str, Any], 
    model: ObjectNeRF, 
    embeddings: Dict[str, Any],
    typ: str, 
    xyz: torch.Tensor, 
    rays_d: torch.Tensor, 
    xyz_sym : torch.Tensor,
    rays_d_sym : torch.Tensor,
    z_vals, 
    chunk: int,
    noise_std: float,
    white_back: bool,
    is_eval: bool = False,
    use_zero_as_last_delta: bool = False,
    forward_instance: bool = True,
    embedding_instance: Optional[torch.Tensor] = None,
    frustum_bound_th: float = 0,
    pass_through_mask: Optional[torch.Tensor] = None,
    rays_in_bbox: bool = False,
    **dummy_kwargs
):
    """
    Helper function that performs model inference.
    Inputs:
        results: a dict storing all results
        model: NeRF model (coarse or fine)
        typ: 'coarse' or 'fine'
        xyz: (N_rays, N_samples_, 3) sampled positions
                N_samples_ is the number of sampled points in each ray;
                            = N_samples for coarse model
                            = N_samples+N_importance for fine model
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        test_time: test time or not
    Outputs:
        if weights_only:
            weights: (N_rays, N_samples_): weights of each sample
        else:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
    """
    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]
    N_rays, N_samples_, _ = xyz.shape
    xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)
    xyz_sym_ = rearrange(xyz_sym, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

    # Embed direction
    dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)
    dir_embedded_ = repeat(
        dir_embedded, "n1 1 c -> (n1 n2) c", n2=N_samples_
    )  # (N_rays*N_samples_, embed_dir_channels)

    dir_embedded_sym = embedding_dir(rays_d_sym)  # (N_rays, embed_dir_channels)
    dir_embedded_sym_ = repeat(
        dir_embedded_sym, "n1 1 c -> (n1 n2) c", n2=N_samples_
    )  # (N_rays*N_samples_, embed_dir_channels)

    obj_codes = repeat(embedding_instance, "n1 c -> (n1 n2) c", n2=N_samples_)
    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []
    instance_sigma_chunk = []
    instance_rgb_chunk = []

    instance_sigma_sym_chunk = []
    instance_rgb_sym_chunk = []
    
    sigma_chunks = []
    rgb_chunks = []

    for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_[i : i + chunk])
        xyz_embedded_sym = embedding_xyz(xyz_sym_[i : i + chunk])

        output = model(
            {
                "emb_xyz": xyz_embedded,
                "emb_dir": dir_embedded_[i : i + chunk],
            }
        )
        rgb_chunks += [output["rgb"]]
        sigma_chunks += [output["sigma"]]
        if forward_instance:
            print("obj_codes", obj_codes.shape, xyz_embedded.shape)
            inst_output = model.forward_instance(
                {
                    "emb_xyz": xyz_embedded,
                    "emb_dir": dir_embedded_[i : i + chunk],
                    "obj_code": obj_codes[i : i + chunk],
                }
            )

            inst_output_sym = model.forward_instance(
                {
                    "emb_xyz": xyz_embedded_sym,
                    "emb_dir": dir_embedded_sym_[i : i + chunk],
                    "obj_code": obj_codes[i : i + chunk],
                }
            )

            instance_sigma_chunk += [inst_output["inst_sigma"]]
            instance_rgb_chunk += [inst_output["inst_rgb"]]

            instance_sigma_sym_chunk += [inst_output_sym["inst_sigma"]]
            instance_rgb_sym_chunk += [inst_output_sym["inst_rgb"]]

    sigmas = torch.cat(sigma_chunks, 0).view(N_rays, N_samples_)
    rgbs = torch.cat(rgb_chunks, 0).view(N_rays, N_samples_, 3)

    if forward_instance:
        instance_sigma = torch.cat(instance_sigma_chunk, 0).view(N_rays, N_samples_)
        instance_rgb = torch.cat(instance_rgb_chunk, 0).view(N_rays, N_samples_, 3)
        #print("instance_sigma", instance_sigma.shape, instance_rgb.shape)
        instance_sigma_sym = torch.cat(instance_sigma_sym_chunk, 0).view(N_rays, N_samples_)
        instance_rgb_sym = torch.cat(instance_rgb_sym_chunk, 0).view(N_rays, N_samples_, 3)
        #print("instance_sigma_sym", instance_sigma_sym.shape, instance_rgb_sym.shape)
    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    delta_zero = torch.zeros_like(deltas[:, :1])
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        # for instance, we always set last delta to zero
    if forward_instance:
        deltas_instance = torch.cat([deltas, delta_zero], -1)  # (N_rays, N_samples_)

    if use_zero_as_last_delta:
        deltas = torch.cat([deltas, delta_zero], -1)  # (N_rays, N_samples_)
    else:
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # compute alpha by the formula (3)
    noise = torch.randn_like(sigmas) * noise_std
    alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
    weights = \
        alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
    weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    results[f'weights_{typ}'] = weights
    results[f'opacity_{typ}'] = weights_sum
    results[f'z_vals_{typ}'] = z_vals

    rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
    depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

    if white_back:
        rgb_map += 1-weights_sum.unsqueeze(1)

    results[f'rgb_{typ}'] = rgb_map
    results[f'depth_{typ}'] = depth_map
    # if is_eval and forward_instance:
    if forward_instance:
        # apply to sigma
        noise_instance = torch.randn_like(instance_sigma) * noise_std
        alphas_instance = 1 - torch.exp(
            -deltas_instance * torch.relu(instance_sigma + noise_instance)
        )  # (N_rays, N_samples_)
        # mask out occlusion
        if not is_eval and frustum_bound_th > 0:
            # mask true -> area farther than the rendered scene depth
            occlusion_mask = (
                results[f"depth_{typ}"].unsqueeze(1).expand(N_rays, N_samples_)
                + frustum_bound_th
            ) < z_vals
            if pass_through_mask is not None:
                # for visible instance area, we do not suppress the alphas
                occlusion_mask[pass_through_mask.expand(N_rays, N_samples_)] = False
            # mask out object alphas farther than the rendered scene depth
            alphas_instance[occlusion_mask] = 0
        alphas_shifted_instance = torch.cat(
            [torch.ones_like(alphas_instance[:, :1]), 1 - alphas_instance + 1e-10],
            -1,
        )  # [1, 1-a1, 1-a2, ...]
        weights_instance = alphas_instance * torch.cumprod(
            alphas_shifted_instance[:, :-1], -1
        )  # (N_rays, N_samples_)

        weights_sum_instance = reduce(weights_instance, "n1 n2 -> n1", "sum")

        # compute instance rgb and depth
        rgb_instance_map = reduce(
            rearrange(weights_instance, "n1 n2 -> n1 n2 1") * instance_rgb,
            "n1 n2 c -> n1 c",
            "sum",
        )
        # weights_instance, 'n1 n2 -> n1 n2 1')*rgbs.detach(), 'n1 n2 c -> n1 c', 'sum')
        depth_instance_map = reduce(weights_instance * z_vals, "n1 n2 -> n1", "sum")
        # if white_back:'
        # always white back for object
        rgb_instance_map = rgb_instance_map + 1 - weights_sum_instance.unsqueeze(-1)
        results[f"rgb_instance_{typ}"] = rgb_instance_map
        results[f"depth_instance_{typ}"] = depth_instance_map
        results[f"opacity_instance_{typ}"] = weights_sum_instance

        results[f"instance_out_sym_{typ}"] = torch.cat((instance_sigma_sym.unsqueeze(-1), instance_rgb_sym), dim=-1)
        results[f"instance_out_{typ}"] = torch.cat((instance_sigma.unsqueeze(-1), instance_rgb), dim=-1)

        results[f"instance_out_sym_{typ}"] = instance_rgb_sym
        results[f"instance_out_{typ}"] = instance_rgb
        
        if rays_in_bbox:  # for pdf sampling, overwrite weights
            results[f"weights_{typ}"] = weights_instance
    return

def render_rays(models: Dict[str, ObjectNeRF],
                embeddings: Dict[str, Any],
                rays: torch.Tensor,
                N_samples: int = 64,
                use_disp: bool = False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                forward_instance: bool = True,
                embedding_instance: Optional[torch.Tensor] = None,
                frustum_bound_th: float = -1,
                pass_through_mask: Optional[torch.Tensor] = None,
                rays_in_bbox: bool = False,
                test_time=False,
                **dummy_kwargs,
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    rays_o_sym, rays_d_sym = apply_symmetry_transform(rays_o, rays_d, 2)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    rays_o_sym = rearrange(rays_o_sym, 'n1 c -> n1 1 c')
    rays_d_sym = rearrange(rays_d_sym, 'n1 c -> n1 1 c')

    # mae = nn.L1Loss()

    # print("mae", mae(rays_o, rays_o_sym))
    # print("mae dir", mae(rays_d, rays_d_sym))


    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    xyz_coarse_symm = rays_o_sym + rays_d_sym * rearrange(z_vals, 'n1 n2 -> n1 n2 1')


    # mae = nn.L1Loss()
    # print("mae xyz", mae(xyz_coarse, xyz_coarse_symm))

    results = {}
    inference_model(
        results=results,
        model=models["coarse"],
        embeddings=embeddings,
        typ="coarse",
        xyz=xyz_coarse,
        rays_d=rays_d,
        xyz_sym = xyz_coarse_symm,
        rays_d_sym = rays_d_sym,
        z_vals=z_vals,
        chunk=chunk,
        noise_std=noise_std,
        white_back=white_back,
        forward_instance=forward_instance,
        embedding_instance=embedding_instance,
        frustum_bound_th=frustum_bound_th,
        pass_through_mask=pass_through_mask,
        rays_in_bbox=rays_in_bbox,
        **dummy_kwargs,
    )
    #inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference_model(
            results=results,
            model=models["fine"],
            embeddings=embeddings,
            typ="fine",
            xyz=xyz_fine,
            rays_d=rays_d,
            z_vals=z_vals,
            chunk=chunk,
            noise_std=noise_std,
            white_back=white_back,
            forward_instance=forward_instance,
            embedding_instance=embedding_instance,
            frustum_bound_th=frustum_bound_th,
            pass_through_mask=pass_through_mask,
            rays_in_bbox=rays_in_bbox,
            **dummy_kwargs,
        )
        #inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, **kwargs)
    return results