import torch
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from models.rendering import *
from models.nerf import *
import metrics
from datasets import dataset_dict
from datasets.llff import *
from torch.utils.data import DataLoader
from functools import partial
from datasets.srn_multi_ae import collate_lambda_train, collate_lambda_val
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
from jupyter_utils import *

dataset = dataset_dict['pd_multi_obj_ae_nocs']
root_dir = '/home/zubairirshad/nerf_pl/PD_single_scene_nocs'
img_wh = (320, 240)

kwargs = {'root_dir': root_dir,
          'img_wh': tuple(img_wh),
         'model_type': 'nerfpp',
         'split': 'val'}

train_dataset = dataset(**kwargs)
dataloader =  DataLoader(train_dataset,
                  shuffle=False,
                  num_workers=0,
                  batch_size=1,
                  pin_memory=False)

print("len train dataset", len(train_dataset))
for i, data in enumerate(dataloader):
    print("i", i)
    for k,v in data.items():
        print(k,v.squeeze(0).shape)
    if i>0:
        break

    print("===============================\n\n\n")

for k,v in data.items():
    data[k] = v.squeeze(0)
    
nerfplusplus = False
contract = True

if nerfplusplus:
    import models.nerfplusplus.helper as helper
    from torch import linalg as LA
    near = torch.full_like(data["rays_o"][..., -1:], 1e-4)
    far = intersect_sphere(data["rays_o"], data["rays_d"])
#     invalid_mask = far >1.0
#     print("invalid_mask mask", invalid_mask.shape)
#     plt.imshow(invalid_mask.reshape(240,320).numpy())
#     plt.show()
#     print("inside_mask", invalid_mask)
    
else:
    import models.vanilla_nerf.helper as helper
    from torch import linalg as LA
    near = 0.2
    far = 3.0

if nerfplusplus:
    obj_t_vals, obj_samples = helper.sample_along_rays(
        rays_o=data["rays_o"],
        rays_d=data["rays_d"],
        num_samples=64,
        near = near,
        far = far,
        randomized=True,
        lindisp=False,
        in_sphere=True,
    )

    bg_t_vals, bg_samples = helper.sample_along_rays(
        rays_o=data["rays_o"],
        rays_d=data["rays_d"],
        num_samples=64,
        near=near,
        far=far,
        randomized=True,
        lindisp=False,
        in_sphere=False,
    )
else:
    all_t_vals, all_samples = helper.sample_along_rays(
        rays_o=data["rays_o"],
        rays_d=data["rays_d"],
        num_samples=64,
        near=near,
        far=far,
        randomized=True,
        lindisp=False,
    )

def contract_pts(pts, radius=1):
    mask = torch.norm(pts, dim=-1).unsqueeze(-1) > radius
    new_pts = pts.clone()/radius
    norm_pts = torch.norm(new_pts, dim=-1).unsqueeze(-1)
    contracted_points = ((1+0.2) - 0.2/(norm_pts))*(new_pts/norm_pts)*radius
    warped_points = mask*contracted_points + (~mask)*pts
    return warped_points

def contract_samples(x, order=1):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

def _contract(x):
    x_mag_sq = torch.sum(x**2, dim=-1, keepdim=True).clip(min=1e-32)
    z = torch.where(
        x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x
    )
    return z

if nerfplusplus:
#     all_samples = torch.cat((obj_samples, bg_samples[:,:,:3]), dim=0)
    all_samples = bg_samples
else:
    if contract:
        all_samples = contract_samples(all_samples)


if nerfplusplus:
    print("bg obj samples",bg_samples.shape, obj_samples.shape)
else:
    print("all_samples",all_samples.shape)


coords = get_image_coords(pixel_offset = 0.5,image_height = 120, image_width = 160)
print(coords.shape)

rays_o = data["rays_o"][:20,:]
rays_d = data["rays_d"][:20,:]
coords = coords[:20,:]

if nerfplusplus:
    num = np.random.choice(bg_samples.shape[0], 20)
    samples = bg_samples[num,:, :3]
#     num_bg = np.random.choice(bg_samples.shape[0], 20)
#     samples_bg = bg_samples[num,:, :3]
#     print(samples.shape, samples_bg.shape)
#     samples = torch.cat((samples, samples_bg), dim=0)
else:
    num = np.random.choice(all_samples.shape[0], 20)
    samples = all_samples[num,:, :3]


fig = vis_camera_samples(samples)
fig.show()

print("torch min max samples", torch.min(samples), torch.max(samples))