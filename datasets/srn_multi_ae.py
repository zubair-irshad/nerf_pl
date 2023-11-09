from base64 import encode
import imageio
import numpy as np
import torch
# import json
# from torchvision import transforms
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
import pickle
import random

img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

exclude_folders_train  = ['425ba801e904cb50f3aaed7e86215c7b', 'd2776550e240c0d642fb51f57882f419', 'd80658a2f50c753cf1335b4fef92b83f', '44f30f4c65c3142a16abce8cb03e7794', '39b307361b650db073a425eed3ac7a0b', '6c6254a92c485787f1ca7626ddabf47', 'e0901a0a26daebea59139efcde1fedcb', '24b9180ac3f89ba4715e04edb8af9c53', 'f48659c519f422132d54e7222448a731', 'cb9577139b34703945e8a12904b50643', 'e3dff7195a2026dba4db43fa521d5c03', '4e009085e3905f2159139efcde1fedcb', 'ca93e4d0ca75ab1bafe1d4530f4c6e24', '4cecc9df9c777804816bd8f64e08b2bc', 'c41fc68f756437b2511afe12ef4fe731', '292f6606c6072b96715e04edb8af9c53', '1f5a6d3c74f32053b6163196882ac0ca', '2c8e9ff5fd58ff3fcd046ccc4d5c3da2', 'bc8e978655bb60c19fec71e8f4aac226', '527d52b26bc5b397d8f9dd7647048a0c', '202fbaeffaf49f4b61c6c61410fc904b', '7edb40d76dff7455c2ff7551a4114669', 'd43dc96daed9ba0f91bfeeca48a08b93', '781b45d3eb625148248a78e10a40d8eb', '525c1f2526cf22be5909c35c7b6459c6', '78c5d8e9acc120ce16abce8cb03e7794', '1c490bf1c6b32ef6ff213501a803f212', 'd84c7484b751357faa70852df8af7bdb', 'ddb4ad84abca0edcdb8ce1e61248143', '17c32e15723ed6e0cd0bf4a0e76b8df5', 'b8599e22b152b96e55e3ad998a1ecb4', '8f87755f22470873e6725f2a23469bfc', '7d099ac5bcc09250e61b9ff60b1be412', 'b866d7e1b0336aff7c719d2d87c850d8', '4036332be89511e31141a7d4d06dc13', 'c30bf6d1ae428497c7f3070d3c7b9f30', 'fe8850296592f2b16abce8cb03e7794', '43a723b6845f6f90b1eebe42821a51d7', '957a686c3c9f956a3d982653fc5fd75b', 'c5bdc334a3df466e8e1630a4c009bdc0', '9171272d0e357c40435b5ce06ecf3e86', 'b5a6e71a63189e53e8a3b392b986583', 'e01a56af0788551e7aa225b44626f301', 'c8fa4fd7fc424121932abeb6e2fd4072', '7478183ebde9c6c2afe717997470b28d', 'd7b8287ca11d565bd9bd5ae694086d5', 'a6fe523f0ef082a2715e04edb8af9c53', '260f0644b293fccbfbc06ad9015523cf', '11d1fdaedf3ab83b8fb28f8a689c8ba3', '9c27cdc4feb2fa5d4244558fce818712', '75221b7668e145b549415f1fb7067088', '846f4ad1db06d8791e0b067dee925db4', '657ea4181e337213fa7c23b34a0b219', '810476f203a99d3586b58a9b1f5938e0', 'f6ed076d16960558e6748b6322a06ee3', '1c86d4441f5f38d552c4c70ef22e33be']
exclude_folders_val = ['8b8f4f12e5b142c016abce8cb03e7794', 'ae9b244f9bee122ba35db63c2ad6fc71', '95ebb3fd80f885ad676f197a68a5168a', 'd18817af1a2591d751a95aaa6caba1d3', 'd967be366b99ac00bac978d4dc005d3', '5d2e6410d4fb760befdff89bf9a96890', 'e4d396067b97f3676dd84bc138e22252', '36b23cc38786599285089a13cc567dbd', 'affba519865b72fc2c95ae1829869305', '819b98c138192c88e5e79d9024e2fcae', '15e52e44cdcc80ed13ded1857c15b5b6']
def get_mask_to_tensor():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.0,), (1.0,))]
    )

def get_bbox_from_mask(inst_mask, imgfile):

    rows = np.any(inst_mask, axis=1)
    cols = np.any(inst_mask, axis=0)
    rnz = np.where(rows)[0]
    cnz = np.where(cols)[0]
    if len(rnz) == 0:
        raise RuntimeError(
            "ERROR: Bad image at", imgfile, "please investigate!"
        )
    rmin, rmax = rnz[[0, -1]]
    cmin, cmax = cnz[[0, -1]]
    return cmin, cmax, rmin, rmax

def load_poses(pose_dir, idxs=[]):
    txtfiles = np.sort([os.path.join(pose_dir, f.name) for f in os.scandir(pose_dir)])
    posefile = np.array(txtfiles)[idxs]
    srn_coords_trans = np.diag(np.array([1, -1, -1, 1])) # SRN dataset

    pose = np.loadtxt(posefile).reshape(4,4)
    pose = pose@srn_coords_trans
    return torch.from_numpy(pose).float()

def normalize_image():
    ops = []
    ops.extend(
        [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    # ops.extend(
    #     [T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])]
    # )
    return T.Compose(ops)

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

def load_latent_codes_dict(latent_code_path):
    with open(latent_code_path, 'rb') as handle:
        latent_dict = pickle.load(handle)
    return latent_dict['shape_code'], latent_dict['texture_code']


class SRN_Multi_AE():
    def __init__(self, cat='srn_cars', split='cars_train',
                img_wh=(128, 128), root_dir = '/data/datasets/code_nerf', 
                num_instances_per_obj = 1, encoder_reg = False, latent_code_path=None):
        """
        cat: srn_cars / srn_chairs
        split: cars_train(/test/val) or chairs_train(/test/val)
        First, we choose the id
        Then, we sample images (the number of instances matter)
        """
        self.define_transforms()
        self.white_back = True
        self.data_dir = os.path.join(root_dir, cat, split)
        # self.ids = np.sort([f.name for f in os.scandir(self.data_dir)])
        if split == 'cars_train':
            self.ids = np.sort([f.name for f in os.scandir(self.data_dir) if f.name not in exclude_folders_train])
        else:
            self.ids = np.sort([f.name for f in os.scandir(self.data_dir) if f.name not in exclude_folders_val])
        self.num_instances_per_obj = num_instances_per_obj
        self.train = True if split.split('_')[1] == 'train' else False
        self.val = True if split.split('_')[1] == 'val' else False
        self.splits = split
        self.lenids = len(self.ids)
        self.normalize_img = normalize_image()
        self.encoder_reg = encoder_reg

<<<<<<< HEAD
=======
        self.samples_per_epoch = 20000
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0
        if self.encoder_reg:
            self.shape_codes, self.texture_codes = load_latent_codes_dict(latent_code_path)

        self.img_wh = img_wh
        # bounds, common for all scenes
        self.cat = cat
        if self.cat == 'srn_cars':
            self.near = 0.8
            self.far = 1.8
        else:
            self.near = 1.25
            self.far = 2.75

        self.bounds = np.array([self.near, self.far])

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
<<<<<<< HEAD
        return self.lenids
=======
        if self.splits == 'cars_train':
            return self.samples_per_epoch
        else:
            return self.lenids

>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0

    def load_img(self, img_dir, idxs = []):

        allimgfiles = np.sort([os.path.join(img_dir, f.name) for f in os.scandir(img_dir)])
        imgfile = np.array(allimgfiles)[idxs]
        img = imageio.imread(imgfile, pilmode='RGB')
        enc_image =  img
        instance_mask = np.sum(img, axis=-1)!=(255*3)
        
        x1, x2, y1, y2 = get_bbox_from_mask(instance_mask, imgfile)
        img = img[y1:y2, x1:x2]
        instance_mask = instance_mask[y1:y2, x1:x2]
        return img, enc_image, instance_mask, (x1, x2, y1, y2)
    
    def __getitem__(self, idx):
<<<<<<< HEAD
        obj_id = self.ids[idx]
        if self.train:
            rays_o, view_dirs, rays_d, img, enc_img, radii, instance_mask = self.return_train_data(obj_id)
        elif self.val:
=======
        if self.train:
            train_idx = random.randint(0, len(self.ids) - 1)
            obj_id = self.ids[train_idx]
            rays_o, view_dirs, rays_d, img, enc_img, radii, instance_mask = self.return_train_data(obj_id)
        elif self.val:
            obj_id = self.ids[idx]
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0
            rays_o, view_dirs, rays_d, img, enc_img, radii, instance_mask = self.return_val_data(obj_id)

        return rays_o, view_dirs, rays_d, img, enc_img, radii, instance_mask
    
    def return_train_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        instances = np.random.choice(50, self.num_instances_per_obj)[0]
        c2w = load_poses(pose_dir, instances)
        img, enc_img, instance_mask, crop_size = self.load_img(img_dir, instances)

        x1, x2, y1, y2 = crop_size
        focal, H, W = load_intrinsic(intrinsic_path)

        w, h = self.img_wh
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        c2w = torch.FloatTensor(c2w)[:3, :4]
        rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)

        rays_o = rays_o.reshape(h,w,3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        rays_d = rays_d.reshape(h,w,3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        view_dirs = view_dirs.reshape(h,w,3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        radii = radii.reshape(h,w,1)[y1:y2, x1:x2].contiguous().view(-1, 1)
            
        return rays_o, view_dirs, rays_d, img, enc_img, radii, instance_mask

    def return_val_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        instances = np.random.choice(50, 1)[0]
        c2w = load_poses(pose_dir, instances)
        img, enc_img, instance_mask, crop_size = self.load_img(img_dir, instances)

        x1, x2, y1, y2 = crop_size
        focal, H, W = load_intrinsic(intrinsic_path)

        w, h = self.img_wh
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        c2w = torch.FloatTensor(c2w)[:3, :4]
        rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)

        rays_o = rays_o.reshape(h,w,3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        rays_d = rays_d.reshape(h,w,3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        view_dirs = view_dirs.reshape(h,w,3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        radii = radii.reshape(h,w,1)[y1:y2, x1:x2].contiguous().view(-1, 1)
            
        return rays_o, view_dirs, rays_d, img, enc_img, radii, instance_mask

def collate_lambda_train(batch, model_type, ray_batch_size=1024):
    imgs = list()
    instance_masks = list()
    rays = list()
    view_dirs = list()
    rays_d = list()
    rgbs = list()
    radii = list()

    for el in batch:
        cam_rays, cam_view_dirs, cam_rays_d, img, enc_img, camera_radii, instance_mask = el
        img = Image.fromarray(np.uint8(img))
        img = T.ToTensor()(img)
        enc_img = T.ToTensor()(enc_img)
        
        instance_mask = T.ToTensor()(instance_mask)
        camera_radii = torch.FloatTensor(camera_radii)
        cam_rays = torch.FloatTensor(cam_rays)
        cam_view_dirs = torch.FloatTensor(cam_view_dirs)
        cam_rays_d = torch.FloatTensor(cam_rays_d)

        _, H, W = img.shape
        instance_mask_t = instance_mask.permute(1,2,0).flatten(0,1).squeeze(-1)
        #equal probability of foreground and background
        N_fg = int(ray_batch_size * 0.7)
        N_bg = ray_batch_size - N_fg
        b_fg_inds = torch.nonzero(instance_mask_t == 1)
        b_bg_inds = torch.nonzero(instance_mask_t == 0)
        b_fg_inds = b_fg_inds[torch.randperm(b_fg_inds.shape[0])[:N_fg]]
        b_bg_inds = b_bg_inds[torch.randperm(b_bg_inds.shape[0])[:N_bg]]
        pix_inds = torch.cat([b_fg_inds, b_bg_inds], 0).squeeze(-1)

        if len(pix_inds) < ray_batch_size:
            pix_inds = torch.cat(
                [
                    pix_inds,
                    torch.tensor(random.choices(pix_inds.tolist(), k=ray_batch_size - len(pix_inds))),
                ]
            )
        
        rgb_gt = img.permute(1,2,0).flatten(0,1)[pix_inds,...] 
        msk_gt = instance_mask.permute(1,2,0).flatten(0,1)[pix_inds,...]
        radii_gt = camera_radii.view(-1)[pix_inds]
        ray = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds]
        ray_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])[pix_inds]
        viewdir = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])[pix_inds]
        imgs.append(
            img_transform(enc_img)
        )
        instance_masks.append(msk_gt)  
        rays.append(ray)
        view_dirs.append(viewdir)
        rays_d.append(ray_d)
        rgbs.append(rgb_gt)
        radii.append(radii_gt)
    
    imgs = torch.stack(imgs) 

    rgbs = torch.stack(rgbs, 0)  
    instance_masks = torch.stack(instance_masks, 0)
    rays = torch.stack(rays, 0)  
    rays_d = torch.stack(rays_d, 0)  
    view_dirs = torch.stack(view_dirs, 0)  
    radii = torch.stack(radii, 0)  
    
    if model_type == "Vanilla":
        sample = {
            "src_imgs": imgs,
            "rays": rays,
            "rgbs": rgbs,
            "instance_mask": instance_masks,
        }
    else:
        sample = {}
        sample["src_imgs"] = imgs
        sample["rays_o"] = rays
        sample["rays_d"] = rays_d
        sample["viewdirs"] = view_dirs
        sample["target"] = rgbs
        sample["radii"] = radii.unsqueeze(-1)
        sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
        sample["normals"] = torch.zeros_like(sample["rays_o"])
        sample["instance_mask"] = instance_masks
    return sample


def collate_lambda_val(batch, model_type):
<<<<<<< HEAD
    mask_to_tensor = get_mask_to_tensor()
=======
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0
    cam_rays, cam_view_dirs, cam_rays_d, img, enc_img, camera_radii, instance_mask = batch[0]
    h,w,_ = img.shape
    img = Image.fromarray(np.uint8(img))
    img = T.ToTensor()(img)
    enc_img = T.ToTensor()(enc_img)
    instance_mask = T.ToTensor()(instance_mask)
    camera_radii = torch.FloatTensor(camera_radii)
    cam_rays = torch.FloatTensor(cam_rays)
    cam_view_dirs = torch.FloatTensor(cam_view_dirs)
    cam_rays_d = torch.FloatTensor(cam_rays_d)

    rgbs = img.permute(1,2,0).flatten(0,1)
    instance_masks = instance_mask.permute(1,2,0).flatten(0,1)
    radii = camera_radii.view(-1)
    rays = cam_rays.view(-1, cam_rays.shape[-1])
    rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
    view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])
    imgs = img_transform(enc_img)
    
    if model_type == "Vanilla":
        sample = {
            "src_imgs": imgs,
            "rays": rays,
            "rgbs": rgbs,
            "instance_mask": instance_masks,
            "img_wh": np.array((w,h))
        }
    else:
        sample = {}
        sample["src_imgs"] = imgs
        sample["rays_o"] = rays
        sample["rays_d"] = rays_d
        sample["viewdirs"] = view_dirs
        sample["target"] = rgbs
        sample["radii"] = radii
        sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
        sample["normals"] = torch.zeros_like(sample["rays_o"])
        sample["instance_mask"] = instance_masks
        sample["img_wh"] = np.array((w,h))
    return sample