import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *
from .nocs_utils import rebalance_mask
import glob

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    print("radius", radius)
    radius = 1.0
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def make_poses(base_dir):
    intrinsic_path = os.path.join(base_dir, 'intrinsics.txt')
    focal, H, W = load_intrinsic(intrinsic_path)

    pose_dir = os.path.join(base_dir, 'pose')
    txtfiles = np.sort([os.path.join(pose_dir, f.name) for f in os.scandir(pose_dir)])
    posefiles = np.array(txtfiles)
    srn_coords_trans = np.diag(np.array([1, -1, -1, 1])) # SRN dataset
    poses = []
    all_c2w = []
    for posefile in posefiles:
        pose = np.loadtxt(posefile).reshape(4,4)
        all_c2w.append(pose@srn_coords_trans)
    return all_c2w, focal, H, W

class SRNDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(128, 128), use_mask = False, crop_img = True):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.all_c2w = []
        self.use_mask = use_mask
        self.white_back = True
        self.crop_img = crop_img
        self.read_meta()


    def read_meta(self):

        self.base_dir = '/data/datasets/code_nerf/srn_cars/cars_train/a7b76ead88d243133ffe0e5069bf1eb5'
                #json_files = [pos_json for pos_json in os.listdir(base_dir) if pos_json.endswith('.json')]
        #sfm_arframe_filename = self.base_dir + '/sfm_arframe.pbdata'
        
        self.all_c2w, self.focal, H, W = make_poses(self.base_dir)
        self.focal *= self.img_wh[0]/128 # modify focal length to match size self.img_wh

        self.meta = sorted(glob.glob(self.base_dir+'/rgb/*.png'))
        
        w, h = self.img_wh
        # bounds, common for all scenes
        self.near = 0.3
        self.far = 2.0
        self.bounds = np.array([self.near, self.far])
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for i, img_name in enumerate(self.meta):
                c2w = self.all_c2w[i]

                img = Image.open(img_name)   
                img = self.transform(img) # (h, w, 3)
                print("img.shape", img.shape)
                if self.crop_img:
                    img = img[:,32:-32,32:-32]
                #     h, w = h // 2, w//2
                # print("img", img.shape)
                # print()

                img = img.contiguous().view(4, -1).permute(1, 0) # (h*w, 4) RGBA 
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB


                directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, rays_d = get_rays(directions, c2w)
                
                #img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
                if not self.use_mask:
                    self.all_rays += [torch.cat([rays_o, rays_d, 
                                    self.near*torch.ones_like(rays_o[:, :1]),
                                    self.far*torch.ones_like(rays_o[:, :1])],
                                    1)] # (h*w, 8)
                    self.all_rgbs += [img]
                else:
                    valid_mask = (img.sum(1)<3).flatten() # (H*W) valid color area
                    ray_array = torch.cat([rays_o, rays_d, 
                                        self.near*torch.ones_like(rays_o[:, :1]),
                                        self.far*torch.ones_like(rays_o[:, :1])],
                                        1) # (h*w, 8)
                    self.all_rays += [ray_array[valid_mask]] # remove valid_mask for later epochs
                    self.all_rgbs += [img[valid_mask]] # remove valid_mask for later epochs

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

        elif self.split =='test':
                #1.0 is radius of hemisphere
                #radius = 1.2 * 1.0
                self.poses_test = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 1 # only validate 8 images (to support <=8 gpus)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
            }

        elif self.split == 'val': # create data for each image separately
            val_idx = 48
            img_name = self.meta[val_idx]
            w, h = self.img_wh
            c2w = self.all_c2w[val_idx]

            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            img = Image.open(os.path.join(self.base_dir, 'images', img_name)) 
            #img = img.convert('RGB')
            img = self.transform(img) # (h, w, 3)
            #img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

            if self.crop_img:
                img = img[:,32:-32,32:-32]
                # h, w = h // 2, w//2

            img = img.contiguous().view(4, -1).permute(1, 0) # (h*w, 4) RGBA 
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            # valid_mask = (img.sum(1)<3).flatten() # (H*W) valid color area
            # img = img[valid_mask] # remove valid_mask for later epochs 

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            sample = {
                "rays": rays,
                "rgbs": img
            }  
        else:
            w, h = self.img_wh
            c2w = torch.FloatTensor(self.poses_test[idx])[:3,:4]
            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            sample = {'rays': rays,
                      'c2w': c2w}   
        return sample