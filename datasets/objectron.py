import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# from .google_scanned_utils import load_image_from_exr, load_seg_from_exr
import struct
from .ray_utils import *
from .nocs_utils import rebalance_mask
import glob

def make_poses_bounds_array(frame_data, near=0.2, far=10):
    # See https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap
    # Returns an array of shape (N, 17).
    rows = []
    all_c2w = []
    all_focal = []
    for frame in frame_data:
        camera = frame.camera      
        focal = camera.intrinsics[0]
        cam_to_world = np.array(camera.transform).reshape(4,4)
        all_c2w.append(cam_to_world)
        all_focal.append(focal)
    return all_c2w, all_focal

def load_frame_data(geometry_filename):
    # See get_geometry_data in objectron-geometry-tutorial.ipynb
    frame_data = []
    with open(geometry_filename, 'rb') as pb:
        proto_buf = pb.read()

        i = 0
        while i < len(proto_buf):
            msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
            i += 4
            message_buf = proto_buf[i:i + msg_len]
            i += msg_len
            frame = ar_metadata_protocol.ARFrame()
            frame.ParseFromString(message_buf)
            frame_data.append(frame)
    return frame_data

class ObjectronDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(1600, 1600)):
        self.root_dir = root_dir
        self.split = split
        print("img_wh", img_wh)
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.all_c2w = []
        self.read_meta()
        self.white_back = False
        # self.focal = 1931.371337890625
        # self.focal *= self.img_wh[0]/1600 # modify focal length to match size self.img_wh

    def read_meta(self):

        self.base_dir = '/home/ubuntu/nerf_pl/data/objectron/chair/chair_batch-24_33'
                #json_files = [pos_json for pos_json in os.listdir(base_dir) if pos_json.endswith('.json')]
        sfm_arframe_filename = self.base_dir + '/sfm_arframe.pbdata'
        frame_data = load_frame_data(sfm_arframe_filename)
        self.all_c2w, self.all_focal = make_poses_bounds_array(frame_data, near=0.2, far=10)
        self.meta = sorted(glob.glob(self.base_dir+'/images/*.png'))
        

        # print("img_wh",  self.img_wh)
        # print("self.focal", self.focal)
        
        # print("self.focal", self.focal)
        w, h = self.img_wh
        # bounds, common for all scenes
        self.near = 0.2
        self.far = 10
        self.bounds = np.array([self.near, self.far])
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for i, img_name in enumerate(self.meta):
                if i>200:
                    continue
                c2w = self.all_c2w[i]

                focal = self.all_focal[i]
                print("self.focal", focal)
                # focal *= self.img_wh[0]/1440 # modify focal length to match size self.img_wh
                # for bottle
                focal *= self.img_wh[0]/1440 # modify focal length to match size self.img_wh
                print("self.focal after", focal)
                directions = get_ray_directions(h, w, focal) # (h, w, 3)
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, rays_d = get_rays(directions, c2w)

                img = Image.open(os.path.join(self.base_dir, 'images', img_name))       
                img = self.transform(img) # (h, w, 3)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1)] # (h*w, 8)
                self.all_rgbs += [img]
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

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
            val_idx = 201
            img_name = self.meta[val_idx]
            w, h = self.img_wh
            c2w = self.all_c2w[val_idx]

            focal = self.all_focal[val_idx]
            focal *= self.img_wh[0]/1440 # modify focal length to match size self.img_wh
            
            directions = get_ray_directions(h, w, focal) # (h, w, 3)

            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            img = Image.open(os.path.join(self.base_dir, 'images', img_name)) 
            img = self.transform(img) # (h, w, 3)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
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
            print("self.poses_test", self.poses_test.shape)
            c2w = torch.FloatTensor(self.poses_test[idx])[:3,:4]
            print("c2w", c2w.shape)
            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            print("directions", directions.shape)
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            sample = {'rays': rays,
                      'c2w': c2w}   
        return sample