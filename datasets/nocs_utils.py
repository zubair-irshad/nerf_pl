import numpy as np
import cv2
import os
import _pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses

@dataclasses.dataclass
class Pose:
  camera_T_object: np.ndarray
  scale_matrix: np.ndarray

def load_depth(depth_path):
    """ Load depth image from img_path. """
    # depth_path = depth_path + '_depth.png'
    # print("depth_path", depth_path)
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. """
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754' or model_id == 'd3b53f56b4a7b3b3c9f016d57db96408':
                continue
            # process foreground objects
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes

def get_GT_poses(data_dir, img_path, class_ids, instance_ids, model_list, bboxes, is_pcd_out=None):
    
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    source = 'Real'
    model_file_path = ['obj_models/camera_valnew.pkl', 'obj_models/real_testnew.pkl']
    # model_file_path = ['obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
    #model_file_path = ['obj_models_transformed/camera_real_val_sdf.pkl']
    
    models = {}
    for path in model_file_path:
        with open(os.path.join(data_dir, path), 'rb') as f:
            models.update(cPickle.load(f))
    model_sizes = {}
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)

    num_insts = len(instance_ids)
    # match each instance with NOCS ground truth to properly assign gt_handle_visibility
    nocs_dir = os.path.join(data_dir, 'results/nocs_results')
    if source == 'CAMERA':
        nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
            img_path.split('/')[-3], img_path.split('/')[-1]))
    else:
        nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(
            img_path.split('/')[-3], img_path.split('/')[-1]))
    with open(nocs_path, 'rb') as f:
        nocs = cPickle.load(f)
    gt_class_ids = nocs['gt_class_ids']
    gt_bboxes = nocs['gt_bboxes']
    gt_sRT = nocs['gt_RTs']
    gt_handle_visibility = nocs['gt_handle_visibility']
    map_to_nocs = []
    for i in range(num_insts):
        gt_match = -1
        for j in range(len(gt_class_ids)):
            if gt_class_ids[j] != class_ids[i]:
                continue
            if np.sum(np.abs(bboxes[i] - gt_bboxes[j])) > 5:
                continue
            # match found
            gt_match = j
            break
        # check match validity
        assert gt_match > -1, print(img_path, instance_ids[i], 'no match for instance')
        assert gt_match not in map_to_nocs, print(img_path, instance_ids[i], 'duplicate match')
        map_to_nocs.append(gt_match)
    # copy from ground truth, re-label for mug category
    handle_visibility = gt_handle_visibility[map_to_nocs]
    sizes = np.zeros((num_insts, 3))
    # poses = np.zeros((num_insts, 4, 4))
    poses = []
    scales = np.zeros(num_insts)
    rotations = np.zeros((num_insts, 3, 3))
    translations = np.zeros((num_insts, 3))
    for i in range(num_insts):
        gt_idx = map_to_nocs[i]
        sizes[i] = model_sizes[model_list[i]]
        sRT = gt_sRT[gt_idx]
        s = np.cbrt(np.linalg.det(sRT[:3, :3]))
        R = sRT[:3, :3] / s
        T = sRT[:3, 3]
        # re-label mug category
        if class_ids[i] == 6:
            T0 = mug_meta[model_list[i]][0]
            s0 = mug_meta[model_list[i]][1]
            T = T - s * R @ T0
            s = s / s0
        # used for test during training
        # scales[i] = s
        # rotations[i] = R
        # translations[i] = T
        # # used for evaluation
        # sRT = np.identity(4, dtype=np.float32)
        # sRT[:3, :3] = s * R
        # sRT[:3, 3] = T

        scale_matrix = np.eye(4)
        scale_mat = s*np.eye(3, dtype=float)
        scale_matrix[0:3, 0:3] = scale_mat
        camera_T_object = np.eye(4)
        camera_T_object[:3,:3] = R
        camera_T_object[:3,3] = T
        poses.append(Pose(camera_T_object=camera_T_object, scale_matrix=scale_matrix))
        # poses[i] = sRT

    model_points = [models[model_list[i]].astype(np.float32) for i in range(len(class_ids))]
    # if is_pcd_out:
    #     model_points = [models[model_list[i]].astype(np.float32) for i in range(len(class_ids))]
    #     latent_embeddings = get_latent_embeddings(data_dir, model_points)
    #     return poses, latent_embeddings
    return poses, model_points

def get_latent_embeddings(data_dir, point_clouds):
    emb_dim = 128
    n_pts = 2048
    model_path = os.path.join(data_dir, 'auto_encoder_model', 'model_50_nocs.pth')
    estimator = PointCloudAE(emb_dim, n_pts)
    estimator.cuda()
    estimator.load_state_dict(torch.load(model_path))
    estimator.eval()

    latent_embeddings =[]
    for i in range(len(point_clouds)):
        batch_xyz = torch.from_numpy(point_clouds[i]).to(device="cuda", dtype=torch.float)
        batch_xyz = batch_xyz.unsqueeze(0)
        emb, _ = estimator(batch_xyz)
        emb = emb.squeeze(0).cpu().detach().numpy()
        latent_embeddings.append(emb)
    return latent_embeddings


class PointCloudEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(PointCloudEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.fc = nn.Linear(1024, emb_dim)

    def forward(self, xyz):
        """
        Args:
            xyz: (B, 3, N)

        """
        np = xyz.size()[2]
        x = F.relu(self.conv1(xyz))
        x = F.relu(self.conv2(x))
        global_feat = F.adaptive_max_pool1d(x, 1)
        x = torch.cat((x, global_feat.repeat(1, 1, np)), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
        embedding = self.fc(x)
        return embedding


class PointCloudDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3*n_pts)

    def forward(self, embedding):
        """
        Args:
            embedding: (B, 512)

        """
        bs = embedding.size()[0]
        out = F.relu(self.fc1(embedding))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out_pc = out.view(bs, -1, 3)
        return out_pc


class PointCloudAE(nn.Module):
    def __init__(self, emb_dim=512, n_pts=1024):
        super(PointCloudAE, self).__init__()
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = PointCloudDecoder(emb_dim, n_pts)

    def forward(self, in_pc, emb=None):
        """
        Args:
            in_pc: (B, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_pc: (B, n_pts, 3)

        """
        if emb is None:
            xyz = in_pc.permute(0, 2, 1)
            emb = self.encoder(xyz)
        out_pc = self.decoder(emb)
        return emb, out_pc


def rebalance_mask(mask, fg_weight=None, bg_weight=None):
    if fg_weight is None and bg_weight is None:
        foreground_cnt = max(mask.sum(), 1)
        background_cnt = max((~mask).sum(), 1)
        balanced_weight = np.ones_like(mask).astype(np.float32)
        balanced_weight[mask] = float(background_cnt) / foreground_cnt
        balanced_weight[~mask] = float(foreground_cnt) / background_cnt
    else:
        balanced_weight = np.ones_like(mask).astype(np.float32)
        balanced_weight[mask] = fg_weight
        balanced_weight[~mask] = bg_weight
    # print('fg {} bg {}'.format(foreground_cnt, background_cnt))
    # print(balanced_weight.min(), balanced_weight.max())
    # print(balanced_weight.shape)
    # cv2.normalize(balanced_weight, balanced_weight, 0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow('img_balanced_weight', balanced_weight)
    # cv2.waitKey(5)
    return balanced_weight


def rebalance_mask_tensor(mask, fg_weight=None, bg_weight=None):
    if fg_weight is None and bg_weight is None:
        foreground_cnt = max(mask.sum(), 1)
        background_cnt = max((~mask).sum(), 1)
        balanced_weight = np.ones_like(mask).astype(np.float32)
        balanced_weight[mask] = float(background_cnt) / foreground_cnt
        balanced_weight[~mask] = float(foreground_cnt) / background_cnt
    else:
        balanced_weight = torch.ones_like(mask, dtype=torch.float32)
        balanced_weight[mask] = fg_weight
        balanced_weight[~mask] = bg_weight
    return balanced_weight