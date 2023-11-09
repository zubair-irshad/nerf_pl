import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
from matplotlib.cm import get_cmap
from collections import namedtuple
from save_nocs_PD.transform_utils import *
from save_nocs_PD.viz_utils import *

def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'Car', 'Person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images

    'cuboid_id'   , # ID that is used for 3d cuboid annotations

    'is_thing'    , # Whether this label distinguishes between single instances or not

    'color'       , # The color of this label (used for generating human readable images)
] )

labels = [
    #      Name                              id  cuboid_id is_thing          color               
    Label( "Animal",                          0,        -1,    True, (220, 20,180) ), 
    Label( "Bicycle",                         1,         8,    True, (119, 11, 32) ),  
    Label( "Bicyclist",                       2,         0,    True, ( 64, 64, 64) ),
    Label( "Building",                        3,        -1,   False, ( 70, 70, 70) ), 
    Label( "Bus",                             4,         3,    True, (  0, 60,100) ),
    Label( "Car",                             5,         2,    True, (  0,  0,142) ), 
    Label( "Caravan/RV",                      6,         3,    True, (  0,  0, 90) ), 
    Label( "ConstructionVehicle",             7,        -1,    True, ( 32, 32, 32) ), 
    Label( "CrossWalk",                       8,        -1,    True, (255,255,255) ), 
    Label( "Fence",                           9,        -1,   False, (190,153,153) ), 
    Label( "HorizontalPole",                 10,        -1,    True, (153,153,153) ), 
    Label( "LaneMarking",                    11,        -1,   False, (220,220,220) ), 
    Label( "LimitLine",                      12,        -1,   False, (180,180,180) ), 
    Label( "Motorcycle",                     13,         4,    True, (  0,  0,230) ), 
    Label( "Motorcyclist",                   14,        11,    True, (128,128,128) ),
    Label( "OtherDriveableSurface",          15,        -1,   False, ( 80,  0,  0) ), 
    Label( "OtherFixedStructure",            16,        -1,   False, (150,  0,  0) ), 
    Label( "OtherMovable",                   17,        -1,    True, (230,  0,  0) ), 
    Label( "OtherRider",                     18,        -1,    True, (192,192,192) ), 
    Label( "Overpass/Bridge/Tunnel",         19,        -1,   False, (150,100,100) ), 
    Label( "OwnCar(EgoCar)",                 20,         2,   False, (128,230,128) ), 
    Label( "ParkingMeter",                   21,        -1,   False, ( 32, 32, 32) ),
    Label( "Pedestrian",                     22,         0,    True, (220, 20, 60) ), 
    Label( "Railway",                        23,        -1,   False, (230,150,140) ), 
    Label( "Road",                           24,        -1,   False, (128, 64,128) ), 
    Label( "RoadBarriers",                   25,        -1,   False, ( 80, 80, 80) ), 
    Label( "RoadBoundary(Curb)",             26,        -1,   False, (100,100,100) ), 
    Label( "RoadMarking",                    27,        -1,   False, (255,220,  0) ), 
    Label( "SideWalk",                       28,        -1,   False, (244, 35,232) ), 
    Label( "Sky",                            29,        -1,   False, ( 70,130,180) ), 
    Label( "TemporaryConstructionObject",    30,        -1,    True, (255,160, 20) ), 
    Label( "Terrain",                        31,        -1,   False, ( 81,  0, 81) ), 
    Label( "TowedObject",                    32,         9,    True, (  0,  0,110) ), 
    Label( "TrafficLight",                   33,        -1,    True, (250,170, 30) ), 
    Label( "TrafficSign",                    34,        -1,    True, (220,220,  0) ), 
    Label( "Train",                          35,         6,    True, (  0, 80,100) ), 
    Label( "Truck",                          36,         1,    True, (  0,  0, 70) ), 
    Label( "Vegetation",                     37,        -1,   False, (107,142, 35) ),   
    Label( "VerticalPole",                   38,        -1,    True, (153,153,153) ), 
    Label( "WheeledSlow",                    39,         5,    True, (  0, 64, 64) ),
    Label( "LaneMarkingOther",               40,        -1,   False, (255,255,  0) ), 
    Label( "LaneMarkingGap",                 41,        -1,   False, (  0,255,255) ), 

    Label( "Fence(Transparent)",             42,        -1,   False, ( 85, 75, 75) ), 
    Label( "StaticObject(Trashcan)",         43,        -1,    True, ( 75,  0,  0) ), 
    Label( "Vegetation(Bush)",               44,        -1,   False, ( 54, 71, 18) ), 

    Label( "OtherPole",                      45,        -1,   False, (200,200,200) ), 

    Label( "Powerline",                      46,        -1,   False, ( 32, 32, 32) ), 
    
    Label( "SchoolBus",                      47,        -1,    True, ( 15,123,122) ),

    Label( "ParkingLot",                     48,        -1,   False, (104, 27, 83) ),

    Label( "RoadMarkingSpeed",               49,        -1,   False, (228,150, 49) ),

    Label( "Vegetation(GroundCover)",        50,        -1,   False, ( 35, 46, 11) ),
    Label( "Vegetation(Grass)",              51,        -1,   False, ( 47,106, 45) ),
    Label( "Vegetation(Tree)",               52,        -1,   False, (107,142, 35) ),

    Label( "Debris",                         53,        -1,   True, ( 80, 41, 21) ),

    Label( "RoadBoundary(CurbFlat)",         54,        -1,   False, (120,120,120) ),
    
    Label( "LaneMarking(Parking)",           55,        -1,   False, (210,210,210) ), 
    Label( "LaneMarking(ParkingIndicator)",  56,        -1,   False, (210,220,210) ), 
    Label( "RoadMarkingArrows",              57,        -1,   False, (228,190, 60) ), 
    Label( "RoadMarkingBottsDots",           58,        -1,   False, (228,120, 49) ), 
    Label( "StopLine",                       59,        -1,   False, (180,150,150) ),

    Label( "ChannelizingDevice",             60,        -1,    True, (237,190,120) ),

    Label( "LaneMarkingSpan",                61,        -1,   False, (  0,180,255) ),

    Label( "StaticObject(BikeRack)",         62,        -1,    True, ( 75,  0, 75) ),

    Label( "ParkingSpot",                    63,        -1,    True, ( 84,155,205) ),

    Label( "RoadBoundary(CurbTop)",          64,        -1,   False, ( 140,140,140) ),
    Label( "RoadBoundary(CurbSide)",         65,        -1,   False, ( 140,160,140) ),
    Label( "RoadBoundary(CurbRoadLevel)",    66,        -1,   False, ( 140,180,140) ),

    Label("Multipath(Noise)",                225,       -1,   False, ( 70, 255, 20) ),
    Label("ThermalNoise(Noise)",             226,       -1,   False, ( 0, 255, 140) ),
    Label("Fog(Noise)",                      227,       -1,   False, ( 114, 0, 255) ),
    Label("Rain(Noise)",                     228,       -1,   False, ( 105, 255,255)),

    Label( "Void",                           255,        -1,   False, (  0,  0,  0) )
]

# name to label object
name_to_label      = { label.name    : label for label in labels }
# id to label object
id_to_label        = { label.id      : label for label in labels }
# id to label color tuple
id_to_color        = { label.id      : label.color for label in labels }

id_to_cuboid_id    = { label.id      : label.cuboid_id for label in labels if label.cuboid_id != -1 }

id_to_is_thing     = { label.id      : label.is_thing for label in labels }

id_to_color_lookup = np.zeros((256, 3), dtype=np.uint8)
for label in labels:
    id_to_color_lookup[label.id] = label.color

def get_seg_rgb(seg_map):
    return id_to_color_lookup[np.array(seg_map)]

def get_inst_rgb(inst_map) -> np.ndarray:
    """
    Returns data array as an RGB instances image.
    """
    # Instance id to color map
    import cv2
    np.random.seed(1234)
    random_ids = np.random.randint(0, 256, 65536, dtype=np.uint8)
    instance_id_to_color_lookup = cv2.applyColorMap(random_ids, cv2.COLORMAP_JET).reshape(-1, 3).astype(np.float32)
    instance_id_to_color_lookup[0][:] = 0  # sets black color for null instance id

    # Generates instances rgb image
    instance_image_rgb = instance_id_to_color_lookup[np.array(inst_map)]
    return instance_image_rgb.astype(np.uint8)

def get_merged_instance_rgb(rgb, instance_image_rgb) -> np.ndarray:
    """
    Returns data array as an RGB instances image.
    The instance image is merged with a given rgb image

    Args:
        rgb: The rgb image with which to merge the instance image
    """
    rgb = np.array(rgb)
    instance_image_rgb = instance_image_rgb.astype(np.float64)
    alpha = np.sum(instance_image_rgb, axis=2)
    alpha[alpha > 0] = 0.5
    alpha = np.expand_dims(alpha, axis=2)
    alpha = np.broadcast_to(alpha, shape=instance_image_rgb.shape)
    # Generates output image by combining rgb image with instances image
    output_img = rgb * (1-alpha) + instance_image_rgb * alpha
    return output_img.astype(np.uint8)
    
def depth2inv(depth):
    """
    Invert a depth map to produce an inverse depth map
    Parameters
    ----------
    depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Depth map
    Returns
    -------
    inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Inverse depth map
    """
    inv_depth = 1. / depth.clamp(min=1e-6)
    inv_depth[depth <= 0.] = 0.
    return inv_depth

def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.
    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization
    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        inv_depth = inv_depth.squeeze(0).squeeze(0)
        # Squeeze if depth channel exists
        # if len(inv_depth.shape) == 3:
        #     inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

# read non-scaled poses for visualization 
def read_poses(pose_dir_train, img_files_train, output_boxes = False):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    focal = data['focal']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        # c2w[:3, 3] = c2w[:3, 3] - obj_location
        # c2w[:3, 3] = c2w[:3, 3]
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    # pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))
    # all_c2w_train[:, :3, 3] *= pose_scale_factor

    all_c2w_val = all_c2w_train[100:]
    all_c2w_train = all_c2w_train[:100]
    # Get bounding boxes for object MLP training only
    if output_boxes:
        all_boxes = []
        all_translations= []
        all_rotations = []
        for k,v in data['bbox_dimensions'].items():
                # all_boxes.append(bbox*pose_scale_factor)
                all_boxes.append(np.array(v))
                #New scene 200 uncomment here
                all_rotations.append(data["obj_rotations"][k])
                # translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor
                translation = np.array(data['obj_translations'][k])
                all_translations.append(translation)
        RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
        return all_c2w_train, all_c2w_val, focal, img_wh, RTs
    else:
        return all_c2w_train, all_c2w_val, focal, img_wh

def preprocess_RTS_for_vis(RTS):
    all_R = RTS['R']
    all_T = RTS['T']
    all_s = RTS['s']

    obj_poses = {}
    obj_poses["bbox_dimensions"] = []
    obj_poses["obj_translations"] = []
    obj_poses["obj_rotations"] = []

    for Rot,Tran,sca in zip(all_R, all_T, all_s):
        bbox_extent = np.array([(sca[1,0]-sca[0,0]), (sca[1,1]-sca[0,1]), (sca[1,2]-sca[0,2])])
        cam_t = Tran
        bbox = np.array(sca)
        bbox_diff = bbox[0,2]+((bbox[1,2]-bbox[0,2])/2)
        cam_t[2] += bbox_diff
        cam_rot = np.array(Rot)[:3, :3]

        obj_poses["bbox_dimensions"].append(bbox_extent)
        obj_poses["obj_translations"].append(cam_t)
        obj_poses["obj_rotations"].append(cam_rot)
    return obj_poses

def get_RTs(obj_poses):
    all_boxes = []
    all_translations = []
    all_rotations= []
    for i, bbox in enumerate(obj_poses['bbox_dimensions']):
            all_boxes.append(bbox)
            translation = np.array(obj_poses['obj_translations'][i])
            all_translations.append(translation)
            all_rotations.append(obj_poses["obj_rotations"][i])
    RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    return RTs

def vis_bounding_box_image(c2w, im, RTs, K_matrix, bbox_save_name):

    im = np.array(im)
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']
    box_obb = []
    axes = []
    for rotation, translation, size in zip(all_rotations, all_translations, bbox_dimensions):
        box = get_3d_bbox(size)
        pose = np.eye(4)
        pose[:3,:3] = np.array(rotation)
        pose[:3, 3] = np.array(translation)
        pose = np.linalg.inv(convert_pose(c2w)) @ pose

        unit_box_homopoints = convert_points_to_homopoints(box.T)
        morphed_box_homopoints = pose @ unit_box_homopoints
        rotated_box = convert_homopoints_to_points(morphed_box_homopoints).T
        points_obb = convert_points_to_homopoints(np.array(rotated_box).T)

        box_obb.append(project(K_matrix, points_obb).T)

        xyz_axis = 1.0*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()

        transformed_axes = transform_coordinates_3d(xyz_axis, pose)
        projected_axes = calculate_2d_projections(transformed_axes, K_matrix[:3,:3])

        axes.append(projected_axes)

    colors_box = [(63, 234, 237)]
    colors_mpl = ['#EAED3F']

    plt.figure()
    plt.clf()
    plt.xlim((0, im.shape[1]))
    plt.ylim((0, im.shape[0]))
    plt.gca().invert_yaxis()
    plt.axis('off')
    for k in range(len(colors_box)):
        for points_2d, axis in zip(box_obb, axes):
            points_2d = np.array(points_2d)
            im = draw_bboxes_mpl_glow(im, points_2d, axis, colors_mpl[k])

    plt.imshow(im)
    # plt.show()
    plt.savefig(bbox_save_name, bbox_inches='tight',pad_inches = 0)
    
