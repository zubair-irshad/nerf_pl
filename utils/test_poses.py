import math
import numpy as np
from scipy.spatial.transform import Rotation

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def convert_pose_spiral(C2W):
    convert_mat = np.zeros((4,4))
    convert_mat[0,1] = 1
    convert_mat[1, 0] = 1
    convert_mat[2, 2] = -1
    convert_mat[3,3] = 1
    C2W = np.matmul(C2W, convert_mat)
    return C2W
    
def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0., -1., 0.])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat, forward

def get_archimedean_spiral(sphere_radius, num_steps=80):
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 40
    r = sphere_radius

    translations = []

    i = a / 2
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        y = r * math.sin(-theta + math.pi) * math.sin(-i)
        z = r * - math.cos(theta)

        translations.append((x, y, z))
        i += a / (2 * num_steps)

    return np.array(translations)

def create_spheric_poses(radius, n_poses=50):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,0.3*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_z = lambda phi : np.array([
            [np.cos(phi),-np.sin(phi),0,0],
            [np.sin(phi),np.cos(phi),0,0],
            [0,0, 1,0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])
        c2w =  rot_theta(theta) @ trans_t(radius) @ rot_phi(phi)
        # c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        # c2w = rot_phi(phi) @ c2w
        return c2w
    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        #spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
        spheric_poses += [spheric_pose(th, -np.pi/15, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    # radii = 0
    radii = 0.005
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


def get_pure_rotation(progress_11: float, max_angle: float = 360):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose

def get_pure_translation(progress_11: float, axis = 'x', max_distance = 2):
    trans_pose = np.eye(4)
    if axis == 'x':
        trans_pose[0, 3] = progress_11 * max_distance
    elif axis == 'y':
        trans_pose[1, 3] = progress_11 * max_distance
    elif axis =='z':
        trans_pose[2, 3] = progress_11 * max_distance
    return trans_pose

def get_transformation_with_duplication_offset(progress, duplication_id: int):
    trans_pose = get_pure_rotation(np.sin(progress * np.pi * 2), max_angle=180)
    offset = 0.05
    if duplication_id > 0:
        trans_pose[0, 3] -= np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] -= 0.2
    else:
        trans_pose[0, 3] += np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] += 0.55
    return trans_pose

def parallel_parking_transform(parallel_park_transform, parallel_park_transform_rotation_cache, parallel_park_transform_translation_cache, idx, total_frames, max_angle, max_distance_1 = 2, max_distance_2 = 1):
    
    t1 = 50
    t2 = 70
    t3 = total_frames
    
    if idx <=t1:
        progress = idx/total_frames
        parallel_park_transform_translation = get_pure_translation(progress_11=(progress * 2 - 1), axis='x', max_distance = max_distance_1)
        parallel_park_transform = parallel_park_transform_translation
        parallel_park_transform_translation_cache = parallel_park_transform_translation
        parallel_park_transform_rotation_cache = None

    elif idx > t1 and idx <=t2:
        progress_rotation = (idx-t1) / (t2-t1)
        parallel_park_transform_rotation = get_pure_rotation(progress_11=(-progress_rotation), max_angle= max_angle)
        parallel_park_transform_rotation = parallel_park_transform_rotation @ parallel_park_transform_translation_cache
        parallel_park_transform_rotation_cache = parallel_park_transform_rotation
        parallel_park_transform = parallel_park_transform_rotation

    elif idx > t2 and idx < t3:
        progress_translation = (idx-t2) / (t3-t2)
        parallel_park_transform_translation2 = get_pure_translation(progress_11=(progress_translation), axis='y', max_distance = max_distance_2)
        parallel_park_transform_translation2 = parallel_park_transform_translation2 @ parallel_park_transform_rotation_cache
        parallel_park_transform = parallel_park_transform_translation2

    return parallel_park_transform, parallel_park_transform_rotation_cache, parallel_park_transform_translation_cache

def get_scale_offset(progress, duplication_cnt):
    trans_pose = get_pure_rotation(np.sin(progress * np.pi * 2), max_angle=20)
    # offset = 0.05
    # trans_pose = np.eye(4)

    if duplication_cnt==0:
        trans_pose[:3,:3] *= 0.17
        trans_pose[1, 3] -= 0.075
        trans_pose[0, 3] += 0.03
        trans_pose[2, 3] -= 0.07
    elif duplication_cnt ==1:
        trans_pose[:3,:3] *= 0.16
        trans_pose[1, 3] -= 0.08
        trans_pose[0, 3] += 0.06
        trans_pose[2, 3] -= 0.09
    elif duplication_cnt ==2:
        trans_pose[:3,:3] *= 0.16
        trans_pose[1, 3] -= 0.075
        trans_pose[0, 3] += 0.0
        trans_pose[2, 3] -= 0.05
    # elif duplication_cnt ==2:
    #     trans_pose[:3,:3] *= 0.15
    #     trans_pose[1, 3] -= 0.06
    #     trans_pose[0, 3] += 0.07

    # else:
    #     trans_pose[0, 3] += np.sin(progress * np.pi * 2) * offset
    #     trans_pose[1, 3] += 0.55


    return trans_pose
