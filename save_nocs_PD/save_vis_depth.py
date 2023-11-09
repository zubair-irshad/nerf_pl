import os
import numpy as np
import cv2

import matplotlib.pyplot as plt

base_dir = '/home/zubairirshad/pd-api-py/data/PDMultiObj_Single_Scene/SF_6thAndMission_medium2'
base_dir_train = os.path.join(base_dir, 'train')

img_files = os.listdir(os.path.join(base_dir_train, 'rgb'))
img_files.sort()


for i, img_file in enumerate(img_files):
    if i!=136:
        continue
    depth_endpath = img_file.split('.')[0]+'.npz'
    depth_path = os.path.join(base_dir_train, 'depth', depth_endpath)

    depth = np.clip(np.load(depth_path, allow_pickle=True)['arr_0'], 0,100)

    min_depth = np.min(depth)
    max_depth = np.max(depth)
    depth_img = (depth - np.min(depth)) / (max(np.max(depth) - np.min(depth), 1e-8))
    depth_img = cv2.applyColorMap((depth_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    plt.imshow(depth_img)
    plt.show()
