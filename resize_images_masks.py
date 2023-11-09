import os
import cv2
from PIL import Image
import numpy as np

images_folder = '/home/zubair/camera_batch-2_1/images'
masks_folder = '/home/zubair/camera_batch-2_1/masks'

images_folder_out = '/home/zubair/camera_batch-2_1/images_12'
masks_folder_out = '/home/zubair/camera_batch-2_1/masks_12'

os.makedirs(images_folder_out, exist_ok=True)
os.makedirs(masks_folder_out, exist_ok=True)

images_path = os.listdir(images_folder)
masks_path = os.listdir(masks_folder)

for image_name, seg_name in zip(images_path, masks_path):
    print("os.path.join(images_folder, image_name)", os.path.join(images_folder, image_name))
    img = Image.open(os.path.join(images_folder, image_name))
    img = img.resize((120,160), Image.LANCZOS)
    seg_mask = cv2.imread(os.path.join(masks_folder, seg_name), cv2.IMREAD_GRAYSCALE)

    seg_mask_filepath = os.path.join(masks_folder_out, seg_name)
    seg_mask = cv2.resize(seg_mask, (120,160), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(seg_mask_filepath, seg_mask)

    img_filepath = os.path.join(images_folder_out, image_name)
    cv2.imwrite(img_filepath, np.array(img)[...,::-1])
#seg_mask = cv2.resize(seg_mask, (120,160), interpolation=cv2.INTER_NEAREST)
