import os

import imageio
import numpy as np
from PIL import Image
import json

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def store_image(dirpath, rgbs):
    for (i, rgb) in enumerate(rgbs):
        imgname = f"image{str(i).zfill(3)}.jpg"
        rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
        imgpath = os.path.join(dirpath, imgname)
        rgbimg.save(imgpath)


def store_video(dirpath, rgbs, depths):
    rgbimgs = [to8b(rgb.cpu().detach().numpy()) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, "images.mp4"), rgbimgs, fps=20, quality=8)


def write_stats(fpath, *stats):

    d = {}
    for stat in stats:
        d[stat["name"]] = {
            k: float(w)
            for (k, w) in stat.items()
            if k != "name" and k != "scene_wise"
        }

    with open(fpath, "w") as fp:
        json.dump(d, fp, indent=4, sort_keys=True)