import os
import numpy

dir = '/home/zubair/compositional_demo_latent/camera'

folders = os.listdir(dir)

for folder in folders:
    depth_path = os.path.join(dir, folder, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path, exist_ok=True)
        os.chdir(os.path.join(dir, folder))
        os.system("mv *depth.png depth/")
        os.system("gifski --fps 25 -o file.gif render_*.png")        
        os.system("ffmpeg -framerate 25 -i render_%04d.png output.mp4")
        os.chdir(depth_path)
        os.system("gifski --fps 25 -o file.gif render_*.png")   
        os.system("ffmpeg -framerate 25 -i render_%04d_depth.png output.mp4")