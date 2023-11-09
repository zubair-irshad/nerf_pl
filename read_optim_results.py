import argparse
import torch
import os
import numpy as np
import pickle

if __name__ == '__main__':
    scene_name = 'chair_optimize_autodecoder'
    optim_results_path = '/home/ubuntu/nerf_pl/results/srn_multi/' +scene_name +'/codes.pth'
    results_dict = torch.load(optim_results_path)

    psnr_eval = results_dict['psnr_eval']
    ssim_eval = results_dict['ssim_eval']

    if 'psnt_init' in results_dict:
        psnr_init = results_dict['psnt_init']
        ssim_init = results_dict['ssim_init']

    mean_psnr_per_object = []
    for k,v in psnr_eval.items():
        mean_psnr_per_object.append(np.mean(v))
        # print("np.mean(v) psnr", np.mean(v))

    print("mean psnr", np.mean(mean_psnr_per_object))

    mean_ssim_per_object = []
    for k,v in ssim_eval.items():
        mean_ssim_per_object.append(np.mean(v))
        # print("np.mean(v) ssim", np.mean(v))
    print("mean ssim", np.mean(mean_ssim_per_object))

    psnr_init_per_object = []
    if 'psnt_init' in results_dict:
        for k,v in psnr_init.items():
            print("k,v", k,v)
            psnr_init_per_object.append(v.cpu().numpy())

        print("mean psnr init", np.mean(psnr_init_per_object))

    ssim_init_per_object = []
    if 'ssim_init' in results_dict:
        for k,v in ssim_init.items():
            ssim_init_per_object.append(v)

        print("mean ssim_init", np.mean(ssim_init_per_object))