#!/bin/sh
CUDA_VISIBLE_DEVICES=4,5,6,7
python nerf_pl/run.py 
--dataset_name pd_multi 
--root_dir data/PDStep/v1 
--exp_name refnerf_autodecoder 
--num_epochs 500 
--img_wh 320 240 
--white_back 
--N_obj_code_length 128 
--N_max_objs 200 
--num_gpus 8