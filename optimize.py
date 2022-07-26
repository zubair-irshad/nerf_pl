import os
import cv2

from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
from datasets import dataset_dict
from models.rendering import render_rays, render_rays_conditional
from models.nerf import *
from losses import loss_dict
from tqdm import tqdm

from utils import load_ckpt, load_latent_codes
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
from metrics import *

torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compute_ssim

# optimizer, scheduler, visualization
from utils import *
import wandb
wandb.init(project="nerf_pl")


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'google_scanned', 'srn', 'srn_multi'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--splits', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--crop_img', default=False, action="store_true",
                        help='whether to save depth prediction')
    # parser.add_argument('--latent_code_path', type=str, required=True,
    #                     help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')
    parser.add_argument('--latent_lr', type=float, default=1.0e-2,
                        help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--cat', type=str, default=None,
                        help='which category to use')

    parser.add_argument('--decay_step', nargs='+', type=int, default=[100],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay amount')
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')

    return parser.parse_args()

def optimize_batch_inference(models, embeddings, shape_code, texture_code,
                      rays, N_samples, N_importance, use_disp,
                      chunk, white_back):
    """Do batched inference on rays using chunk."""

    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays_conditional(models,
                        embeddings,
                        rays[i:i+chunk],
                        shape_code,
                        texture_code,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    return results


@torch.no_grad()
def batched_inference(models, embeddings, shape_code, texture_code,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays_conditional(models,
                        embeddings,
                        rays[i:i+chunk],
                        shape_code,
                        texture_code,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        test_dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def save_opts(num_obj, data_dir, optimized_shapecodes, optimized_texturecodes, psnr_eval, ssim_eval, psnr_init, ssim_init):
    saved_dict = {
        'num_obj' : num_obj,
        'optimized_shapecodes' : optimized_shapecodes,
        'optimized_texturecodes': optimized_texturecodes,
        'psnr_eval': psnr_eval,
        'ssim_eval': ssim_eval,
        'psnt_init' : psnr_init,
        'ssim_init' : ssim_init
    }
    torch.save(saved_dict, os.path.join(data_dir, 'codes.pth'))
    print('We finished the optimization of ' + str(num_obj))


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    psnr_init = {}
    ssim_init = {}

    psnr_eval = {}
    ssim_eval = {}

    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    nerf_coarse = ConditionalNeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=6*args.N_emb_dir+3)
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    loss_opt = loss_dict['color'](coef=1)

    if args.N_importance > 0:
        nerf_fine = ConditionalNeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                         in_channels_dir=6*args.N_emb_dir+3)
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine

    shape_codes, texture_codes = load_latent_codes(args.ckpt_path)
    mean_shape = torch.mean(shape_codes, dim=0).reshape(1,-1)
    mean_texture = torch.mean(texture_codes, dim=0).reshape(1,-1)


    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(data_dir = args.root_dir, splits=args.splits, cat = args.cat,
                                 img_wh = args.img_wh, crop_img = args.crop_img)

    test_dataloader =  DataLoader(test_dataset,
                            shuffle=False,
                            num_workers=4,
                            batch_size=1,
                            pin_memory=True)

    imgs, depth_maps, psnrs = [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    optimized_shapecodes = torch.zeros(len(test_dataloader), mean_shape.shape[1])
    optimized_texturecodes = torch.zeros(len(test_dataloader), mean_texture.shape[1])

    print("optimized shapecodes", optimized_shapecodes.shape, optimized_texturecodes.shape)

    for i in tqdm(range(len(test_dataloader))):
        sample = test_dataset[i]
        rays_all = sample['rays']
        rgbs_all = sample['rgbs']

        rays, rgbs = rays_all[0].cuda(), rgbs_all[0].cuda()
        # print("rays, rgbs", rays.shape, rgbs.shape)
        #optimize here:
        shape_code = mean_shape.cuda().clone().detach().requires_grad_()
        texture_code = mean_texture.cuda().clone().detach().requires_grad_()
        optimizer_latent = get_optimizer_latent_opt(args, shape_code, texture_code)
        scheduler_latent = get_scheduler(args, optimizer_latent)
        
        num_opts = 200
        nopts = 0
        print("Optimizing object num...", i)
        for opt_num in tqdm(range(200)):
        # while nopts < num_opts:
            pred_image = {}
            # pred_image['rgb_coarse'] = torch.empty((rgbs.shape)).type_as(rgbs)
            # if args.N_importance>0:
            #     pred_image['rgb_fine'] = torch.empty((rgbs.shape)).type_as(rgbs)

            indices = torch.randperm(rgbs.shape[0])
            rgbs = rgbs[indices].float()
            rays = rays[indices]
            batch_size = 4096

            # for i in range(0, rays.shape[0], batch_size):
            results = optimize_batch_inference(models, embeddings, shape_code, texture_code,
                            rays, args.N_samples, args.N_importance, args.use_disp,
                            args.chunk, test_dataset.white_back)
            
            # print("results['rgb_coarse']", results['rgb_coarse'].shape)
            
            pred_image['rgb_coarse'] = results['rgb_coarse']
            loss_img = loss_opt(pred_image, rgbs)
            reg_loss = torch.norm(shape_code, dim=0) + torch.norm(texture_code, dim=0)
            loss_reg = 1e-4 * torch.mean(reg_loss)
            loss = loss_img + loss_reg

            optimizer_latent.zero_grad()
            loss.backward()
            optimizer_latent.step()

            with torch.no_grad():
                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                psnr_ = psnr(pred_image[f'rgb_{typ}'], rgbs)

                if opt_num ==0:
                    ssim_ = compute_ssim(pred_image[f'rgb_{typ}'].cpu().numpy(), rgbs.cpu().numpy(), multichannel=True)
                    ssim_init[i] = ssim_
                    psnr_init[i] = psnr_
            
            

            
            wandb.log({'opt/loss_img': loss_img})
            wandb.log({'opt/loss_latent': loss_reg})
            wandb.log({'opt/psnr': psnr_})
            nopts+=1

        # now evaluate
        evaluate_num = 50
        # not considering 0 here since that was used for optimization

        dir_name_img = dir_name + str(i)
        os.makedirs(dir_name_img, exist_ok=True)
        os.makedirs(dir_name, exist_ok=True)
        for eval_num in tqdm(range(1, evaluate_num)):
            rays, img_gt = rays_all[eval_num].cuda(), rgbs_all[eval_num]
            rays = rays.cuda()
            results = batched_inference(models, embeddings, shape_code, texture_code, rays,
                                        args.N_samples, args.N_importance, args.use_disp,
                                        args.chunk)
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)

            img_gt = img_gt.view(h, w, 3)
            if eval_num == 1:
                psnr_eval[i] = [metrics.psnr(img_gt, img_pred).item()]
            else: 
                psnr_eval[i].append(metrics.psnr(img_gt, img_pred).item())
            img_pred_np = img_pred
            img_gt_np = img_gt.numpy()
            ssim_ev = compute_ssim(img_pred_np, img_gt_np, multichannel=True)
            if eval_num == 1:
                ssim_eval[i] = [ssim_ev]
            else:
                ssim_eval[i].append(ssim_ev)
                
        print("psnr init[i], sssim init[i]", psnr_init[i], ssim_init[i])
        print("psnr_eval[i]", np.mean(psnr_eval[i]), "ssim_eval[i]", np.mean(ssim_eval[i]) )

        wandb.log({'psnr/eval': np.mean(psnr_eval[i])})
        wandb.log({'ssim/eval': np.mean(ssim_eval[i])})

        # Save the optimized codes
        optimized_shapecodes[i] = shape_code.detach().cpu()
        optimized_texturecodes[i] = texture_code.detach().cpu()
        save_opts(i, dir_name, optimized_shapecodes, optimized_texturecodes, psnr_eval, ssim_eval, psnr_init, ssim_init)


