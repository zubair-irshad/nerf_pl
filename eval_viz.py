import os
import cv2

from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays, render_rays_conditional
from models.nerf import *

from utils import load_ckpt, load_latent_codes
import metrics
import pickle

from datasets import dataset_dict
from datasets.depth_utils import *
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


torch.backends.cudnn.benchmark = True


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
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--cat', type=str, required=True,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--crop_img', default=False, action="store_true",
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    # parser.add_argument('--latent_code_path', type=str, required=True,
    #                     help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=True, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    return parser.parse_args()

def scatter(x, colors):
    palette = np.array(sns.color_palette("hls", 6))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=colors)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('tight')
    ax.set_xlabel('t-SNE-1')
    ax.set_ylabel('t-SNE-2')

    txts = None
    lg = None

    return f, ax, sc, txts,lg

def generate_masked_tensor(input, mask, fill=0):
    masked_tensor = np.zeros((input.shape)) + fill
    masked_tensor[mask] = input[mask]
    return masked_tensor

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
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    # kwargs = {'root_dir': args.root_dir,
    #           'split': args.split,
    #           'img_wh': tuple(args.img_wh)}
    # if args.dataset_name == 'llff':
    #     kwargs['spheric_poses'] = args.spheric_poses
    kwargs = {'data_dir': args.root_dir,
              'splits': args.split,
              'cat': args.cat,
              'img_wh': tuple(args.img_wh),
              'crop_img': args.crop_img,
              'num_instances_per_obj': 50}

    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    nerf_coarse = ConditionalNeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=6*args.N_emb_dir+3)
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if args.N_importance > 0:
        nerf_fine = ConditionalNeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                         in_channels_dir=6*args.N_emb_dir+3)
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine

    latent_dict = {}
    shape_codes, texture_codes = load_latent_codes(args.ckpt_path)
    latent_dict['shape_code'] = shape_codes
    latent_dict['texture_code'] = texture_codes
    latent_code_save_path = args.ckpt_path +'_latent_dict'
    f = open(latent_code_save_path,"wb")
    pickle.dump(latent_dict,f)
    f.close()

    plot_latent = True
    if plot_latent:
        sns.palplot(np.array(sns.color_palette("hls", 18)))
        print("plotting shape latent...")
        colors = []
        for _ in range(shape_codes.shape[0]):
            colors.append(np.array([0.2668, 0.2637, 0.2659]))
        RS = 20150101
        
        kmeans = KMeans(n_clusters=4, random_state=0).fit(shape_codes.numpy())
        Y=kmeans.labels_ # a vector
        tsne = TSNE(random_state=RS).fit_transform(shape_codes.numpy())

        f, ax, sc, txts,lg = scatter(tsne, Y)
        plt.savefig('shape_latent_auto_decoder', dpi=120, bbox_inches='tight')
        plt.close()

        print("plotting texture latent...")
        RS = 20150101
        kmeans = KMeans(n_clusters=5, random_state=0).fit(texture_codes.numpy())
        Y=kmeans.labels_ # a vector
        tsne = TSNE(random_state=RS).fit_transform(texture_codes.numpy())

        f, ax, sc, txts,lg = scatter(tsne, Y)
        plt.savefig('texture_latent_auto_decoder', dpi=120, bbox_inches='tight')
        plt.close()
        
    print("shape_codes", shape_codes.shape)
    imgs, depth_maps, psnrs = [], [], []
    valid_masks = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    
    evaluate_num = 1
    for j in tqdm(range(evaluate_num)):
        dir_name = dir_name + str(j)
        os.makedirs(dir_name, exist_ok=True)
        sample = dataset[j]
        obj_idx = sample['obj_id']
        print(obj_idx)
        shape_code, texture_code = shape_codes[torch.tensor(25)].cuda(), texture_codes[obj_idx].cuda()
        print(len(sample['rays']))
        for i, rays in enumerate(sample['rays']):
            rays = rays.cuda()
            texture_code = texture_codes[torch.tensor(i)].cuda()
            results = batched_inference(models, embeddings, shape_code, texture_code, rays,
                                        args.N_samples, args.N_importance, args.use_disp,
                                        args.chunk)
            typ = 'fine' if 'rgb_fine' in results else 'coarse'

            img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)
            if args.save_depth:
                depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
                depth_maps += [depth_pred]
                if args.depth_format == 'pfm':
                    save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
                else:
                    with open(os.path.join(dir_name, f'depth_{i:03d}'), 'wb') as f:
                        f.write(depth_pred.tobytes())
            # print(img_pred.shape)
            # print(img_pred)
            valid_masks += [(img_pred.sum(2)<3)]
            img_pred_ = (img_pred * 255).astype(np.uint8)
            imgs += [img_pred_]
            # print(valid_masks[0])
            imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

            if 'rgbs' in sample:
                rgbs = sample['rgbs'][i]
                img_gt = rgbs.view(h, w, 3)
                psnrs += [metrics.psnr(img_gt, img_pred).item()]

    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=5)

    if args.save_depth:
        min_depth = np.min(depth_maps)
        max_depth = np.max(depth_maps)
        depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
        depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
        print("depth imgs",depth_imgs_[0].shape)
        print("depth imgs masks",depth_imgs_[0][valid_masks[0]].shape)
        print(depth_imgs[0])
        depth_imgs_ = [generate_masked_tensor(img, mask) for img, mask in zip(depth_imgs,valid_masks)]

        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_depth.gif'), depth_imgs_, fps=5)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')

