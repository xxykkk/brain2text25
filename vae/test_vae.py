import torch
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image

import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

from model import *
import kornia
from lpips import LPIPS

os.environ['CUDA_VISIBLE_DEVICES']= '0'


parser = argparse.ArgumentParser(description="Testing Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=False, default='/home/liuke/data/Dataset/celeba/img/train')

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--norm_type", "-nt",
                    help="Type of normalisation layer used, BatchNorm (bn) or GroupNorm (gn)", type=str, default="bn")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=32)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=128)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)

parser.add_argument("--num_res_blocks", '-nrb',
                    help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=256)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 4, 8))
# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_path", '-lp', type = str, default = '/home/liuke/data/Kaggle/brain2text/vae/Models/test_run_128.pt')
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--deep_model", '-dm', action='store_true',
                    help="Deep Model adds an additional res-identity block to each down/up sampling stage")
args = parser.parse_args()



device = torch.device('cuda')
lpips_vgg=LPIPS(net='vgg').to(device)
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

data_set = ImageOnlyDataset(args.dataset_root, transform=transform)
#data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)

# Randomly split the dataset with a fixed random seed for reproducibility
test_split = 0.9
n_train_examples = int(len(data_set) * test_split)
n_test_examples = len(data_set) - n_train_examples
train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                    generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
dataiter = iter(test_loader)
test_images = next(dataiter)



vae_net = VAE(channel_in=test_images.shape[1],
                ch=args.ch_multi,
                blocks=args.block_widths,
                latent_channels=args.latent_channels,
                num_res_blocks=args.num_res_blocks,
                norm_type=args.norm_type,
                deep_model=args.deep_model).to(device)

save_file_name = args.model_name + "_" + str(args.image_size)
checkpoint = torch.load(args.load_path)
vae_net.load_state_dict(checkpoint['model_state_dict'])

# if args.load_checkpoint:
    # if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
    #     checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
    #                             map_location="cpu")
    #     print("-Checkpoint loaded!")
    #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     vae_net.load_state_dict(checkpoint['model_state_dict'])

    #     # if not optimizer.param_groups[0]["lr"] == args.lr:
    #     #     print("Updating lr!")
    #     #     optimizer.param_groups[0]["lr"] = args.lr

    #     # start_epoch = checkpoint["epoch"]
    #     data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
    # else:
    #     raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")

def get_metric(images, wm_images, messages = None, out_messages = None):
    psnr = kornia.losses.psnr_loss(wm_images.detach(), images, max_val=2)
    ssim = 1 - 2 * kornia.losses.ssim_loss(wm_images.detach(), images, max_val=2.0, window_size=5, reduction="mean")
    dist=lpips_vgg(wm_images.detach(), images).mean()
    #acc_rate = decoded_message_acc_rate_bit(messages, out_messages)
    #acc_batch = decoded_message_acc_rate_batch(messages, out_messages)
    diffH = ((wm_images*0.5+0.5)-(images*0.5+0.5)).abs().mean() * 255
    #diffR = ((recover*0.5+0.5)-(serect_image*0.5+0.5)).abs().mean() * 255
    
    #js_div = compute_js_divergence(wm_images.detach(), images)
    #wass_dist = compute_wasserstein(wm_images.detach(), images)
    #fid = compute_fid(wm_images.detach(), images)
    #psnr_piq = piq.psnr(wm_images.detach()*0.5+0.5, images*0.5+0.5, data_range=1.0)
    #ssim_piq = piq.ssim(wm_images.detach()*0.5+0.5, images*0.5+0.5, data_range=1.0, reduction='mean')
    metric = {
            #"acc_rate": acc_rate,
            #"acc_batch": acc_batch,
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": dist.item(),

            "diffH": diffH.item(),

            #"js_div": js_div.item(),
            #"wass_dist": wass_dist.item(),
            #"fid": fid.item(),
            #"psnr_piq": psnr_piq.item(),
            #"ssim_piq": ssim_piq.item(),
        }
    return metric

vae_net.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast():
        # Save an example from testing and log a test loss
        recon_img, mu, log_var = vae_net(test_images.to(device))
        #print(recon_img.shape, test_images.shape)

        #print(recon_img.max(), recon_img.min(), test_images.max(), test_images.min())
        #input("xx")

        img_cat = torch.cat((recon_img.cpu(), test_images), 2).float()
        #print(img_cat.shape)
        met = get_metric(recon_img, test_images.to(device))

        vutils.save_image(img_cat,
                        "%s/%s/%s_%d_%.2f_%.2f_test.png" % (args.save_dir,
                                                    "Results",
                                                    args.model_name,
                                                    args.image_size,
                                                    -met['psnr'], met['diffH']
                                                    ),
                        normalize=True)
        
        print(met)

