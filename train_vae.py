import torch
import torch.nn as nn
import torch.utils.data
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image

import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

#import Helpers as hf

class VGG19(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """

    def __init__(self, channel_in=3, width=64):
        super(VGG19, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.load_params_()

    def load_params_(self):
        # Download and load Pytorch's pre-trained weights
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

    def feature_loss(self, x):
        return (x[:x.shape[0] // 2] - x[x.shape[0] // 2:]).pow(2).mean()

    def forward(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        x = self.conv1(x)
        loss = self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        loss += self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        loss += self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.feature_loss(x)

        return loss/16

def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_out // 2) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_in // 2) + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        first_out = channel_in // 2 if channel_in == channel_out else (channel_in // 2) + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))

        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]

        # If channel_in == channel_out we do a simple identity skip
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]

        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:
                # Add an additional non down-sampling block before down-sampling
                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, norm_type=norm_type))

            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))

        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        if self.training or sample:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]

        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)

        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type))

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, norm_type=norm_type))

        self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        return torch.tanh(self.conv_out(x))


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in=3, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, log_var
    
    
def KL_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


class ImageOnlyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Params")
    # string args
    parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
    parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=False, default='/home/liuke/data/Dataset/celeba/img/train')

    parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
    parser.add_argument("--norm_type", "-nt",
                        help="Type of normalisation layer used, BatchNorm (bn) or GroupNorm (gn)", type=str, default="bn")

    # int args
    parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
    parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=128)
    parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
    parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)

    parser.add_argument("--num_res_blocks", '-nrb',
                        help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

    parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
    parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=256)
    parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
    parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                        type=int, nargs='+', default=(1, 2, 4, 8))
    # float args
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
    parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

    # bool args
    parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
    parser.add_argument("--deep_model", '-dm', action='store_true',
                        help="Deep Model adds an additional res-identity block to each down/up sampling stage")

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device_index if use_cuda else "cpu")
    print("")

    # Create dataloaders
    # This code assumes there is no pre-defined test/train split and will create one for you
    print("-Target Image Size %d" % args.image_size)
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Get a test image batch from the test_loader to visualise the reconstruction quality etc
    dataiter = iter(test_loader)
    test_images = next(dataiter)

    # Create AE network.
    vae_net = VAE(channel_in=test_images.shape[1],
                ch=args.ch_multi,
                blocks=args.block_widths,
                latent_channels=args.latent_channels,
                num_res_blocks=args.num_res_blocks,
                norm_type=args.norm_type,
                deep_model=args.deep_model).to(device)

    # Setup optimizer
    optimizer = optim.Adam(vae_net.parameters(), lr=args.lr)

    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

    if args.norm_type == "bn":
        print("-Using BatchNorm")
    elif args.norm_type == "gn":
        print("-Using GroupNorm")
    else:
        ValueError("norm_type must be bn or gn")

    # Create the feature loss module if required
    if args.feature_scale > 0:
        feature_extractor = VGG19().to(device)
        print("-VGG19 Feature Loss ON")
    else:
        feature_extractor = None
        print("-VGG19 Feature Loss OFF")

    # Let's see how many Parameters our Model has!
    num_model_params = 0
    for param in vae_net.parameters():
        num_model_params += param.flatten().shape[0]

    print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
    fm_size = args.image_size//(2 ** len(args.block_widths))
    print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))

    # Create the save directory if it does not exist
    if not os.path.isdir(args.save_dir + "/Models"):
        os.makedirs(args.save_dir + "/Models")
    if not os.path.isdir(args.save_dir + "/Results"):
        os.makedirs(args.save_dir + "/Results")

    # Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
    # If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
    # it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
    save_file_name = args.model_name + "_" + str(args.image_size)
    if args.load_checkpoint:
        if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
            checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                                    map_location="cpu")
            print("-Checkpoint loaded!")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            vae_net.load_state_dict(checkpoint['model_state_dict'])

            if not optimizer.param_groups[0]["lr"] == args.lr:
                print("Updating lr!")
                optimizer.param_groups[0]["lr"] = args.lr

            start_epoch = checkpoint["epoch"]
            data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
        else:
            raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")
    else:
        # If checkpoint does exist raise an error to prevent accidental overwriting
        if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
            raise ValueError("Warning Checkpoint exists -> add -cp flag to use this checkpoint")
        else:
            print("Starting from scratch")
            start_epoch = 0
            # Loss and metrics logger
            data_logger = defaultdict(lambda: [])
    print("")

    # Start training loop
    for epoch in trange(start_epoch, args.nepoch, leave=False):
        vae_net.train()
        for i, (images) in enumerate(tqdm(train_loader, leave=False)):
            current_iter = i + epoch * len(train_loader)
            images = images.to(device)
            bs, c, h, w = images.shape

            # We will train with mixed precision!
            with torch.cuda.amp.autocast():
                recon_img, mu, log_var = vae_net(images)

                kl_loss = KL_loss(mu, log_var)
                mse_loss = F.mse_loss(recon_img, images)
                loss = args.kl_scale * kl_loss + mse_loss

                # Perception loss
                if feature_extractor is not None:
                    feat_in = torch.cat((recon_img, images), 0)
                    feature_loss = feature_extractor(feat_in)
                    loss += args.feature_scale * feature_loss
                    data_logger["feature_loss"].append(feature_loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()

            # Log losses and other metrics for evaluation!
            data_logger["mu"].append(mu.mean().item())
            data_logger["mu_var"].append(mu.var().item())
            data_logger["log_var"].append(log_var.mean().item())
            data_logger["log_var_var"].append(log_var.var().item())

            data_logger["kl_loss"].append(kl_loss.item())
            data_logger["img_mse"].append(mse_loss.item())

            # Save results and a checkpoint at regular intervals
            if (current_iter + 1) % args.save_interval == 0:
                # In eval mode the model will use mu as the encoding instead of sampling from the distribution
                vae_net.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        # Save an example from testing and log a test loss
                        recon_img, mu, log_var = vae_net(test_images.to(device))
                        data_logger['test_mse_loss'].append(F.mse_loss(recon_img,
                                                                    test_images.to(device)).item())

                        img_cat = torch.cat((recon_img.cpu(), test_images), 2).float()
                        vutils.save_image(img_cat,
                                        "%s/%s/%s_%d_test.png" % (args.save_dir,
                                                                    "Results",
                                                                    args.model_name,
                                                                    args.image_size),
                                        normalize=True)

                    # Keep a copy of the previous save in case we accidentally save a model that has exploded...
                    if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                        shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt",
                                        dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

                    # Save a checkpoint
                    torch.save({
                                'epoch': epoch + 1,
                                'data_logger': dict(data_logger),
                                'model_state_dict': vae_net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, args.save_dir + "/Models/" + save_file_name + ".pt")

                    # Set the model back into training mode!!
                    vae_net.train()




'''
为更大的图像定义更深层的架构：
示例展示了如何通过改变块的数量来改变所使用的图像大小 (128x128)，同时保持相同的潜在表示 (256x4x4)。
python train_vae.py -mn test_run --image_size 128 --block_widths 1 2 4 4 8 --dataset_root #path to dataset root#
使用 128x128 图像和更深的模型进行训练，方法是在每个下采样/上采样阶段添加 Res-Identity Blocks，而无需额外的下采样。

python train_vae.py -mn test_run --image_size 128 --deep_model  --latent_channels 64 --dataset_root #path to dataset root#
潜在空间表示将为 64x8x8，潜在变量的数量与之前相同，但形状不同！
'''