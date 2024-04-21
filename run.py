import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms 
from torch.utils.data import DataLoader
from dataloader import CustomImageDataset
from utils import linear_beta_schedule, get_value_at_time_t, save_plot
from model import UnetArchitecture
from tqdm import tqdm
import glob
import os

parser = argparse.ArgumentParser(description = "Denoising Diffusion Model in Pytorch")
parser.add_argument('-lr', '--learning_rate', default = 4e-3)
parser.add_argument('-dim','--dim', default=128)
parser.add_argument('-ep', '--epoch', default = 20)
parser.add_argument('-rl', '--reload_epoch', default = 0)
parser.add_argument('-m', '--mode', required=True)
parser.add_argument('-d', '--device', default="cuda")
args = parser.parse_args()
device = args.device

LR = args.learning_rate
DIM = int(args.dim)
EPOCH = int(args.epoch)
RELOAD = int(args.reload_epoch)
BATCH_SIZE = 1
MODE = args.mode
transform_train = A.Compose([A.Resize(height=DIM, width=DIM),
                                    ToTensorV2(),
                                ])

model = UnetArchitecture()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=3, gamma=0.05)
training_data = CustomImageDataset(transform=transform_train, mode="train", device=device)
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

betas = linear_beta_schedule()
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def forward_noise_sampler(x_0, t, device):
    noise = torch.rand_like(x_0)
    sqrt_alphas_comprod_t = get_value_at_time_t(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_value_at_time_t(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    output = sqrt_alphas_comprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    return output, noise.to(device)

def get_loss_value(model, x_0, t, mode="train"):
    if mode=="train":
        model.train()
    x_plus_noise, noise = forward_noise_sampler(x_0, t, device)
    noise_pred = model(x_plus_noise, t)
    return F.l1_loss(noise, noise_pred)

def train():
    epoch_tr_loss = []
    min_train_loss = 1000
    for ep in range(RELOAD,EPOCH):
        with tqdm(train_dataloader, unit=" Train batch") as tepoch:
            tepoch.set_description(f"Train Epoch {ep+1}")
            batch_train_loss = []
            for image, img_path in tepoch:
                optimizer.zero_grad()
                t = torch.randint(0, 300, (BATCH_SIZE,), device=device).long()
                loss = get_loss_value(model, image.float(), t, "train")
                loss.backward()
                optimizer.step()
                batch_train_loss.append(loss.item())
        scheduler.step()
        ep_loss = sum(batch_train_loss)/(len(batch_train_loss)+0.0000001)
        epoch_tr_loss.append(ep_loss)
        if ep_loss < min_train_loss:
            min_train_loss = ep_loss
            os.system("rm ./weights/unet*")
            torch.save(model.state_dict(), "./weights/unet_"+str(ep+1)+".pt")
            print("saved model weights at epoch: ",ep+1)
    return epoch_tr_loss


if MODE == "train":
    epoch_tr_loss_list = train()
    save_plot(epoch_tr_loss_list)
    # image without noise = model input image - model output image

    