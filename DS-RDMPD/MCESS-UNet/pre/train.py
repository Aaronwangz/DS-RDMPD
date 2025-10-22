import torch.backends.cudnn as cudnn
cudnn.enabled = False  # 禁用cuDNN
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from MCESS_UNet import U_Net
from dataset import myImageFlodertrain, myImageFloderval
import torch.nn as nn
import torch
from PIL import Image
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('ConvBlock') != -1:
        for child in m.children():
            if isinstance(child, torch.nn.Conv2d):
                torch.nn.init.normal_(child.weight.data, 0.0, 0.02)
                if hasattr(child, 'bias') and child.bias is not None:
                    torch.nn.init.constant_(child.bias.data, 0.0)
            elif isinstance(child, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(child.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(child.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def custom_psnr(image1, image2):
    mse = F.mse_loss(image1, image2)
    max_pixel = 1.0
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value.item()


def BatchPSNR(tar_img, prd_img):
    psnr = [custom_psnr(tar, prd) for tar, prd in zip(tar_img, prd_img)]
    return torch.tensor(psnr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def compute_mse_loss(img1, img2):
    return torch.mean((img1 - img2) ** 2)


def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input - output))

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="dehazing", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=6, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=1200, help='size of image height')
parser.add_argument('--img_width', type=int, default=1600, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--data_url', type=str, default="", help='name of the dataset')
parser.add_argument('--init_method', type=str, default="", help='name of the dataset')
parser.add_argument('--train_url', type=str, default="", help='name of the dataset')

opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
EPS = 1e-12

# 定义损失函数
criterion_pixelwise = torch.nn.L1Loss()

patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

img_channel = 3
dim = 32
enc_blks = [2, 2, 2, 4]
middle_blk_num = 6
dec_blks = [2, 2, 2, 2]
generator = U_Net(dim=64)
pytorch_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("Total_params_model: {}M".format(pytorch_total_params / 1000000.0))

if cuda:
    generator = generator.to(device)
    criterion_pixelwise.to(device)

if opt.epoch != 0:
    generator.load_state_dict(torch.load('train_result_NH-003/prebest.pth'), strict=True)
else:
    generator.apply(weights_init_normal)

if cuda and torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200, eta_min=1e-6)

mytransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512,512))
])

train_root = r'E:\Mymodel\data\NH-HAZE\train'
val_roots = [r'E:\Mymodel\data\NH-HAZE\test']

myfolder = myImageFlodertrain(root=train_root, transform=mytransform, crop=False, resize=False)
dataloader = DataLoader(myfolder, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True, persistent_workers=True)
print('data loader finish！')

val_datasets = {os.path.basename(root): myImageFloderval(root=root, transform=mytransform, resize=False) for root in val_roots}
val_dataloaders = {name: DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True) for name, dataset in val_datasets.items()}

def sample_images(epoch, i, real_A, real_B, fake_B):
    data, pred, label = real_A * 255, fake_B * 255, real_B * 255
    data = data.cpu()
    pred = pred.cpu()
    label = label.cpu()
    pred = torch.clamp(pred.detach(), 0, 255)
    data, pred, label = data.int(), pred.int(), label.int()
    h, w = pred.shape[-2], pred.shape[-1]
    img = np.zeros((h, 1 * 3 * w, 3))
    for idx in range(0, 1):
        row = idx * h
        tmplist = [data[idx], pred[idx], label[idx]]
        for k in range(3):
            col = k * w
            tmp = np.transpose(tmplist[k], (1, 2, 0))
            img[row:row + h, col:col + w] = np.array(tmp)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save("./images/%03d_%06d.png" % (epoch, i))

def evaluate_model(generator, val_dataloaders):
    generator.eval()
    total_loss = {name: 0 for name in val_dataloaders}
    count = {name: 0 for name in val_dataloaders}

    criterion = nn.L1Loss()

    for name, dataloader in val_dataloaders.items():
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                real_A, real_B = batch
                real_A, real_B = real_A.to(cuda), real_B.to(cuda)
                fake_B = generator(real_A)
                loss = criterion(fake_B, real_B).mean().item()
                total_loss[name] += loss
                count[name] += 1

    avg_loss = {name: total_loss[name] / count[name] for name in total_loss}
    mean_loss = np.mean(list(avg_loss.values()))
    generator.train()
    return mean_loss, avg_loss

prev_time = time.time()
step = 0
save_dir = 'train_result_NH-003'

csv_file = os.path.join(save_dir, 'training_log.csv')
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Mean_Loss', 'Val_Mean_Loss'])

for epoch in range(opt.epoch, opt.n_epochs):
    epoch_loss = []

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch}/{opt.n_epochs}", unit="batch", colour="white") as pbar:
        for i, batch in enumerate(dataloader):
            step += 1
            real_A, real_B = batch[0].to(device), batch[1].to(device)

            optimizer_G.zero_grad()
            fake_B = generator(real_A)

            loss_pixel = criterion_pixelwise(fake_B, real_B).mean()
            loss_mse = compute_mse_loss(fake_B, real_B).mean()
            p0 = compute_l1_loss(fake_B * 255, real_B * 255).mean() * 2

            loss_G = loss_pixel + loss_mse + p0
            loss_G.backward()
            optimizer_G.step()

            epoch_loss.append(loss_G.item())

            pbar.set_postfix({
                'loss_pixel': f"{loss_pixel.item():.6f}",
                'loss_mse': f"{loss_mse.item():.6f}",
                'p0': f"{p0.item():.6f}",
                'loss_G': f"{loss_G.item():.6f}"
            })
            pbar.update(1)

            if i % 3000 == 0:
                sample_images(epoch, i, real_A, real_B, fake_B)

    mean_loss = np.mean(epoch_loss)
    print(f"Epoch [{epoch}/{opt.n_epochs}], Mean Loss: {mean_loss:.6f}")

    val_mean_loss, val_avg_loss = evaluate_model(generator, val_dataloaders)
    print(f"Epoch [{epoch}/{opt.n_epochs}], Val Mean Loss: {val_mean_loss:.6f}")

    best_model_state = generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict()
    torch.save(best_model_state, os.path.join(save_dir, 'prebest.pth'))

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, mean_loss, val_mean_loss])

    scheduler_G.step()

