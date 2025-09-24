import os
import sys
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, UnetRes, set_seed)
from MCESS_UNet import U_Net
from dataset import myImageFlodertrain, myImageFloderval
from pathlib import Path
from tqdm import tqdm

# 初始化
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.stdout.flush()
set_seed(10)

# 设置路径
train_folder = "../data/RSID/train"
val_folder = "../data/RSID/test"

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练集和测试集
train_dataset = myImageFlodertrain(root=train_folder, transform=transform, crop=False, resize=False)
val_dataset = myImageFloderval(root=val_folder, transform=transform, resize=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# 定义第一阶段的 U-Net 模型，用于预处理
unet_stage1 = U_Net(dim=64).to('cuda')
unet_stage1.load_state_dict(torch.load("generator_epoch_91.pth"))
unet_stage1.eval()

# 定义第二阶段的残差去噪扩散模型
diffusion = ResidualDiffusion(
    model=UnetRes(dim=64, dim_mults=(1, 2, 4, 8), condition=True, input_condition=False),
    image_size=256,
    timesteps=1000,
    sampling_timesteps=5,
    objective='pred_res',
    loss_type='l1',
    condition=True,
    sum_scale=1,
    input_condition=False,
    input_condition_mask=False
)


class CustomTrainer(Trainer):
    def __init__(self, diffusion_model, unet_stage1, train_loader, val_loader, train_folder, val_folder,
                 train_lr, train_num_steps, save_and_sample_every, num_samples, results_folder,
                 pretrained_weights_path=None, load_pretrained_weights=False):
        super().__init__(diffusion_model=diffusion_model, train_folder=train_folder, val_folder=val_folder,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         save_and_sample_every=save_and_sample_every, num_samples=num_samples,
                         results_folder=results_folder)

        self.unet_stage1 = unet_stage1
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step = 0
        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # 初始化CSV文件，仅保留 Step 和 Loss 两列
        self.loss_log_file = self.results_folder / "loss_log.csv"
        if not self.loss_log_file.exists():
            pd.DataFrame(columns=["Step", "Average Loss"]).to_csv(self.loss_log_file, index=False)

        if load_pretrained_weights and pretrained_weights_path is not None:
            print(f"Attempting to load pretrained weights from {pretrained_weights_path}")
            try:
                checkpoint = torch.load(pretrained_weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model'], strict=False)
                if 'step' in checkpoint:
                    self.step = checkpoint['step']
                print(f"Resumed training from step {self.step}")
            except FileNotFoundError:
                print(f"Pretrained weights file {pretrained_weights_path} not found. Starting from scratch.")
        else:
            print("Starting training from scratch (no pretrained weights loaded).")

    def train(self):
        for epoch in range(self.train_num_steps // len(self.train_loader)):
            epoch_loss = 0
            num_batches = 0

            self.unet_stage1.eval()
            self.model.train()

            for hazy_image, gt_image in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training"):
                hazy_image, gt_image = hazy_image.to(self.device), gt_image.to(self.device)

                with torch.no_grad():
                    preprocessed_image = self.unet_stage1(hazy_image)

                input_data = [gt_image, preprocessed_image, hazy_image]

                loss = self.model(input_data)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                self.step += 1

                if self.step % self.save_and_sample_every == 0:
                    self.save(self.step)
                    avg_epoch_loss = epoch_loss / num_batches
                    self.log_loss(self.step, avg_epoch_loss)

            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

        print("训练完成。")

    def log_loss(self, step, avg_loss):
        df = pd.DataFrame([[step, avg_loss]], columns=["Step", "Average Loss"])
        df.to_csv(self.loss_log_file, mode='a', header=False, index=False)
        print(f"Step {step} - Loss logged: {avg_loss:.4f}")


# 初始化并启动训练
trainer = CustomTrainer(
    diffusion_model=diffusion,
    unet_stage1=unet_stage1,
    train_loader=train_loader,
    val_loader=val_loader,
    train_folder=train_folder,
    val_folder=val_folder,
    train_lr=8e-5,
    train_num_steps=9000000,
    save_and_sample_every=900,
    num_samples=1,
    results_folder='./results_1Konlyxpre',
    pretrained_weights_path='results_1Knew_new/model-14400.pt',
    load_pretrained_weights=False
)

# 启动训练
trainer.train()
