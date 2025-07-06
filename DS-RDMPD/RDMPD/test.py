import os
import sys
import torch
from dataset import myImageFlodertest
from src.residual_denoising_diffusion_pytorch import ResidualDiffusion, UnetRes, set_seed
from MCESS_UNet import U_Net
from metrics import ssim, psnr
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image


# 初始化环境
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.stdout.flush()
set_seed(10)

# 设置路径
# test_folder = "../data/RealBlur/RealBlur-J/test"
# results_folder = 'test_result_RealBlur'
test_folder = r'E:\Mymodel\data\NH-HAZE\test'
results_folder = 'test_result_RSID'

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像转换和加载测试集
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])
test_dataset = myImageFlodertest(root=test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# 定义模型的第一阶段和第二阶段
unet_stage1 = U_Net(dim=64).to(device)
unet_stage1.load_state_dict(torch.load(r"E:\Mymodel\diffusemodel\RDDM001\generator_epoch_91.pth", map_location=device))
unet_stage1.eval()

# 定义扩散模型
diffusion = ResidualDiffusion(
    model=UnetRes(dim=64, dim_mults=(1, 2, 4, 8), condition=True, input_condition=True),
    image_size=256,
    timesteps=1000,
    sampling_timesteps=5,
    objective='pred_res',
    loss_type='l1',
    condition=True,
    sum_scale=1,
    input_condition=True,
    input_condition_mask=False
).to(device)



# dehazer = ImageDehazer()

# 自定义测试函数
def test_model():
    # 检查并创建结果文件夹
    results_folder_path = Path(results_folder)
    results_folder_path.mkdir(parents=True, exist_ok=True)

    # 设置扩散模型为评估模式
    diffusion.eval()

    # 加载模型权重
    checkpoint = torch.load(r"E:\Mymodel\diffusemodel\RDDM001\results_LHID_new\model-156000.pt", map_location=device)
    diffusion.load_state_dict(checkpoint['model'])  # 只加载模型参数

    # 初始化指标收集
    ssim_scores, psnr_scores = [], []

    with torch.no_grad():
        for i, (hazy_image, gt_image) in enumerate(test_loader):
            hazy_image, gt_image = hazy_image.to(device), gt_image.to(device)

            # 使用 U-Net 进行预处理
            preprocessed_image = unet_stage1(hazy_image).to(device)
            # dcp_image = dehazer.DCP(hazy_image)

            # 只将预处理图像和原始模糊图像作为模型的输入
            input_data = [preprocessed_image, hazy_image]

            # 使用扩散模型进行测试，得到生成的清晰图像
            generated_images = diffusion.sample(input_data, batch_size=1, last=True)
            if isinstance(generated_images, list):
                generated_image = generated_images[-1].to(device)  # 使用列表中最后一个生成的图像
            else:
                generated_image = generated_images.to(device)

            # 计算各项指标
            ssim_val = ssim(generated_image, gt_image).item()
            psnr_val = psnr(generated_image, gt_image)

            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)

            # 获取hazy图像的文件路径并提取文件名
            hazy_image_path = os.path.join(test_loader.dataset.hazy_dir, test_loader.dataset.hazy_images[i])
            hazy_image_name = Path(hazy_image_path).stem  # 提取不带扩展名的文件名

            # 保存生成的清晰图像，命名方式更新为原图名称+psnr+ssim
            save_path = results_folder_path / f"{hazy_image_name}_psnr{psnr_val:.4f}_ssim{ssim_val:.4f}.png"
            save_image(generated_image, save_path, nrow=1)
            print(f"Test sample saved at {save_path}")

    # 打印平均结果
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Average PSNR: {np.mean(psnr_scores):.4f}")
    print("测试完成。")


# 运行测试
if __name__ == "__main__":
    test_model()






