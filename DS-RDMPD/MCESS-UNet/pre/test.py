import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml
from types import SimpleNamespace
from MCESS_UNet import U_Net
from torch.utils.data import DataLoader
from dataset import myImageFlodertest
from metrics import ssim, psnr, lpips, niqe
import lpips as lpips_pkg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return dict_to_namespace(config)


def compute_metrics(image1, image2, lpips_model):
    ssim_val = ssim(image1, image2)
    psnr_val = psnr(image1, image2)
    lpips_val = lpips(image1, image2, lpips_model)
    niqe_val = niqe(image1)
    return ssim_val, psnr_val, lpips_val, niqe_val


def save_reconstructed_image(reconstructed_image, idx, ssim_val, psnr_val, output_dir):
    """Save the reconstructed image with SSIM and PSNR in the filename."""
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with SSIM and PSNR values
    reconstructed_filename = os.path.join(output_dir,
                                          f"{idx}_SSIM_{ssim_val:.4f}_PSNR_{psnr_val:.2f}.png")

    # Save the reconstructed image
    save_image(reconstructed_image, reconstructed_filename)


def test(config_path, dataset_path, output_dir='test-image-Lmse'):
    config = load_config(config_path)
    model = U_Net(dim=64).to(device)
    model.load_state_dict(torch.load('/opt/data/private/diffusemodel/pre/train_result_RSID/generator_epoch_91.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512))

    ])

    dataset = myImageFlodertest(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    total_ssim, total_psnr, total_lpips, total_niqe, count = 0, 0, 0, 0, 0

    # 初始化LPIPS模型
    lpips_model = lpips_pkg.LPIPS(net='alex').to(device)

    for idx, batch in enumerate(dataloader):
        blurry_image, clear_image = batch
        blurry_image, clear_image = blurry_image.to(device), clear_image.to(device)

        with torch.no_grad():
            reconstructed_image = model(blurry_image)

        ssim_val, psnr_val, lpips_val, niqe_val = compute_metrics(reconstructed_image, clear_image, lpips_model)
        total_ssim += ssim_val
        total_psnr += psnr_val
        total_lpips += lpips_val
        total_niqe += niqe_val

        # Save only the reconstructed image with SSIM and PSNR values in filenames
        save_reconstructed_image(reconstructed_image, idx, ssim_val, psnr_val, output_dir)

        count += 1

        print(f'Image {idx}')

    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    avg_lpips = total_lpips / count
    avg_niqe = total_niqe / count

    print(
        f'Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average LPIPS: {avg_lpips:.4f}, Average NIQE: {avg_niqe:.4f}')


if __name__ == "__main__":
    config_path = "dehazing.yml"
    data_root = "../data/RealBlur/RealBlur-J/test"
    test(config_path, data_root)

