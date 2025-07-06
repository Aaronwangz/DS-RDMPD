import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import natsort
from torchvision import transforms
from pathlib import Path


def default_loader(path1, path2, crop=True, resize=False, crop_size=256, resize_size=256):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    w, h = img1.size

    if crop:
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        img1 = img1.crop((x, y, x + crop_size, y + crop_size))
        img2 = img2.crop((x, y, x + crop_size, y + crop_size))

    if resize:
        img1 = img1.resize((resize_size, resize_size), Image.BILINEAR)
        img2 = img2.resize((resize_size, resize_size), Image.BILINEAR)

    return img1, img2

class myImageFloderval(Dataset):
    def __init__(self, root, transform=None, resize=False, resize_size=512):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'hazy')
        self.gt_dir = os.path.join(root, 'GT')

        # Get sorted lists of files
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))

        # Ensure directories have the same number of images
        if len(self.hazy_images) != len(self.gt_images):
            raise ValueError("Mismatch in number of images between 'hazy' and 'GT' directories.")

        self.resize = resize
        self.resize_size = resize_size

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_image, gt_image = default_loader(hazy_image_path, gt_image_path,
                                              crop=False, resize=self.resize,
                                              resize_size=self.resize_size)

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image = self.transform(gt_image)

        return hazy_image, gt_image

class myImageFlodertest(Dataset):
    def __init__(self, root, transform=None, resize=False, resize_size=512):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'hazy')
        self.gt_dir = os.path.join(root, 'GT')

        # Get sorted lists of files
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))

        # Ensure directories have the same number of images
        if len(self.hazy_images) != len(self.gt_images):
            raise ValueError("Mismatch in number of images between 'hazy' and 'GT' directories.")

        self.resize = resize
        self.resize_size = resize_size

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_image, gt_image = default_loader(hazy_image_path, gt_image_path,
                                              crop=False, resize=self.resize,
                                              resize_size=self.resize_size)

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image = self.transform(gt_image)

        return hazy_image, gt_image


class myImageFlodertrain(Dataset):
    def __init__(self, root, transform=None, crop=False, resize=False, crop_size=256, resize_size=256):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root,  'blur')
        self.gt_dir = os.path.join(root,  'gt')

        # Get sorted lists of files
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))

        # Ensure directories have the same number of images
        if len(self.hazy_images) != len(self.gt_images):
            raise ValueError("Mismatch in number of images between 'hazy' and 'GT' directories.")

        self.crop = crop
        self.resize = resize
        self.crop_size = crop_size
        self.resize_size = resize_size

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_image, gt_image = default_loader(hazy_image_path, gt_image_path,
                                              crop=self.crop, resize=self.resize,
                                              crop_size=self.crop_size, resize_size=self.resize_size)

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image = self.transform(gt_image)

        return hazy_image, gt_image


class myImageFlodertest2(Dataset):
    def __init__(self, root, transform=None, crop=False, resize=False, crop_size=256, resize_size=512):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'test', 'hazy')  # 假设测试集在 'test/hazy' 文件夹
        self.gt_dir = os.path.join(root, 'test', 'GT')      # 假设测试集对应的GT在 'test/GT' 文件夹

        # 获取排序后的文件列表
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))

        # 确保两个文件夹中的图像数量一致
        if len(self.hazy_images) != len(self.gt_images):
            raise ValueError("Mismatch in number of images between 'hazy' and 'GT' directories.")

        self.crop = crop
        self.resize = resize
        self.crop_size = crop_size
        self.resize_size = resize_size

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_image, gt_image = default_loader(
            hazy_image_path, gt_image_path,
            crop=self.crop, resize=self.resize,
            crop_size=self.crop_size, resize_size=self.resize_size
        )

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image = self.transform(gt_image)

        image_name = Path(self.hazy_images[idx]).stem

        return hazy_image, gt_image, image_name