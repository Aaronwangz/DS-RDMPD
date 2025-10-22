
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import natsort


def default_loader(path1, path2, crop=False, resize=False, crop_size=512, resize_size=512):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    w, h = img1.size

    if crop:
        crop_width, crop_height = crop_size  
        x = random.randint(0, w - crop_width)
        y = random.randint(0, h - crop_height)
        img1 = img1.crop((x, y, x + crop_width, y + crop_height))
        img2 = img2.crop((x, y, x + crop_width, y + crop_height))

    if resize:
        img1 = img1.resize(resize_size, Image.BILINEAR)
        img2 = img2.resize(resize_size, Image.BILINEAR)

    return img1, img2


class myImageFloderval(Dataset):
    def __init__(self, root, transform=None, resize=False, resize_size=512):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'hazy')
        self.gt_dir = os.path.join(root, 'GT')
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))
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
    def __init__(self, root, transform=None, crop=False, resize=False, crop_size=512, resize_size=512):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root,  'hazy')
        self.gt_dir = os.path.join(root,  'GT')
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))
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
