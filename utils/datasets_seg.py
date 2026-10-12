import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LoadImagesAndMasks(Dataset):
    def __init__(self, path, path_fake, imgsz, batch_size, augment=False, hyp=None, rect=False, cache_images=False,
                 single_cls=False, stride=32, pad=0.0, image_weights=False, prefix=''):
        # 加载真实图像路径
        self.img_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
        # 加载伪图像路径
        self.img_files_fake = sorted(
            [os.path.join(path_fake, f) for f in os.listdir(path_fake) if f.endswith(('.jpg', '.png'))])

        # 确认真实图像和伪图像数量相等
        assert len(self.img_files) == len(self.img_files_fake), "Images and fake images count do not match."

        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp  # 超参数
        self.rect = rect
        self.single_cls = single_cls
        self.stride = stride
        self.pad = pad
        self.cache_images = cache_images

        # 如果启用缓存，则缓存图像和伪图像
        if self.cache_images:
            self.img_cache = [self.load_image(img) for img in self.img_files]
            self.img_fake_cache = [self.load_image(img_fake) for img_fake in self.img_files_fake]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img_fake_path = self.img_files_fake[index]

        if self.cache_images:
            img = self.img_cache[index]
            img_fake = self.img_fake_cache[index]
        else:
            img = self.load_image(img_path)
            img_fake = self.load_image(img_fake_path)

        # 进行数据增强（如果启用）
        if self.augment:
            img, img_fake = self.apply_augmentation(img, img_fake)

        # 转换为张量并归一化
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [H, W, C] 转换为 [C, H, W]，并归一化到 [0, 1]
        img_fake = torch.from_numpy(img_fake).permute(2, 0, 1).float() / 255.0  # 同样对伪图像处理

        # 这里只返回空的标签，可以根据需要扩展标签逻辑
        labels_out = torch.zeros((0, 5))  # 假设没有标签

        # 返回图像、伪图像、标签和图像路径
        return img, img_fake, labels_out, img_path, img_fake_path

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.resize(img, (self.imgsz, self.imgsz))  # 调整大小
        return img

    def apply_augmentation(self, img, img_fake):
        # 在这里定义数据增强逻辑
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            img_fake = np.fliplr(img_fake).copy()
        return img, img_fake

    @staticmethod
    def collate_fn(batch):
        imgs, imgs_fake, labels, img_paths, img_fake_paths = zip(*batch)
        imgs = torch.stack(imgs, 0)
        imgs_fake = torch.stack(imgs_fake, 0)
        return imgs, imgs_fake, labels, img_paths, img_fake_paths
