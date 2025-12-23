from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import os
from random import *
import nibabel as nib
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
from skimage.transform import resize

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
import pywt

from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)
from dataset.utils import load_file_list, center_crop_arr, random_crop_arr
from utils.common import instantiate_from_config


class CodeformerDataset(data.Dataset):
    
    def __init__(
        self,
        gt_file: str,
        lq_file:str,
        frq_file:str,
        file_backend_cfg: Mapping[str, Any],
        # out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        # self.gt_file_list = gt_file_list
        # self.gt_image_files = load_file_list(gt_file_list)
        # self.lq_file_list = lq_file_list
        # self.lq_image_files = load_file_list(lq_file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        # self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        # img1.png
        self.gt_file = gt_file
        self.frq_file = frq_file
        self.lq_file = lq_file
 
        gt_imgs = os.listdir(gt_file)
        self.total_imgs = sorted(gt_imgs)
        self._init_map()

    # def load_gt_image(self, image_path: str, max_retry: int=5) -> Optional[np.ndarray]:
    #     image_bytes = None
    #     while image_bytes is None:
    #         if max_retry == 0:
    #             return None
    #         image_bytes = self.file_backend.get(image_path)
    #         max_retry -= 1
    #         if image_bytes is None:
    #             time.sleep(0.5)
    #     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    #     if self.crop_type != "none":
    #         if image.height == self.out_size and image.width == self.out_size:
    #             image = np.array(image)
    #         else:
    #             if self.crop_type == "center":
    #                 image = center_crop_arr(image, self.out_size)
    #             elif self.crop_type == "random":
    #                 image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
    #     else:
    #         assert image.height == self.out_size and image.width == self.out_size
    #         image = np.array(image)
    #     # hwc, rgb, 0,255, uint8
    #     return image
    def load_gt_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        img = nib.load(image_path).get_fdata()
        img = np.squeeze(img)

        coeffs2 = pywt.dwt2(img, 'haar')
    
        # 提取高频分量（水平、垂直、对角线）
        cA, (cH, cV, cD) = coeffs2

        # 计算高频部分
        high_freq = np.sqrt(cH**2 + cV**2 + cD**2)
        high_freq_upsampled = resize(high_freq, img.shape, mode='reflect', anti_aliasing=False)

        # if len(img.shape) != 2:
        #     raise ValueError("图像尺寸不符合要求")
        # # print(img.shape)

        # img = Image.fromarray(img)
        # # print(img.size)
        # img = img.resize((192, 256), resample=Image.BOX)
        # img = np.array(img)
        # # 在这一步可视化

        # img = to_tensor(img)  # 转换为PyTorch张量

        # img = img.to(torch.float64)
        
        # min_val = img.min()
        # max_val = img.max()
        # img = (img - min_val) / (max_val - min_val)
        # img = img.numpy()
        # img = np.squeeze(img)


        # # 以下判断和调整图像尺寸的代码保持不变
        # if self.crop_type != "none":
        #     if image.height == self.out_size and image.width == self.out_size:
        #         image = np.array(image)
        #     else:
        #         if self.crop_type == "center":
        #             image = center_crop_arr(image, self.out_size)
        #         elif self.crop_type == "random":
        #             image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        # else:
        #     assert image.height == self.out_size and image.width == self.out_size
        #     image = np.array(image)
        return img, high_freq_upsampled
    

    def load_lq_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        img = nib.load(image_path).get_fdata()
        img = np.squeeze(img)
        # print(img.shape)

        # if len(img.shape) != 2:
        #     raise ValueError("图像尺寸不符合要求")
        # # print(img.shape)

        # img = Image.fromarray(img)
        # # print(img.size)
        # img = img.resize((128, 256), resample=Image.BOX)
        # img = np.array(img)
        # # # 在这一步可视化

        # img = to_tensor(img)  # 转换为PyTorch张量

        # img = img.to(torch.float64)
        
        # min_val = img.min()
        # max_val = img.max()
        # img = (img - min_val) / (max_val - min_val)
        # img = img.numpy()
        # img = np.squeeze(img)

        return img

# total_imgs:拍过顺序的原本图像  all_imgs：lq image
    def _init_map(self):
        self.degrade_map = {}
        for img_path in self.total_imgs:
            self.degrade_map[img_path] = []
        degrade_imgs = os.listdir(self.lq_file)
        total_imgs = sorted(degrade_imgs)
        for img_path in total_imgs:
            _, img_name = os.path.split(img_path)
            image_id_split = img_name.split('_')[:-1]
            image_id = ''
            for idx, word in enumerate(image_id_split):
                image_id += str(word)
                if idx != len(image_id_split) - 1:
                    image_id += '_'
            image_id += '.nii.gz'
            self.degrade_map[image_id].append(img_path)


    # def __getitem__(self, index: int):
    #     # load gt image
    #     # img_gt = None
    #     # while img_gt is None:
    #     #     # load meta file
    #     #     gt_image_file = self.gt_image_files[index]
    #     #     gt_path = gt_image_file["image_path"]
    #     #     prompt = gt_image_file["prompt"]
    #     #     img_gt = self.load_gt_image(gt_path)
    #     #     if img_gt is None:
    #     #         print(f"filed to load {gt_path}, try another image")
    #     #         index = random.randint(0, len(self) - 1)


    #     gt_name = self.total_imgs[index % len(self.total_imgs)]
    #     gt_path = os.path.join(self.gt_file, gt_name) # 路径
    #     img_gt = self.load_gt_image(gt_path)

    #     noise_image_list= self.degrade_map[gt_name]
    #     r= randint(0,len(noise_image_list)-1)# choose degraded image randomly
    #     noise_loc = os.path.join(self.lq_file, noise_image_list[r]) # 路径
    #     noise_image = self.load_gt_image(noise_loc)

    #     # # ------------------------ generate lq image ------------------------ #
    #     # # blur
    #     # kernel = random_mixed_kernels(
    #     #     self.kernel_list,
    #     #     self.kernel_prob,
    #     #     self.blur_kernel_size,
    #     #     self.blur_sigma,
    #     #     self.blur_sigma,
    #     #     [-math.pi, math.pi],
    #     #     noise_range=None
    #     # )
    #     # img_lq = cv2.filter2D(img_gt, -1, kernel)
    #     # # downsample
    #     # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
    #     # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
    #     # # noise
    #     # if self.noise_range is not None:
    #     #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
    #     # # jpeg compression
    #     # if self.jpeg_range is not None:
    #     #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
    #     # # resize to original size
    #     # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
    #     # BGR to RGB, [-1, 1]
    #     gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
    #     # BGR to RGB, [0, 1]
    #     lq = noise_image[..., ::-1].astype(np.float32)
        
    #     prompt = ""
    #     # return dict(jpg=gt, txt="", hint=lq)
    #     return gt, lq, prompt


    def __getitem__(self, index: int):
        gt_name = self.total_imgs[index % len(self.total_imgs)]
        gt_path = os.path.join(self.gt_file, gt_name)  # 路径
        # img_gt, img_frq = self.load_gt_image(gt_path)
        img_gt, _ = self.load_gt_image(gt_path)

        # frq_name = gt_name.replace('.nii.gz', '.nii')
        frq_name = gt_name
        frq_path = os.path.join(self.frq_file, frq_name)
        img_frq, _ = self.load_gt_image(frq_path)
        
        noise_image_list = self.degrade_map[gt_name]
        r = randint(0, len(noise_image_list) - 1)  # choose degraded image randomly
        noise_loc = os.path.join(self.lq_file, noise_image_list[r])  # 路径
        noise_image, _ = self.load_gt_image(noise_loc)

        # 对于灰度图像，需要调整图像的形状，以添加一个单通道的维度
        gt = img_gt.astype(np.float32) # 假设标准化到[0, 1]
        gt = gt[:, :, None]  # 从(H, W)转为(H, W, 1)形状

        gt_frq = img_frq.astype(np.float32) # 假设标准化到[0, 1]
        gt_frq = gt_frq[:, :, None]  # 从(H, W)转为(H, W, 1)形状

        lq = noise_image.astype(np.float32)
        lq = lq[:, :, None]  # 从(H, W)转为(H, W, 1)形状

        prompt = ""
        # print(gt.shape)
        return gt, lq, gt_frq

    def __len__(self) -> int:
        return len(self.total_imgs)
