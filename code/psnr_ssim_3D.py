import os
import nibabel as nib

from torch.nn import functional as F

import torch


from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB image to YCbCr in PyTorch tensor format."""
    convert_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                   [-0.168736, -0.331264, 0.5],
                                   [0.5, -0.418688, -0.081312]]).to(img.device)

    img = torch.tensordot(img, convert_matrix, dims=1)
    img += torch.tensor([0, 128, 128]).view(1, 3, 1, 1).to(img.device)
    
    if y_only:
        return img[:, 0:1, :, :]
    return img

def traverse_folder_for_images(folder_path):
    """遍历文件夹查找所有NIfTI图像的路径"""
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                image_paths.append(os.path.join(root, file))
    return image_paths

def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    # img = img.to(torch.float64)
    img = torch.tensor(img)
    # img2 = img2.to(torch.float64)
    img2 = torch.tensor(img2)

    # mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    mse = torch.mean((img - img2)**2)
    return 10. * torch.log10(1. / (mse + 1e-8))
# 原始和修改图像文件夹路径
original_images_folder = '/data1/baihy/denoise/data/ANDI/Gt_fsl_norm/test'
# original_images_folder = '/data1/baihy/denoise/data/IXI-T1-HH-ISECRET/test/crop_good'
# original_images_folder = '/data1/baihy/denoise/data/IXI-T1-HH-Main-norm-slice/test'
# modified_images_folder = "/data1/baihy/denoise/data/drndcmb"
# modified_images_folder = "/data1/baihy/contrast/Andi_bm4d"
modified_images_folder = "/data1/baihy/denoise/InverseSR/results/ADNI/stage2-right/gard_cond_choose"
# modified_images_folder = "/data1/baihy/tools/DiffBIR/results/ADNI/stage1/test_combined"



original_images_paths = traverse_folder_for_images(original_images_folder)
modified_images_paths = traverse_folder_for_images(modified_images_folder)

print(len(original_images_paths))
print(len(modified_images_paths))

# print(original_images_paths)

# 初始化用于累积PSNR和SSIM值的变量
total_psnr = 0
total_ssim = 0
count = 0  # 计数器，用于记录有效比较的对数

for original_path in original_images_paths:
    original_img = nib.load(original_path).get_fdata()
    # original_img_all = nib.load(original_path).get_fdata()
    # original_img = original_img_all[:, :, 77:83]
    img_basename = os.path.basename(original_path)

    # print(img_basename[:-7])
    # modified_pattern = img_basename.replace('.nii.gz', '').replace('.nii', '') + "_"
    # print(modified_pattern)
    
    for modified_path in modified_images_paths:
        if img_basename[:-7] in modified_path:
            modified_img = nib.load(modified_path).get_fdata()
            # img = nib.load(modified_path)
            # all_data = img.get_fdata()
            # modified_img = all_data[:, :, 77:83]

            # 计算PSNR
            max_value = np.max(original_img)
            # psnr_value = peak_signal_noise_ratio(original_img, modified_img, data_range=max_value)
            psnr_value = calculate_psnr_pt(original_img, modified_img, crop_border=0)
        
            ssim_value =  structural_similarity(original_img, modified_img,
                                data_range=max_value, multichannel=True, channel_axis=2)
            
            print(modified_path[65:-4], psnr_value, ssim_value)
            total_psnr += psnr_value
            total_ssim += ssim_value
            count += 1
            
# 计算平均PSNR和SSIM
average_psnr = total_psnr / count
average_ssim = total_ssim / count

print(f"所有图像对的平均PSNR: {average_psnr}, 平均SSIM: {average_ssim}")
