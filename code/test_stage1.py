import os
from argparse import ArgumentParser
import warnings
import lpips
import numpy as np
from torch.nn import functional as F
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import cv2
import nibabel as nib  # 导入nibabel库

from models.network_2D.unet_lesion import SwinIR
from utils.common import instantiate_from_config
# from utils.sampler import SpacedSampler

def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
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
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img

def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    data_range = img2.max()
    
    
    return 10. * torch.log10(data_range**2 / (mse + 1e-8))
    
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                        (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def main(args) -> None:
    device = 'cuda'
    cfg = OmegaConf.load(args.config)

    exp_dir = cfg.train.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Experiment directory created at {exp_dir}")

    # cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    # print('--------------------------------------------------------')
    # sd = torch.load(cfg.train.sd_path, map_location="cpu")
    # cldm.load_pretrained_sd(sd)
    # sd = torch.load(cfg.train.controlnet_ckpath, map_location="cpu")
    # cldm.load_controlnet_from_ckpt(sd)
    

    unet_l: UNetGenerator = instantiate_from_config(cfg.model.unet_l)
    # swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    # sd = {
    #     (k[len("module."):] if k.startswith("module.") else k): v
    #     for k, v in torch.load(cfg.train.swinir_path, map_location="cpu").items()
    # }
    sd = torch.load(cfg.train.unet_path, map_location="cpu")
    unet_l.load_state_dict(sd, strict=True)

    for p in unet_l.parameters():
        p.requires_grad = False
    
    # diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False, drop_last=False
    )

    # cldm.eval().to(device)
    unet_l.eval().to(device)
    # diffusion.to(device)
    
    # sampler = SpacedSampler(diffusion.betas)
    writer = SummaryWriter(exp_dir)
            
    val_loss = []
    clean_psnr = []
    val_lpips = []
    val_psnr = []
    clean_ssim = []
    val_ssim = []
    
    total_psnr = 0
    total_ssim = 0
    count = 0  # 计数器，用于记录有效比较的对数
    global_step = 1
    for val_lq, prompt, info in val_loader:
        # print('===================================')
        # print(val_lq.shape)
        
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
        # print(val_lq.shape)
        with torch.no_grad():
            
            # clean = swinir(val_lq)
            pred, pred_frq = unet_l(val_lq)
            # z_0 = cldm.vae_encode(val_lq)
            # z1 = cldm.vae_decode(z_0)
        #     z_3 = cldm.vae_encode(clean)
        #     cond = cldm.prepare_condition(clean, prompt)
        # t = torch.randint(0, diffusion.num_timesteps, (z_3.shape[0],), device=device)
        # pred_z1, loss = diffusion.p_losses(cldm, z_3, t, cond)
        # with torch.no_grad():
        #     z_1 = sampler.sample(
        #         model=cldm, device=device, steps=50, batch_size=len(val_lq), x_size=z_3.shape[1:],
        #         cond=cond, uncond=None, cfg_scale=1.0, x_T=None,
        #         progress_leave=False
        #     )
        # with torch.no_grad():
            # val_pred = cldm.vae_decode(z_0)
            # val_loss.append(loss.item())
            
            for i in range(val_lq.size(0)):
                # single_image = val_pred[i].clamp(0, 1).cpu().numpy()
                single_clean = pred[i].clamp(0, 1).cpu().numpy()
                single_clean_energy = pred_frq[i].clamp(0, 1).cpu().numpy()

                # 保存为 NIfTI 格式 
                # os.makedirs('results/new2/', exist_ok=True)
                # os.makedirs('results/swinir_norm/train/', exist_ok=True)

                # 假设 `info["name"][i]` 是当前处理的图像的原始文件名
                # 假设 `single_image` 是需要转化为NIfTI格式的图像数据
                # 假设 `single_clean` 是另一幅需要转化为NIfTI格式的清晰图像数据

                # 生成基于原始文件名和当前时间戳的新文件名，以确保不会覆盖现有文件
                new_filename2 = f"{info['name'][i].split('.')[0]}"
             
                # 转换图像数据为NIfTI格式，并保存到新文件名
                # image_nifti = nib.Nifti1Image(single_clean, affine=np.eye(4))
                # nib.save(image_nifti, os.path.join('results/swinir2/val/', new_filename2))
                # os.makedirs('results/ADNI/stage1_unet_v2/test', exist_ok=True)  # 确保目录存在
                # original_volume = nib.load('/data1/baihy/denoise/data/ANDI/Gt_Slice/test/0030_0_slice_0000.nii.gz')
                # image_clean_nifti = nib.Nifti1Image(single_clean, affine=original_volume.affine)
                # nib.save(image_clean_nifti, os.path.join('results/ADNI/stage1_unet_v2/test', new_filename2))
                
                # os.makedirs('results/ADNI_new/MRART/test', exist_ok=True)  # 确保目录存在
                # os.makedirs('results/ADNI_new/stage1/intensity_noise/1+5', exist_ok=True)  # 确保目录存在
                # nib.save(image_clean_nifti, os.path.join('results/ADNI_new/stage1/intensity_artifact/1+20', new_filename2))
                # os.makedirs('results/ADNI_new/stage1/only/40/', exist_ok=True)  # 确保目录存在
                # original_volume = nib.load('/data1/baihy/denoise/data/anat/sub/test_slice/sub-06_acq-t1wmprage070iso_T1w_slice_0000.nii')
                # original_volume = nib.load('/data1/baihy/denoise/data/ANDI/Gt_Slice/train/0015_0_slice_0000.nii.gz')
                # image_clean_nifti = nib.Nifti1Image(single_clean, affine=original_volume.affine)
                # nib.save(image_clean_nifti, os.path.join('results/ADNI_new/MRART/test', new_filename2))

                # os.makedirs('/data1/baihy/denoise/data/anat/sub/energy_sg1', exist_ok=True)  # 确保目录存在
                # os.makedirs('results/ADNI_new/MRART/test_energy', exist_ok=True)
                # original_volume = nib.load('/data1/baihy/denoise/data/anat/sub/test_slice/sub-06_acq-t1wmprage070iso_T1w_slice_0000.nii')
                # original_volume = nib.load('/data1/baihy/denoise/data/ANDI/Gt_Slice/train/0015_0_slice_0000.nii.gz')
                # image_clean_nifti = nib.Nifti1Image(single_clean_energy, affine=original_volume.affine)
                # nib.save(image_clean_nifti, os.path.join('results/ADNI_new/MRART/test_energy/', new_filename2))

                # os.makedirs('results/ADNI/stage1_unet_v2/test_energy', exist_ok=True)  # 确保目录存在
                # # original_volume = nib.load('/data1/baihy/denoise/data/anat/sub/test_slice/sub-06_acq-t1wmprage070iso_T1w_slice_0000.nii')
                # original_volume = nib.load('/data1/baihy/denoise/data/ANDI/Gt_Slice/test_energy_up/0030_0_slice_0000.nii')
                # image_clean_nifti = nib.Nifti1Image(single_clean_energy, affine=original_volume.affine)
                # nib.save(image_clean_nifti, os.path.join('results/ADNI/stage1_unet_v2/test_energy', new_filename2))

                os.makedirs('results/ADNI/stage1_ixi/train', exist_ok=True)  # 确保目录存在
                # original_volume = nib.load('/data1/baihy/denoise/data/anat/sub/test_slice/sub-06_acq-t1wmprage070iso_T1w_slice_0000.nii')
                original_volume = nib.load('/data1/baihy/denoise/data/ANDI/Gt_Slice/test_energy_up/0030_0_slice_0000.nii')
                image_clean_nifti = nib.Nifti1Image(single_clean, affine=original_volume.affine)
                nib.save(image_clean_nifti, os.path.join('results/ADNI/stage1_ixi/train', new_filename2))

                os.makedirs('results/ADNI/stage1_ixi/train_energy', exist_ok=True)  # 确保目录存在
                # original_volume = nib.load('/data1/baihy/denoise/data/anat/sub/test_slice/sub-06_acq-t1wmprage070iso_T1w_slice_0000.nii')
                original_volume = nib.load('/data1/baihy/denoise/data/ANDI/Gt_Slice/test_energy_up/0030_0_slice_0000.nii')
                image_clean_nifti = nib.Nifti1Image(single_clean_energy, affine=original_volume.affine)
                nib.save(image_clean_nifti, os.path.join('results/ADNI/stage1_ixi/train_energy', new_filename2))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
