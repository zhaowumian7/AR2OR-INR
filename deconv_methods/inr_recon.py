# ----------------------------------------------#
# Pro    : AR2OR_INR
# Date   : 2024/2/22
# Author : Youshen Xiao
# Email  : xiaoysh2023@shanghaitech.edu.cn
# ----------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import os
from torch.optim import lr_scheduler
from utils import calculate_psnr, calculate_ssim, conv_kernel_torch, save_img
import lpips
from loss import *
from monai.losses import SSIMLoss


class INRRecon:
  def __init__(self, device, img, config):
    self.device = device
    self.config = config
    self.img = img
    self.image_size_x = img.shape[2]
    self.image_size_y = img.shape[3]
    

  def norm(self, x):
    return 2*((x - x.min())/(x.max() - x.min())) - 1


  def recon(self, blurred_img ,args):
        # load coordinates
        xs = torch.linspace(0, 1, self.image_size_x, device=self.device)
        ys = torch.linspace(0, 1, self.image_size_y, device=self.device)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        pos = torch.stack((yv.flatten(), xv.flatten())).t()
        model_input = pos
        FWHM_pre = torch.nn.Parameter(torch.tensor(0.70, dtype=torch.float16, device=self.device), requires_grad=True)
        model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                            encoding_config=self.config["encoding"],
                                            network_config=self.config["network"]).to(self.device)
        opt = torch.optim.Adam([{'params': model.parameters()},{'params': FWHM_pre, 'lr': self.config["train"]["lr"]}])
        scheduler = lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)

        # Losses
        #------------------------------
        l2_loss = nn.MSELoss()
        ssim_loss = SSIMLoss(spatial_dims=2, data_range=1.0, kernel_type='gaussian', win_size=7, k1=0.01, k2=0.1)
        tv_loss = TotalVariationLoss()
        loss_fn_alex = lpips.LPIPS(net='alex').half().to(self.device)
        target=blurred_img.unsqueeze(0).unsqueeze(0)

        # train
        #------------------------------
        img_ori=np.array(self.img.cpu().detach().squeeze(0).squeeze(0)).astype(np.float16)

        psnrs = []
        ssims = []
        perceps = []
        images = []
        psfs = []

        for i in tqdm(range(args.epoch)):
            
            opt.zero_grad()
           
            x_output0 = model(model_input).reshape(self.image_size_x, self.image_size_y)
            PSF_pre = conv_kernel_torch((FWHM_pre)*100, self.device).unsqueeze(0).unsqueeze(0).half()
            output_image = F.conv2d(x_output0.unsqueeze(0).unsqueeze(0), PSF_pre, padding='same')        
            
            # loss computation
            loss = dict()
            if args.loss == 'l2':
                    l2_intensity_loss = l2_loss(output_image[:,:,::args.down_ratio,::args.down_ratio], target)
                    loss["l2"] = (1., l2_intensity_loss)
            elif args.loss == 'ssim':
                    ssim_intensity_loss = ssim_loss(output_image[:,:,::args.down_ratio,::args.down_ratio], target)
                    loss["ssim"] = (args.ssim_weight, ssim_intensity_loss)
                    l2_intensity_loss = l2_loss(output_image[:,:,::args.down_ratio,::args.down_ratio], target)
                    loss["l2"] = (args.l2_weight, l2_intensity_loss)
                    total_variation_loss = tv_loss(x_output0.unsqueeze(0).unsqueeze(0))
                    loss["total_variation"] = (1e-6, total_variation_loss)
            
            total_loss = 0.
            for key, loss_value in loss.items():
                # print(key, loss_value)
                total_loss += loss_value[0] * loss_value[1]
            total_loss.backward()
            
            opt.step()
            scheduler.step()
            

            

            if not (i+1) % args.summary_epoch:
                images.append(x_output0.float().cpu().detach().numpy())
                psfs.append(PSF_pre.squeeze().squeeze().cpu().detach().numpy())

                psnr_est = calculate_psnr(img_ori,np.array(x_output0.cpu().detach()))
                ssim_est = calculate_ssim(img_ori,np.array(x_output0.cpu().detach()))
                gt = torch.from_numpy(self.norm(self.img.squeeze().cpu().numpy()))[None, None, ...].repeat(1, 3, 1, 1).to(self.device)
                est = torch.from_numpy(self.norm(x_output0.cpu().detach().squeeze().numpy()))[None, None, ...].repeat(1, 3, 1, 1).to(self.device)
                percep = loss_fn_alex(gt, est).item()

                psnrs.append(psnr_est)
                ssims.append(ssim_est)
                perceps.append(percep)

                print(f"PSNR: {psnr_est:.2f}, SSIM: {ssim_est:.2f}, Percep: {percep:.3f}")
                print("Epoch: %d, Total loss %0.9f" % (i, total_loss.item()))
             

                # # save PSF
                # fig, ax = plt.subplots(1, 2, figsize=(11, 5))  # 1 行 2 列

                # ax[0].imshow(self.conv_pam.squeeze(0).squeeze(0).cpu())
                # ax[0].axis('off')  
                # ax[0].set_title('original PSF')
                # ax[0].text(s='FWHM={:.4f}um'.format(args.FWHM), x=0, y=2, fontsize=14, color='white')

                # ax[1].imshow(PSF_pre.squeeze(0).squeeze(0).cpu().detach())
                # ax[1].axis('off')  
                # ax[1].set_title('predicted PSF')
                # ax[1].text(s='FWHM={:.4f}um'.format(FWHM_pre.item()*100), x=0, y=2, fontsize=14, color='white')

                # plt.savefig('PSF.png', bbox_inches='tight') 

                save_img(x_output0.cpu().detach(), os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method,'deconv_img_'+str(i)+'.png'))               
                np.save(os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method,'deconv_img_'+str(i)+'.npy'), x_output0.cpu().detach().numpy())

               
                # np.save(os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method,'Pre_PSF'+str(i)+'.npy'), PSF_pre.squeeze(0).squeeze(0).cpu().detach())
        np.save(os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method,'pre_psf.npy'), psfs)
        return images, psnrs, ssims, perceps            
           
