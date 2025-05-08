import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(gt,pre):
    z=1
    psnr_list=[]
    for i in range(z):
        psnr=peak_signal_noise_ratio(gt[:,:],pre[:,:])
        psnr_list.append(psnr)
    return sum(psnr_list)/len(psnr_list)     

def calculate_ssim(gt,pre):
    z=1
    ssim_list=[]
    for i in range(z):
        ssim=structural_similarity(gt[:,:],pre[:,:],data_range=1.0)
        ssim_list.append(ssim)
    return sum(ssim_list)/len(ssim_list) 

def conv_kervel(FWHM_um):
    pixel_size_um = 5  
    FWHM_pixels = FWHM_um / pixel_size_um  
    sigma_pixels = FWHM_pixels / (2 * np.sqrt(2 * np.log(2)))
    image_size = (31, 31)
    x, y = np.meshgrid(np.arange(-image_size[1] // 2, image_size[1] // 2),
                    np.arange(-image_size[0] // 2, image_size[0] // 2))

    PSF = np.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))
    PSF /= np.sum(PSF) 
    return PSF

def conv_kernel_torch(FWHM_um, device='cuda'):

    pixel_size_um = torch.tensor(5.0, dtype=torch.float16, device=device)  
    FWHM_pixels = FWHM_um / pixel_size_um  
    sigma_pixels = FWHM_pixels / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, dtype=torch.float16, device=device))) )
    image_size = (31, 31)
    x, y = torch.meshgrid(
    torch.arange(-image_size[1] // 2, image_size[1] // 2, dtype=torch.float16, device=device),
    torch.arange(-image_size[0] // 2, image_size[0] // 2, dtype=torch.float16, device=device),
    indexing='ij')
  
    PSF = torch.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))
    PSF = PSF / torch.sum(PSF)  
    PSF = PSF.requires_grad_()
    return PSF

def conv_kernel_torch_f(FWHM_um, device='cuda'):
    pixel_size_um = torch.tensor(5.0, dtype=torch.float, device=device)  # 每

    FWHM_pixels = FWHM_um / pixel_size_um 


    sigma_pixels = FWHM_pixels / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, dtype=torch.float, device=device))) )
    image_size = (30, 30)


    x, y = torch.meshgrid(
        torch.arange(-image_size[1] // 2, image_size[1] // 2, dtype=torch.float, device=device),
        torch.arange(-image_size[0] // 2, image_size[0] // 2, dtype=torch.float, device=device)
    )


    PSF = torch.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))
    PSF = PSF / torch.sum(PSF)  

    PSF = PSF.requires_grad_()

    return PSF

def add_noise(img, SNR, mode='peak'):

    img = img.astype(np.float32)  
    signal_power = np.max(img) ** 2 if mode == 'peak' else np.mean(img ** 2)  
    noise_power = signal_power / (10 ** (SNR / 10))  

    noise = np.random.normal(0, np.sqrt(noise_power), img.shape)  
    noisy_img = img + noise  #

    actual_snr = 10 * np.log10(signal_power / np.mean(noise ** 2))

    return noisy_img, actual_snr

def save_img(img, path, log=False): 
  plt.clf()
  if log:
    plt.imshow(img, origin='lower')
    plt.colorbar(label='dB')
    plt.clim(vmin=0, vmax=1) 
  else:
    plt.imshow(img,cmap='hot',vmin=0,vmax=1)

  plt.show()
  plt.savefig(path)