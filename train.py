import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import torch
from PIL import Image
import torch.nn.functional as F
import os
import commentjson as json
from utils import conv_kervel, save_img

import configargparse
from loss import *
from deconv_methods.inr_recon import INRRecon

# train processing
def train():
    # Hash MLP config   Molde、GPU、lr
    #---------------------------------------
    with open('config.json') as config_file:
        config = json.load(config_file)


    # config
    #---------------------------------------
    parser = configargparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='data/leaf.png',help='original image - full sampled/ground truth.')
    parser.add_argument('--method', type=str, default='INR', help='reconstruction method.')
    parser.add_argument('--FWHM', type=int, default=50, help='set FWHM.  /um.')
    parser.add_argument('--loss', type=str, default='ssim', help='set different loss function ssim/l2.')
    parser.add_argument('--ssim_weight', type=float, default=0.3, help='loss - ssim_weight.')
    parser.add_argument('--l2_weight', type=float, default=0.7, help='loss - l2_weight.')
    parser.add_argument('--down_ratio', type=int, default=4, help='under sampled ratio.')
    parser.add_argument('--SNR', type=int, default=35, help='noise level.')
    parser.add_argument('--epoch', type=int, default=5000, help='training epoch.')
    parser.add_argument('--summary_epoch', type=int, default=100, help='summary epoch.')
 
    args = parser.parse_args()

    # creat output file
    #---------------------------------------
    os.makedirs(os.path.join('deconv_result',args.input_path,str(args.down_ratio),args.method), exist_ok=True)

    # CPU Or GPU
    #---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id=config["train"]["gpu"]
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    print(device)

    # load or-pam image (GT)
    #---------------------------------------
    img = Image.open(args.input_path)
    img= np.array(img)
    img=(img-img.min())/(img.max()-img.min())

    # load ar-pam image (for real ar-pam image, stimulation doesn't need)
    #---------------------------------------
    # arimg = Image.open('arpam.png')
    # arimg= np.array(arimg)
    # arimg=torch.tensor((arimg-arimg.min())/(arimg.max()-arimg.min())).to(device) 

    np.save(os.path.join('deconv_result',args.input_path,str(args.down_ratio),args.method,'gt.npy'),img)
    # save_img(img,f'original_img.png')
    # hot_img(img,'original_img.jpg')

    img=torch.tensor(img).to(device).half().unsqueeze(0).unsqueeze(0)

    # Create original PSF (Stimulation, real ar-pam doesn't need)
    #---------------------------------------
    conv_pam=conv_kervel(args.FWHM)
    conv_pam = torch.tensor(conv_pam, dtype=torch.float16).to(device).unsqueeze(0).unsqueeze(0)
  
    # Convolution, or-pam ---> ar-pam
    blurred_img = F.conv2d(img, conv_pam, padding='same').squeeze(0).squeeze(0)

    #******  Load AR-PAM mouse data, only for in-vivo exp  ********* 
    # To do
    # blurred_img = arimg[::args.down_ratio,::args.down_ratio]*0.4

    # down-sampling
    blurred_img = blurred_img[::args.down_ratio,::args.down_ratio]

    np.save(os.path.join('deconv_result',args.input_path,str(args.down_ratio),args.method,'arpam.npy'),blurred_img.cpu().detach())  
    blurred_img=blurred_img.half().to(device)

    
    save_img(blurred_img.cpu().detach(),f'stimulated_arpam.png')

    

    if args.method == 'INR':
        model = INRRecon(device, img, config)
    
    images, psnrs, ssims, lpips = model.recon(blurred_img ,args)

    img = img.squeeze().squeeze().cpu().detach().numpy()
    save_img(img, os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method,'gt.png'))
    save_img(blurred_img.cpu().detach().numpy(), os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method,'arpam.png'))
    
    # save quantitative result
    #---------------------------------------
    np.save(os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method, 'deconv_result.npy'), images)
    np.save(os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method,  'psnrs.npy'), np.asarray(psnrs))
    np.save(os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method, 'ssims.npy'), np.asarray(ssims))
    np.save(os.path.join('deconv_result', args.input_path, str(args.down_ratio), args.method, 'lpips.npy'), np.asarray(lpips))


if __name__ == '__main__':
    train()
