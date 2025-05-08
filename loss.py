import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


class TotalVariationLoss(nn.Module):
    """
    Calculates the Total Variation (TV) loss for an input image.
    
    The TV loss measures the amount of variation or smoothness in an image by computing the differences
    between neighboring pixel values. It is commonly used in image processing tasks to encourage smoothness
    and reduce noise in the output image.
    
    Args:
        None
        
    Returns:
        loss (torch.Tensor): The computed TV loss for the input image.
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        """
        Computes the TV loss for the input image.
        
        Args:
            image (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).
        
        Returns:
            loss (torch.Tensor): The computed TV loss for the input image.
        """
        horizontal_diff = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
        vertical_diff = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
        # diagonal_diff = torch.abs(image[:, :, 1:, 1:] - image[:, :, :-1, :-1])
        # anti_diagonal_diff = torch.abs(image[:, :, 1:, :-1] - image[:, :, :-1, 1:])
        
        loss = torch.sum(horizontal_diff) + torch.sum(vertical_diff) # + torch.sum(diagonal_diff) + torch.sum(anti_diagonal_diff)
        loss /= 2
        return loss
