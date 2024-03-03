# MAE MSE PSNR SSIM CCR
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

def mae_metric(img1, img2, mask):
    mae = np.mean(np.abs(img1[mask==1]-img2[mask==1]))

    return mae

def mse_metric(img1, img2, mask):
    mse = np.average(img1[mask==1]-img2[mask==1]**2)

    return mse

def psnr_metric(img1, img2, mask):
    mse = np.average(img1[mask==1]-img2[mask==1]**2)
    max_signal = np.max(img1[mask==1])

    return 20 * math.log10(max_signal / math.sqrt(mse))

def ssim_metrix(img1, img2, mask, k1 = 1e-6, k2= 1e-6):
    if k1 and k2:
        value = ssim(img1[mask==1],img2[mask==1])
    else:
        value = ssim(img1[mask==1],img2[mask==1],k1,k2)
    
    return value

def cross_corr(img1, img2, mask):
    a = (img1[mask==1] - np.mean(img1[mask==1])) / (np.std(img1[mask==1]) * len(img1[mask==1]))
    b = (img2[mask==1] - np.mean(img2[mask==1])) / (np.std(img2[mask==1]))
    cc = np.correlate(a, b)
    
    return cc


