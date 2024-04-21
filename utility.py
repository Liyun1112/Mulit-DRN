import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_ssim(image_path1, image_path2):
    # 加载图片
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    # 确保img2的尺寸与img1相同
    img2 = img2.resize(img1.size)

    # 将PIL图片转换为PyTorch张量
    img1 = to_tensor(img1)
    img2 = to_tensor(img2)

    # 计算PSNR和SSIM
    psnr_value = psnr(img1.numpy(), img2.numpy(), data_range=1)
    ssim_value = ssim(img1.numpy().transpose(1, 2, 0), img2.numpy().transpose(1, 2, 0), multichannel=True)

    return psnr_value, ssim_value
