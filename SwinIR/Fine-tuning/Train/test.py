import os, argparse
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from metrics import psnr, ssim  # metrics.py에서 가져오기

abs_path = os.getcwd() + '/'

def tensorShow(tensors, titles=['haze']):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(221 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()

# =======================
# Argument
# =======================
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='its', help='its, ots, nh4k')
parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Test imgs folder')
opt = parser.parse_args()

dataset = opt.task
gps = 3
blocks = 19

# =======================
# Paths
# =======================
img_dir = os.path.join(abs_path, opt.test_imgs)
hazy_dir = os.path.join(img_dir, 'hazy')
clear_dir = os.path.join(img_dir, 'clear')
output_dir = os.path.join(abs_path, f'pred_FFA_{dataset}/')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

model_dir = os.path.join(abs_path, f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =======================
# Model load
# =======================
ckp = torch.load(model_dir, map_location=device, weights_only=False)
net = FFA(gps=gps, blocks=blocks)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

# =======================
# Inference + Metrics
# =======================
psnr_list, ssim_list = [], []

for im in os.listdir(hazy_dir):
    if not im.lower().endswith('.png'):
        continue  # PNG만 처리

    print(f'Processing {im}')

    haze_path = os.path.join(hazy_dir, im)

    # hazy 파일명에서 GT 파일명 추출 (예: "12_3.png" → "12.png")
    gt_name = im.split('_')[0] + '.png'
    gt_path = os.path.join(clear_dir, gt_name)

    haze = Image.open(haze_path).convert('RGB')
    haze_input = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64,0.6,0.58], std=[0.14,0.15,0.152])
    ])(haze)[None, ::]

    haze_no = tfs.ToTensor()(haze)[None, ::]

    with torch.no_grad():
        pred = net(haze_input).clamp(0,1).cpu()

    # 결과 저장 (이름 충돌 방지: '.' → '_')
    save_name = im.replace('.', '_') + '_FFA.png'
    vutils.save_image(pred.squeeze(), os.path.join(output_dir, save_name))

    # PSNR / SSIM 계산 (GT 존재할 때만)
    if os.path.exists(gt_path):
        gt = Image.open(gt_path).convert('RGB')
        gt_tensor = tfs.ToTensor()(gt)[None, ::]

        psnr_val = psnr(pred, gt_tensor)
        ssim_val = ssim(pred, gt_tensor).item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        print(f" -> GT: {gt_name}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

# =======================
# 평균 결과 출력
# =======================
if len(psnr_list) > 0:
    print("\n=== Evaluation Results ===")
    print(f"Average PSNR: {np.mean(psnr_list):.4f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
