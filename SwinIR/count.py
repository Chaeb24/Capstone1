import cv2
import os
import re
import csv
import numpy as np

# 디렉토리 경로 (필요에 맞게 수정)
clear_dir = r"D:\Capstone-2025\SwinIR\result\Clear"
ffa_dir   = r"D:\Capstone-2025\SwinIR\result\FFA"
swinir_dir= r"D:\Capstone-2025\SwinIR\result\Swin"

output_csv = r"D:\Capstone-2025\SwinIR\result\result.csv"

def psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 * img1, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12   = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# Clear 폴더에서 숫자 추출
def extract_number(filename):
    match = re.search(r"\d+", filename)
    return match.group(0) if match else None

# 결과 CSV 작성
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageID", "PSNR_FFA", "SSIM_FFA", "PSNR_SwinIR", "SSIM_SwinIR"])
    
    for clear_name in os.listdir(clear_dir):
        num_id = extract_number(clear_name)  # ex: 123
        if not num_id:
            continue
        
        clear_path = os.path.join(clear_dir, clear_name)

        # FFA/SwinIR 폴더에서 해당 숫자가 들어간 첫 번째 파일 찾기
        ffa_file = next((f for f in os.listdir(ffa_dir) if num_id in f), None)
        swinir_file = next((f for f in os.listdir(swinir_dir) if num_id in f), None)
        
        if not ffa_file or not swinir_file:
            continue
        
        ffa_path = os.path.join(ffa_dir, ffa_file)
        swinir_path = os.path.join(swinir_dir, swinir_file)
        
        clear = cv2.imread(clear_path)
        ffa   = cv2.imread(ffa_path)
        swinir= cv2.imread(swinir_path)
        
        if clear is None or ffa is None or swinir is None:
            continue
        
        # 크기 다르면 resize (안전장치)
        if clear.shape != ffa.shape:
            ffa = cv2.resize(ffa, (clear.shape[1], clear.shape[0]))
        if clear.shape != swinir.shape:
            swinir = cv2.resize(swinir, (clear.shape[1], clear.shape[0]))
        
        psnr_ffa  = psnr(clear, ffa)
        ssim_ffa  = ssim(clear, ffa)
        psnr_swin = psnr(clear, swinir)
        ssim_swin = ssim(clear, swinir)
        
        writer.writerow([num_id, psnr_ffa, ssim_ffa, psnr_swin, ssim_swin])