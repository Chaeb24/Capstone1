import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
from PIL import Image, ImageTk

import sys
import gc
import os
import numpy as np
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils

from main_test_swinir import define_model, setup, test

class ClearVision:
    def __init__(self, root):
        self.root = root
        self.root.title('Clear Vision - FFA-NET 안개 제거')
        self.root.geometry('1800x800')
        self.root['bg'] = 'white'
        self.clear_path = None
        self.model_path = None
        self.swinir_path = None

        # 헤더 프레임 (제목 + 멤버명 수평 정렬)
        header_frame = tk.Frame(self.root, bg='#4a90e2')
        header_frame.pack(fill='x',pady=(10, 0))

        self.title_label = tk.Label(header_frame, text='🔬 ClearVision',
                                    font=('Segoe UI', 18, 'bold'), bg='#4a90e2', fg='white')
        self.title_label.pack(side='left', padx=10, pady=10)

        self.members_label = tk.Label(header_frame,
                                      text='박채빈, 서지은, 이수진, 이효재, 윤마로, 허원석',
                                      font=('Segoe UI', 11), bg='#4a90e2', fg='#e3f2fd')
        self.members_label.pack(side='right', padx=10, pady=10)

        # 논문 링크 프레임
        link_frame = tk.Frame(self.root, bg='#e3f2fd')
        link_frame.pack(fill='x', pady=(0, 20))

        self.link_label = tk.Label(link_frame, text="📄 FFA-NET 논문 보러가기", 
                                   font=('Segoe UI', 11, 'underline'),
                                   bg='#e3f2fd', fg='#1976d2', cursor='hand2', pady=10)
        self.link_label.pack(pady=10)
        self.link_label.bind("<Button-1>", self.open_paper_link)

         # 질문 프레임
        question_frame = tk.LabelFrame(self.root, text=" 연구 질문 ", 
                                     font=('Segoe UI', 12, 'bold'), 
                                     bg='white', fg='#424242', relief='groove', bd=2)
        question_frame.pack(fill='x',padx=10)

         # Q1
        q1_frame = tk.Frame(question_frame, bg='white')
        q1_frame.pack(fill='x', padx=15, pady=10)
        
        q1_title = tk.Label(q1_frame, text="Q1.", font=('Segoe UI', 11, 'bold'), 
                           bg='white', fg='#1976d2')
        q1_title.pack(anchor='w')
        
        q1_text = """FFA-Net의 경우 비균일한 안개에서만 특히 결과가 안 좋은데,
단순히 트레이닝시 이를 감안하지 않은 데이터셋을 사용했기 때문이 아닌가?"""
        
        q1_label = tk.Label(q1_frame, text=q1_text, font=('Segoe UI', 10), 
                           bg='white', fg='#424242', justify='left', wraplength=800)
        q1_label.pack(anchor='w', padx=(20, 0), pady=(5, 0))
        
        # Q2
        q2_frame = tk.Frame(question_frame, bg='white')
        q2_frame.pack(fill='x', padx=15, pady=10)
        
        q2_title = tk.Label(q2_frame, text="Q2.", font=('Segoe UI', 11, 'bold'), 
                           bg='white', fg='#1976d2')
        q2_title.pack(anchor='w')
        
        q2_text = """여러가지 안개 제거 네트워크가 안개 제거 후 색상이 원본과 다르게 복구되는 경우가
많은데, 이를 별도의 네트워크(SwinIR) 등을 통해 후보정을 하여 개선할 수 있지 않을까?"""
        
        q2_label = tk.Label(q2_frame, text=q2_text, font=('Segoe UI', 10), 
                           bg='white', fg='#424242', justify='left', wraplength=800)
        q2_label.pack(anchor='w', padx=(20, 0), pady=(5, 0))


        # 버튼 프레임
        button_frame = tk.LabelFrame(self.root, bg='white',fg='#424242', relief='groove', bd=2)
        button_frame.pack(fill='x',padx=10,pady=10)

        self.upload_btn = tk.Button(button_frame, text='🔗 FFA-NET 모델 연결', font=('Segoe UI', 11, 'bold'), command=self.select_model)
        self.upload_btn.pack(side='left',padx=5)
        
        self.upload_btn2 = tk.Button(button_frame, text='🔗 SwinIR 모델 연결', font=('Segoe UI', 11, 'bold'), command=self.select_model_swinir)
        self.upload_btn2.pack(side='left',padx=5)

        self.image_btn = tk.Button(button_frame, text='🖼️ 이미지 선택', font=('Segoe UI', 11, 'bold'),command=self.open_file)
        self.image_btn.pack(side='left',padx=5)

        self.result_btn = tk.Button(button_frame, text='🔍 결과 보기', font=('Segoe UI', 11, 'bold'), command=self.run_model)
        self.result_btn.pack(side='left',padx=5)
        
        self.model_label = tk.Label(button_frame, text='선택된 모델 없음', font=('Segoe UI', 10), 
                           bg='white', fg='#424242', justify='left', wraplength=800)
        self.model_label.pack(side='left',padx=5)
        
        images_frame = tk.Frame(self.root, bg='white')
        images_frame.pack(fill='x', padx=10)
        
        frame_out = tk.Frame(images_frame, bg="#ffffff")
        self.hazy_frame = tk.Frame(frame_out, width=500, height=300, bg="#eeeeee")
        self.hazy_frame.pack()
        frame_label = tk.Label(frame_out, text='Hazy Image', font=('Segoe UI', 10), bg="#ffffff")
        frame_label.pack()
        frame_out.pack(side='left',padx=10, pady=10)
        
        frame_out = tk.Frame(images_frame, bg="#ffffff")
        self.swinir_frame = tk.Frame(frame_out, width=500, height=300, bg="#eeeeee")
        self.swinir_frame.pack()
        frame_label = tk.Label(frame_out, text='FFA-NET + SwinIR', font=('Segoe UI', 10), bg="#ffffff")
        frame_label.pack()
        frame_out.pack(side='right',padx=10, pady=10)
        
        frame_out = tk.Frame(images_frame, bg="#ffffff")
        self.clear_frame = tk.Frame(frame_out, width=500, height=300, bg="#eeeeee")
        self.clear_frame.pack()
        frame_label = tk.Label(frame_out, text='FFA-NET', font=('Segoe UI', 10), bg="#ffffff")
        frame_label.pack()
        frame_out.pack(side='right',padx=10, pady=10)
    
    def select_model(self):
        file_path = filedialog.askopenfilename(
            title="FFA-NET 모델 선택",
            filetypes=[("Trained model", "*.pk")],
            initialdir='trained_models'
        )
    
        if not file_path:
            return
        
        self.model_path = file_path
        if self.swinir_path:
            self.model_label.config(text = 'FFA-NET: ' + os.path.basename(self.model_path)
                                    + ' / SwinIR: ' + os.path.basename(self.swinir_path))
        else:
            self.model_label.config(text = 'SwinIR 모델을 선택하세요')
    
    def select_model_swinir(self):
        file_path = filedialog.askopenfilename(
            title="SwinIR 모델 선택",
            filetypes=[("Trained model", "*.pth")],
            initialdir='trained_models'
        )
    
        if not file_path:
            return
        
        self.swinir_path = file_path
        if self.model_path:
            self.model_label.config(text = 'FFA-NET: ' + os.path.basename(self.model_path)
                                    + ' / SwinIR: ' + os.path.basename(self.swinir_path))
        else:
            self.model_label.config(text = 'FFA-NET 모델을 선택하세요')
    
    def create_default_swinir_opts(self):
        if not self.swinir_path:
            return None
        
        class SwinIrArgs:
            def __init__(self, model_path):
                self.task = 'color_dn'
                self.scale = 1
                self.noise = 15
                self.jpeg = 40
                self.training_patch_size = 128
                self.large_model = False
                self.model_path = model_path
                self.folder_lq = '.'
                self.folder_gt = None
                self.tile = None
                self.tile_overlap = 32
        
        args = SwinIrArgs(self.swinir_path)
        return args
    
    def run_model(self):
        if not self.model_path or not self.clear_path:
            tk.messagebox.showerror(title='Error', message='이미지와 모델을 선택하세요.')
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckp=torch.load(self.model_path,map_location=device,weights_only=False)
        net=FFA(gps=3,blocks=19)
        net = nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
        net.eval()
        
        haze = Image.open(self.clear_path).convert('RGB')
        re_size = (self.hazy_frame.winfo_width() * 3, self.hazy_frame.winfo_height() * 3)
        print("이미지 크기 조정:", re_size)
        haze.thumbnail(re_size)
        haze1= tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
        ])(haze)[None,::]
        
        print("FFA-NET 시작")
        with torch.no_grad():
            pred = net(haze1)
        print("FFA-NET 완료")
        
        pred = pred.clamp(0,1)
        ts=torch.squeeze(pred.cpu())
        
        clear_img = tfs.ToPILImage()(ts)
        swinir_img = None
        
        del ts
        del ckp
        del net
        del haze
        del haze1
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.swinir_path:
            args = self.create_default_swinir_opts()
            model = define_model(args)
            model.eval()
            model = model.to(device)
            folder, _, border, window_size = setup(args)
            
            img_lq = pred
            print("SwinIR 시작")
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = test(img_lq, model, args, window_size)
                output = output[..., :h_old * args.scale, :w_old * args.scale]
            print("SwinIR 완료")
            
            output = output.data.squeeze().float().cpu().clamp_(0, 1)
            swinir_img = tfs.ToPILImage()(output)
            torch.cuda.empty_cache()
        
        try:
            #clear_img.save("ffa-net.png")
            image = clear_img.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
    
            # 이전 이미지 제거
            for widget in self.clear_frame.winfo_children():
                widget.destroy()
    
            # 새 이미지 표시
            img_label = tk.Label(self.clear_frame, image=photo)
            img_label.image = photo
            img_label.pack()
            
            if swinir_img:
                #swinir_img.save("swinir.png")
                image = swinir_img.resize((400, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
        
                # 이전 이미지 제거
                for widget in self.swinir_frame.winfo_children():
                    widget.destroy()
        
                # 새 이미지 표시
                img_label = tk.Label(self.swinir_frame, image=photo)
                img_label.image = photo
                img_label.pack()
        except Exception as e:
            messagebox.showerror("이미지 로드 실패", str(e))
        
       
    def open_paper_link(self, event):
        """FFA-NET 논문 링크 열기"""
        webbrowser.open("https://arxiv.org/abs/1911.07559")

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")],
            initialdir='test_imgs'
        )
    
        if not file_path:
            return
    
        try:
            image = Image.open(file_path)
            image = image.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
    
            # 이전 이미지 제거
            for widget in self.hazy_frame.winfo_children():
                widget.destroy()
    
            # 새 이미지 표시
            img_label = tk.Label(self.hazy_frame, image=photo)
            img_label.image = photo
            img_label.pack()
            
            self.clear_path = file_path
    
        except Exception as e:
            messagebox.showerror("이미지 로드 실패", str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = ClearVision(root)
    root.mainloop()