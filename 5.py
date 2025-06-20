"""
基于Dlib的人脸识别监控程序（优化弹窗版本）
功能：
1. 实时捕获屏幕画面并进行人脸检测 
2. 对检测到的人脸进行识别和跟踪 
3. 支持透明窗口显示，不影响其他操作 
4. 系统托盘图标控制 
5. 智能管理新面孔弹窗，避免过多弹窗 
"""
 
import dlib 
import numpy as np 
import cv2 
import pandas as pd 
import os 
import time 
import logging 
from PIL import Image, ImageDraw, ImageFont, ImageTk 
import pyautogui 
import mss 
import ctypes 
import tkinter as tk 
import pystray 
from tkinter import Canvas, Toplevel, Label, simpledialog 
import random 
import string 
from datetime import datetime 
import threading 
import csv 
 
# 配置日志记录 
logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')
 
# 加载Dlib预训练模型 
cnn_face_detector = dlib.cnn_face_detection_model_v1('data/data_dlib/mmod_human_face_detector.dat') 
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat') 
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat") 
 
class TransparentFaceRecognizer:
    def __init__(self):
        # 人脸数据库相关 
        self.face_feature_known_list  = []
        self.face_name_known_list  = []
        self.face_image_path_list  = []
        
        # 当前帧数据 
        self.current_frame_face_feature_list  = []
        self.current_frame_face_cnt  = 0 
        self.current_frame_face_name_list  = []
        self.current_frame_face_position_list  = []
        self.current_frame_face_known_list  = []
        
        # 性能统计 
        self.fps_show  = 0 
        self.frame_cnt  = 0 
        self.start_time  = time.time() 
        
        # 屏幕捕获 
        self.sct  = mss.mss() 
        self.screen_width  = pyautogui.size().width  
        self.screen_height  = pyautogui.size().height  
        self.monitor  = {"top": 0, "left": 0, "width": self.screen_width,  "height": self.screen_height} 
        
        # 界面相关 
        self.font_chinese  = ImageFont.truetype("simsun.ttc",  30)
        self.root  = tk.Tk()
        self.root.attributes("-alpha",  0.7)
        self.root.attributes("-transparentcolor",  "white")
        self.root.attributes("-fullscreen",  True)
        self.root.attributes("-topmost",  True)
        self.root.overrideredirect(True) 
        self.canvas  = Canvas(self.root,  bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH,  expand=True)
        
        # 新面孔管理 
        self.last_new_face_time  = 0 
        self.new_face_cooldown  = 5  # 新面孔检测冷却时间(秒)
        self.current_new_face  = None  # 当前正在处理的新面孔 
        self.is_processing_new_face  = False  # 是否正在处理新面孔 
        self.shown_faces  = set()  # 已显示的面孔 
        self.show_popup  = True  # 是否显示弹窗 
        self.processed_features  = set()  # 已处理的特征集合 
        
        # 初始化 
        self.set_window_clickthrough() 
        self.get_face_database() 
        self.setup_exit_controls() 
        self.create_system_tray_icon() 
 
    def create_system_tray_icon(self):
        """创建系统托盘图标"""
        image = Image.new('RGBA',  (16, 16), (0, 120, 212, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((2,  2, 14, 14), fill=(255, 255, 255, 255))
        
        self.toggle_popup_item  = pystray.MenuItem(
            lambda item: f"{'关闭' if self.show_popup  else '开启'}弹窗显示",
            self.toggle_popup  
        )
        
        menu = pystray.Menu(
            self.toggle_popup_item, 
            pystray.MenuItem('退出', self.quit_program) 
        )
        
        self.tray_icon  = pystray.Icon("人脸识别监控", image, "人脸识别监控程序", menu)
        threading.Thread(target=self.tray_icon.run,  daemon=True).start()
 
    def toggle_popup(self, icon=None, item=None):
        """切换弹窗显示状态"""
        self.show_popup  = not self.show_popup  
        if icon:
            icon.update_menu() 
        logging.info(f" 弹窗显示已 {'开启' if self.show_popup  else '关闭'}")
 
    def setup_exit_controls(self):
        """设置退出控制"""
        self.root.bind('<Escape>',  lambda e: self.quit_program()) 
        menu = tk.Menu(self.root,  tearoff=0)
        menu.add_command(label=" 退出", command=self.quit_program) 
        self.root.bind("<Button-3>",  lambda e: menu.tk_popup(e.x_root,  e.y_root))
 
    def quit_program(self):
        """退出程序"""
        self.root.quit() 
        self.root.destroy() 
        self.sct.close() 
        cv2.destroyAllWindows() 
        logging.info(" 程序已退出")
 
    def set_window_clickthrough(self):
        """设置窗口可穿透点击"""
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) 
        style = ctypes.windll.user32.GetWindowLongA(hwnd,  -20)
        ctypes.windll.user32.SetWindowLongA(hwnd,  -20, style | 0x00000020 | 0x80000)
 
    def get_face_database(self):
        """从CSV文件加载人脸数据库"""
        path = "data/features_all.csv"  
        if os.path.exists(path)  and os.path.getsize(path)  > 0:
            try:
                # 添加dtype参数确保第一列作为字符串读取 
                df = pd.read_csv(path,  header=None, dtype={0: str})
                if not df.empty:  
                    for i in range(df.shape[0]):  
                        name = str(df.iloc[i][0])   # 确保转换为字符串 
                        features = [float(x) if x != '' else 0.0 for x in df.iloc[i][1:129]]  
                        feature_str = ','.join(map(str, features))
                        self.face_name_known_list.append(name)  
                        self.face_feature_known_list.append(features)  
                        self.face_image_path_list.append(f"data/data_faces_from_camera/person_{name}/img_face_1.jpg")  
                        self.processed_features.add(feature_str)  
                    logging.info(f" 已加载 {len(self.face_feature_known_list)}  张人脸")
                else:
                    logging.info(" 特征文件为空，没有加载任何人脸")
            except Exception as e:
                logging.error(f" 加载人脸数据库时出错: {str(e)}")
        else:
            logging.info(" 特征文件不存在或为空，跳过加载")
 
    @staticmethod 
    def return_euclidean_distance(f1, f2):
        """计算两个特征向量之间的欧氏距离"""
        return np.linalg.norm(np.array(f1)  - np.array(f2)) 
 
    def update_fps(self):
        """更新帧率统计"""
        now = time.time() 
        self.frame_cnt  += 1 
        if now - self.start_time  >= 1:
            self.fps_show  = self.frame_cnt  / (now - self.start_time) 
            self.start_time  = now 
        self.frame_cnt  = 0 
 
    def get_screen(self):
        """捕获屏幕图像"""
        img = np.array(self.sct.grab(self.monitor)) 
        img = cv2.cvtColor(img,  cv2.COLOR_BGRA2RGB)
        return img 
 
    def show_face_info(self, name, idx):
        """显示已知人脸信息"""
        if not self.show_popup  or name in self.shown_faces: 
            return 
            
        self.shown_faces.add(name) 
 
        def show():
            popup = Toplevel(self.root) 
            popup.title(f" 识别到: {name}")
            popup.attributes("-topmost",  True)
            
            img_path = self.face_image_path_list[idx]  if idx < len(self.face_image_path_list)  else ""
            
            if os.path.exists(img_path): 
                try:
                    img = Image.open(img_path).resize((200,  200))
                    img = ImageTk.PhotoImage(img)
                    label_img = Label(popup, image=img)
                    label_img.image  = img 
                    label_img.pack(pady=10) 
                except Exception as e:
                    Label(popup, text="图片加载失败").pack(pady=10)
            else:
                Label(popup, text="无图片").pack(pady=10)
                
            info = f"姓名: {name}\n识别时间: {time.strftime('%Y-%m-%d  %H:%M:%S')}"
            Label(popup, text=info, font=('Arial', 12)).pack(pady=10)
            
            def close():
                self.shown_faces.discard(name) 
                popup.destroy() 
                
            tk.Button(popup, text="关闭", command=close).pack(pady=10)
            popup.after(5000,  close)
 
        threading.Thread(target=show).start()
 
    def create_new_face_data(self, img, face_rect, shape, feature):
        """处理新检测到的人脸"""
        current_time = time.time() 
        
        # 检查冷却时间 
        if current_time - self.last_new_face_time  < self.new_face_cooldown: 
            logging.debug(" 新面孔检测过于频繁，已忽略")
            return 
            
        # 检查是否已经处理过这个特征 
        feature_str = ','.join(map(str, feature))
        if feature_str in self.processed_features: 
            logging.debug(" 已处理过此特征的人脸，跳过")
            return 
            
        # 如果已经有正在处理的新面孔，则跳过 
        if self.is_processing_new_face: 
            logging.debug(" 已有正在处理的新面孔，跳过")
            return 
            
        # 记录当前新面孔 
        self.current_new_face  = {
            'img': img,
            'face_rect': face_rect,
            'shape': shape,
            'feature': feature,
            'detect_time': current_time,
            'feature_str': feature_str 
        }
        
        # 标记为正在处理 
        self.is_processing_new_face  = True 
        self.last_new_face_time  = current_time 
        self.processed_features.add(feature_str) 
        
        # 直接处理这个新面孔 
        self.process_new_face() 
 
    def process_new_face(self):
        """处理当前新面孔"""
        if not self.current_new_face: 
            self.is_processing_new_face  = False 
            return 
            
        face_data = self.current_new_face  
 
        def ask_name_with_preview():
            popup = Toplevel(self.root) 
            popup.title(" 发现新的人脸")
            popup.geometry("300x360") 
            popup.attributes("-topmost",  True)
            popup.grab_set() 
 
            # 截取人脸图像 
            rect = face_data['face_rect']
            img = face_data['img']
            top = max(0, rect.top()) 
            bottom = min(img.shape[0],  rect.bottom()) 
            left = max(0, rect.left()) 
            right = min(img.shape[1],  rect.right()) 
            face_img = img[top:bottom, left:right]
 
            # 显示图像 
            try:
                pil_img = Image.fromarray(face_img).resize((200,  200))
                tk_img = ImageTk.PhotoImage(pil_img)
                img_label = Label(popup, image=tk_img)
                img_label.image  = tk_img 
                img_label.pack(pady=5) 
            except Exception as e:
                Label(popup, text="图像显示失败").pack()
 
            Label(popup, text="请输入此人姓名：").pack(pady=5)
            name_entry = tk.Entry(popup)
            name_entry.pack(pady=5) 
            name_entry.focus_set() 
 
            def on_confirm():
                name = name_entry.get().strip()  or "Unknown_" + ''.join(random.choices(string.ascii_letters  + string.digits,  k=4))
                folder = f"data/data_faces_from_camera/person_{name}"
                os.makedirs(folder,  exist_ok=True)
                
                # 查找可用的文件名 
                img_index = 1 
                while os.path.exists(os.path.join(folder,  f"img_face_{img_index}.jpg")):
                    img_index += 1 
                    
                img_filename = os.path.join(folder,  f"img_face_{img_index}.jpg")
                cv2.imwrite(img_filename,  cv2.cvtColor(face_img,  cv2.COLOR_RGB2BGR))
                
                # 更新内存中的数据库 
                self.face_name_known_list.append(name) 
                self.face_feature_known_list.append(face_data['feature']) 
                self.face_image_path_list.append(img_filename) 
                
                # 更新CSV文件 
                self.update_face_database_csv(name,  face_data['feature'])
                
                logging.info(f" 新增人脸: {name}，图像保存为 {img_filename}")
                popup.destroy() 
                
                # 处理完成 
                self.is_processing_new_face  = False 
                self.current_new_face  = None 
 
            def on_cancel():
                popup.destroy() 
                # 处理完成 
                self.is_processing_new_face  = False 
                self.current_new_face  = None 
 
            tk.Button(popup, text="确认", command=on_confirm).pack(pady=5)
            tk.Button(popup, text="跳过", command=on_cancel).pack(pady=5)
            popup.bind('<Return>',  lambda e: on_confirm())
            popup.protocol("WM_DELETE_WINDOW",  on_cancel)
 
        self.root.after(0,  ask_name_with_preview)
 
    def update_face_database_csv(self, name, feature):
        """更新人脸特征CSV文件"""
        try:
            with open("data/features_all.csv",  'a', newline='') as f:
                writer = csv.writer(f) 
                writer.writerow([name]  + list(feature))
        except Exception as e:
            logging.error(f" 更新特征文件时出错: {str(e)}")
 
    def process_frame(self):
        """处理每一帧图像"""
        self.canvas.delete("all") 
        img = self.get_screen() 
        scale = 0.5 
        img_small = cv2.resize(img,  (0, 0), fx=scale, fy=scale)
        
        # 人脸检测 
        faces = cnn_face_detector(img_small, 0)
        
        # 清空当前帧数据 
        self.current_frame_face_feature_list.clear() 
        self.current_frame_face_position_list.clear() 
        self.current_frame_face_name_list.clear() 
        self.current_frame_face_known_list.clear() 
        self.current_frame_face_cnt  = len(faces)
        
        for face in faces:
            rect = face.rect  
            rect = dlib.rectangle( 
                int(rect.left()  / scale), 
                int(rect.top()  / scale),
                int(rect.right()  / scale), 
                int(rect.bottom()  / scale)
            )
            
            shape = predictor(img, rect)
            feature = face_reco_model.compute_face_descriptor(img,  shape)
            name = "Unknown"
            known = False 
            
            if self.face_feature_known_list: 
                # 计算与已知人脸的相似度 
                distances = [self.return_euclidean_distance(feature, f) for f in self.face_feature_known_list] 
                min_dist = min(distances)
                
                if min_dist < 0.4:  # 识别阈值 
                    idx = distances.index(min_dist) 
                    name = self.face_name_known_list[idx] 
                    known = True 
                    
                    if name not in self.shown_faces  and self.show_popup: 
                        self.root.after(100,  lambda n=name, i=idx: self.show_face_info(n,  i))
                else:
                    # 未知人脸，尝试添加到处理 
                    self.create_new_face_data(img,  rect, shape, feature)
            else:
                # 没有已知人脸，尝试添加到处理 
                self.create_new_face_data(img,  rect, shape, feature)
                
            # 更新当前帧数据 
            self.current_frame_face_feature_list.append(feature) 
            self.current_frame_face_position_list.append((rect.left(),  rect.top(),  rect.right(),  rect.bottom())) 
            self.current_frame_face_name_list.append(name) 
            self.current_frame_face_known_list.append(known) 
            
        # 绘制结果 
        self.draw_results() 
        self.update_fps() 
        self.root.after(50,  self.process_frame) 
 
    def draw_results(self):
        """在画布上绘制检测结果"""
        # 绘制人脸框和名称 
        for i, (left, top, right, bottom) in enumerate(self.current_frame_face_position_list): 
            color = 'red' if self.current_frame_face_known_list[i]  else 'cyan'
            self.canvas.create_rectangle(left,  top, right, bottom, outline=color, width=2)
            self.canvas.create_text( 
                left, bottom + 20, 
                text=self.current_frame_face_name_list[i], 
                fill='yellow', 
                font=('Arial', 12, 'bold'), 
                anchor='nw'
            )
        
        # 显示统计信息 
        info = f"Faces: {self.current_frame_face_cnt}  | FPS: {self.fps_show:.1f}" 
        self.canvas.create_text( 
            20, 20, 
            text=info, 
            fill='lime', 
            font=('Arial', 12, 'bold'), 
            anchor='nw'
        )
 
    def run(self):
        """运行主循环"""
        self.process_frame() 
        self.root.mainloop() 
 
def main():
    try:
        recognizer = TransparentFaceRecognizer()
        recognizer.run() 
    except Exception as e:
        logging.error(f" 程序运行出错: {str(e)}")
        raise 
 
if __name__ == '__main__':
    main()