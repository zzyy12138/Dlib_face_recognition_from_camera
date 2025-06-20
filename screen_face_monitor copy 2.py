import dlib 
import numpy as np 
import cv2 
import pandas as pd 
import os 
import time 
import logging 
from PIL import Image, ImageDraw, ImageFont 
import pyautogui 
import mss 
import ctypes 
import tkinter as tk 
import pystray 
from tkinter import Canvas, Toplevel, Label, PhotoImage 
from PIL import ImageTk 
 
# Dlib 初始化 
detector = dlib.get_frontal_face_detector()    
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')    
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")    
 
class TransparentFaceRecognizer:
    def __init__(self):
        # 人脸数据库 
        self.face_feature_known_list  = []
        self.face_name_known_list  = []
        self.face_image_path_list  = []  # 存储人脸图片路径 
        
        # 当前帧数据 
        self.current_frame_face_feature_list  = []
        self.current_frame_face_cnt  = 0 
        self.current_frame_face_name_list  = []
        self.current_frame_face_position_list  = []
        self.current_frame_face_known_list  = []  # 标记人脸是否在数据库中 
        
        # 性能监控 
        self.fps  = 0 
        self.fps_show  = 0 
        self.frame_start_time  = 0 
        self.frame_cnt  = 0 
        self.start_time  = time.time()    
        
        # 屏幕设置 
        self.sct  = mss.mss()    
        self.screen_width  = pyautogui.size().width     
        self.screen_height  = pyautogui.size().height     
        self.monitor  = {"top": 0, "left": 0, "width": self.screen_width,  "height": self.screen_height}    
        
        # 字体设置 
        self.font_chinese  = ImageFont.truetype("simsun.ttc",  30)
        
        # 创建透明窗口 
        self.root  = tk.Tk()
        self.root.attributes("-alpha",  0.7)  # 设置透明度 
        self.root.attributes("-transparentcolor",  "white")  # 设置白色为透明色 
        self.root.attributes("-fullscreen",  True)  # 全屏 
        self.root.attributes("-topmost",  True)  # 置顶 
        self.root.overrideredirect(True)    # 无边框 
        
        # 创建画布 
        self.canvas  = Canvas(self.root,  bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH,  expand=True)
        
        # 设置窗口点击穿透 
        self.set_window_clickthrough()    
        
        # 加载人脸数据库 
        self.get_face_database()    
        
        # 绑定退出热键和右键菜单 
        self.setup_exit_controls()  
 
        # 创建系统托盘图标 
        self.create_system_tray_icon()  
        
        # 记录已经弹出过的人脸，避免重复弹出 
        self.shown_faces  = set()
    
    def create_system_tray_icon(self):
        # 创建一个16x16的图标，使用蓝色作为背景 
        image = Image.new('RGBA',  (16, 16), (0, 120, 212, 255))
        draw = ImageDraw.Draw(image)
        # 绘制一个简单的圆形 
        draw.ellipse((2,  2, 14, 14), fill=(255, 255, 255, 255))
        
        menu = pystray.Menu(
            pystray.MenuItem('退出', self.quit_program)  
        )
        
        self.tray_icon  = pystray.Icon(
            "人脸识别监控",
            image,
            "人脸识别监控程序",
            menu 
        )
        
        # 在单独的线程中运行托盘图标 
        import threading 
        threading.Thread(target=self.tray_icon.run,  daemon=True).start()
        
    def setup_exit_controls(self):
        """设置退出控制"""
        # 绑定ESC键退出 
        self.root.bind('<Escape>',  lambda e: self.quit_program())  
        
        # 创建右键菜单 
        self.menu  = tk.Menu(self.root,  tearoff=0)
        self.menu.add_command(label="  退出", command=self.quit_program)  
        
        # 绑定右键菜单 
        self.root.bind("<Button-3>",  self.show_context_menu)  
        
    def show_context_menu(self, event):
        """显示右键菜单"""
        try:
            self.menu.tk_popup(event.x_root,  event.y_root)
        finally:
            self.menu.grab_release()  
        
    def quit_program(self):
        """退出程序"""
        # 停止主循环 
        self.root.quit()  
        # 销毁窗口 
        self.root.destroy()  
        # 确保资源释放 
        self.cleanup_resources()  
        
    def cleanup_resources(self):
        """清理资源"""
        self.sct.close()  
        cv2.destroyAllWindows()  
        logging.info("  程序已退出，资源已释放")
        
    def set_window_clickthrough(self):
        """设置窗口点击穿透"""
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())    
        style = ctypes.windll.user32.GetWindowLongA(hwnd,  -20)
        ctypes.windll.user32.SetWindowLongA(hwnd,  -20, style | 0x00000020 | 0x80000)
    
    def get_face_database(self):
        """加载人脸数据库"""
        if os.path.exists("data/features_all.csv"):     
            csv_rd = pd.read_csv("data/features_all.csv",  header=None)
            for i in range(csv_rd.shape[0]):     
                features = []
                name = csv_rd.iloc[i][0] 
                self.face_name_known_list.append(name)     
                
                # 构建图片路径: data/data_faces_from_camera/person_[姓名]/img_face_1.jpg 
                img_path = f"data/data_faces_from_camera/person_{name}/img_face_1.jpg" 
                self.face_image_path_list.append(img_path) 
                
                for j in range(1, 129):
                    features.append(float(csv_rd.iloc[i][j])  if csv_rd.iloc[i][j]  != '' else 0.0)
                self.face_feature_known_list.append(features)     
            logging.info("Loaded  %d faces from database", len(self.face_feature_known_list))     
            return True 
        else:
            logging.error("Database  file not found")
            return False 
    
    @staticmethod 
    def return_euclidean_distance(feature_1, feature_2):
        """计算欧式距离"""
        return np.sqrt(np.sum(np.square(np.array(feature_1)  - np.array(feature_2))))    
    
    def update_fps(self):
        """更新帧率"""
        now = time.time()    
        if str(self.start_time).split(".")[0]  != str(now).split(".")[0]:
            self.fps_show  = self.fps     
        self.start_time  = now 
        self.frame_time  = now - self.frame_start_time     
        self.fps  = 1.0 / self.frame_time     
        self.frame_start_time  = now 
    
    def get_screen(self):
        """获取屏幕截图"""
        sct_img = self.sct.grab(self.monitor)    
        img = np.array(sct_img)    
        img = cv2.cvtColor(img,  cv2.COLOR_BGRA2BGR)
        return img 
    
    def show_face_info(self, name, idx):
        """显示人脸信息窗口"""
        if name in self.shown_faces: 
            return  # 已经显示过，不再重复显示 
            
        self.shown_faces.add(name) 
        
        # 创建弹出窗口 
        popup = Toplevel(self.root) 
        popup.title(f" 识别到: {name}")
        popup.attributes("-topmost",  True)  # 置顶 
        
        # 尝试加载图片 
        img_path = self.face_image_path_list[idx]  if idx < len(self.face_image_path_list)  else ""
        
        if img_path and os.path.exists(img_path): 
            try:
                # 使用PIL打开图片并调整大小 
                pil_img = Image.open(img_path) 
                pil_img = pil_img.resize((200,  200), Image.LANCZOS)
                img = ImageTk.PhotoImage(pil_img)
                
                # 显示图片 
                img_label = Label(popup, image=img)
                img_label.image  = img  # 保持引用 
                img_label.pack(pady=10) 
            except Exception as e:
                logging.error(f" 加载图片失败: {e}")
                Label(popup, text="无法加载图片").pack(pady=10)
        else:
            Label(popup, text="无可用图片").pack(pady=10)
        
        # 显示基本信息 
        info_text = f"姓名: {name}\n识别时间: {time.strftime('%Y-%m-%d  %H:%M:%S')}"
        Label(popup, text=info_text, font=('Arial', 12)).pack(pady=10)
        
        # 添加关闭按钮 
        close_btn = tk.Button(popup, text="关闭", command=popup.destroy) 
        close_btn.pack(pady=10) 
        
        # 5秒后自动关闭 
        popup.after(50000,  popup.destroy) 
    
    def process_frame(self):
        """处理每一帧"""
        self.frame_cnt  += 1 
        self.canvas.delete("all")    # 清除上一帧 
        
        # 获取屏幕截图 
        img = self.get_screen()    
        
        # 检测人脸 
        faces = detector(img, 0)
        
        # 更新当前帧数据 
        self.current_frame_face_feature_list  = []
        self.current_frame_face_cnt  = len(faces)
        self.current_frame_face_name_list  = []
        self.current_frame_face_position_list  = []
        self.current_frame_face_known_list  = []  # 重置已知人脸标记 
        
        if faces:
            # 提取特征 
            for face in faces:
                shape = predictor(img, face)
                self.current_frame_face_feature_list.append(    
                    face_reco_model.compute_face_descriptor(img,  shape))
            
            # 识别每个人脸 
            for i, face in enumerate(faces):
                # 默认未知 
                self.current_frame_face_name_list.append("Unknown")    
                self.current_frame_face_known_list.append(False)    # 默认不在数据库中 
                
                # 计算与数据库中所有人脸的欧式距离 
                distances = []
                for known_feature in self.face_feature_known_list:    
                    distances.append(self.return_euclidean_distance(    
                        self.current_frame_face_feature_list[i],  known_feature))
                
                # 找到最相似的人脸 
                min_dist = min(distances)
                if min_dist < 0.4:  # 阈值 
                    idx = distances.index(min_dist)    
                    name = self.face_name_known_list[idx] 
                    self.current_frame_face_name_list[i]  = name   
                    self.current_frame_face_known_list[i]  = True  # 标记为已知人脸 
                    
                    # 如果是已知人脸且之前没有显示过，弹出信息窗口 
                    if name != "Unknown" and name not in self.shown_faces: 
                        self.root.after(100,  lambda n=name, i=idx: self.show_face_info(n,  i))
                
                # 记录人脸位置 
                self.current_frame_face_position_list.append((    
                    face.left(),  face.top(),  face.right(),  face.bottom()))    
        
        # 在画布上绘制识别结果 
        self.draw_results()    
        
        # 更新FPS 
        self.update_fps()    
        
        # 下一帧 
        self.root.after(10,  self.process_frame)    
    
    def draw_results(self):
        """在画布上绘制识别结果"""
        # 绘制人脸框 
        for i, (left, top, right, bottom) in enumerate(self.current_frame_face_position_list):    
            # 根据是否在数据库中选择框的颜色 
            box_color = 'red' if self.current_frame_face_known_list[i]  else 'cyan'
            
            # 人脸框 
            self.canvas.create_rectangle(    
                left, top, right, bottom, 
                outline=box_color, width=2)
            
            # 人名 
            self.canvas.create_text(    
                left, bottom + 20, 
                text=self.current_frame_face_name_list[i],    
                fill='yellow', font=('Arial', 12, 'bold'), anchor='nw')
        
        # 显示FPS信息 
        info_text = f"Faces: {self.current_frame_face_cnt}  | FPS: {self.fps_show:.1f}"    
        self.canvas.create_text(    
            20, 20, text=info_text,
            fill='lime', font=('Arial', 12, 'bold'), anchor='nw')
    
    def run(self):
        """运行程序"""
        try:
            self.process_frame()    
            self.root.mainloop()    
        finally:
            # 确保资源释放 
            self.cleanup_resources()  
 
def main():
    logging.basicConfig(level=logging.INFO)    
    recognizer = TransparentFaceRecognizer()
    recognizer.run()    
 
if __name__ == '__main__':
    main()