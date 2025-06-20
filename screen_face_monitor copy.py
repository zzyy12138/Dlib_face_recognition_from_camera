# 透明覆盖屏幕的人脸识别监控程序 

import dlib 
import numpy as np 
import cv2 
import pandas as pd 
import os 
import time 
import logging 
import dxcam
import win32api
import win32gui
import win32con
import win32ui
from PIL import Image, ImageDraw, ImageFont 
import tkinter as tk 
import pystray 
from tkinter import Canvas 
from PIL import ImageTk 

# Dlib 初始化 
detector = dlib.get_frontal_face_detector()   
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')   
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")   

class TransparentFaceRecognizer:
    def __init__(self):
        # 人脸数据库 
        self.face_feature_known_list = []
        self.face_name_known_list = []
        
        # 当前帧数据 
        self.current_frame_face_feature_list = []
        self.current_frame_face_cnt = 0 
        self.current_frame_face_name_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_known_list = []  # 标记人脸是否在数据库中 
        
        # 性能监控 
        self.fps = 0 
        self.fps_show = 0 
        self.frame_start_time = 0 
        self.frame_cnt = 0 
        self.start_time = time.time()   
        
        # 屏幕设置 
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.camera.start(target_fps=20, video_mode=True)
        
        # 字体设置 
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)
        
        # 创建透明覆盖窗口
        self.create_overlay_window()
        
        # 加载人脸数据库 
        self.get_face_database()   
        
        # 创建系统托盘图标 
        self.create_system_tray_icon()
        
        # 启动处理线程
        self.running = True
        threading.Thread(target=self.process_loop, daemon=True).start()
        
        # 设置关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        """处理窗口关闭事件"""
        self.running = False
        self.root.destroy()
    
    def create_overlay_window(self):
        """创建透明覆盖窗口"""
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏主窗口
        
        # 创建透明覆盖窗口
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = {}
        wc.lpszClassName = "FaceRecognitionOverlay"
        wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)
        wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        class_atom = win32gui.RegisterClass(wc)

        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
            class_atom, None, win32con.WS_POPUP,
            0, 0, self.screen_width, self.screen_height,
            0, 0, 0, None
        )

        win32gui.SetLayeredWindowAttributes(self.hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

        # 创建绘图资源
        self.hdc = win32gui.GetDC(self.hwnd)
        self.mem_dc = win32gui.CreateCompatibleDC(self.hdc)
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(win32ui.CreateDCFromHandle(self.hdc), self.screen_width, self.screen_height)
        win32gui.SelectObject(self.mem_dc, self.bitmap.GetHandle())
        self.cdc = win32ui.CreateDCFromHandle(self.mem_dc)
        
        # 创建绘图工具
        self.box_pen_known = win32gui.CreatePen(win32con.PS_SOLID, 2, win32api.RGB(255, 0, 0))  # 已知人脸红色
        self.box_pen_unknown = win32gui.CreatePen(win32con.PS_SOLID, 2, win32api.RGB(0, 255, 255))  # 未知人脸青色
        self.text_font = win32ui.CreateFont({'name': 'Arial', 'height': -12, 'weight': win32con.FW_BOLD})
        self.name_font = win32ui.CreateFont({'name': 'SimSun', 'height': -20, 'weight': win32con.FW_BOLD})
        
        # 创建空画刷，避免填充矩形内部
        self.null_brush = win32gui.GetStockObject(win32con.NULL_BRUSH)
        self.background_brush = win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0))
    
    def create_system_tray_icon(self):
        """创建系统托盘图标"""
        image = Image.new('RGBA', (16, 16), (0, 120, 212, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((2, 2, 14, 14), fill=(255, 255, 255, 255))
        
        menu = pystray.Menu(
            pystray.MenuItem('退出', self.quit_program) 
        )
        
        self.tray_icon = pystray.Icon(
            "人脸识别监控",
            image,
            "人脸识别监控程序",
            menu 
        )
        
        # 在单独的线程中运行托盘图标 
        threading.Thread(target=self.tray_icon.run, daemon=True).start()
    
    def quit_program(self):
        """退出程序"""
        self.running = False
        self.root.quit()
        self.cleanup_resources()
        
    def cleanup_resources(self):
        """清理资源"""
        self.camera.stop()
        if hasattr(self, 'bitmap'):
            win32gui.DeleteObject(self.bitmap.GetHandle())
        if hasattr(self, 'mem_dc'):
            win32gui.DeleteDC(self.mem_dc)
        if hasattr(self, 'hwnd'):
            win32gui.ReleaseDC(self.hwnd, self.hdc)
            win32gui.DestroyWindow(self.hwnd)
        logging.info("程序已退出，资源已释放")
    
    def get_face_database(self):
        """加载人脸数据库"""
        if os.path.exists("data/features_all.csv"):   
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):   
                features = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])   
                for j in range(1, 129):
                    features.append(float(csv_rd.iloc[i][j]) if csv_rd.iloc[i][j] != '' else 0.0)
                self.face_feature_known_list.append(features)   
            logging.info("Loaded %d faces from database", len(self.face_feature_known_list))   
            return True 
        else:
            logging.error("Database file not found")
            return False 
    
    @staticmethod 
    def return_euclidean_distance(feature_1, feature_2):
        """计算欧式距离"""
        return np.sqrt(np.sum(np.square(np.array(feature_1) - np.array(feature_2))))   
    
    def update_fps(self):
        """更新帧率"""
        now = time.time()   
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps    
        self.start_time = now 
        self.frame_time = now - self.frame_start_time    
        self.fps = 1.0 / self.frame_time    
        self.frame_start_time = now 
    
    def process_loop(self):
        """处理循环"""
        while self.running:
            frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 检测人脸 
            faces = detector(frame, 0)
            
            # 更新当前帧数据 
            self.current_frame_face_feature_list = []
            self.current_frame_face_cnt = len(faces)
            self.current_frame_face_name_list = []
            self.current_frame_face_position_list = []
            self.current_frame_face_known_list = []  # 重置已知人脸标记 
            
            if faces:
                # 提取特征 
                for face in faces:
                    shape = predictor(frame, face)
                    self.current_frame_face_feature_list.append(
                        face_reco_model.compute_face_descriptor(frame, shape))
                
                # 识别每个人脸 
                for i, face in enumerate(faces):
                    # 默认未知 
                    self.current_frame_face_name_list.append("Unknown")   
                    self.current_frame_face_known_list.append(False)   # 默认不在数据库中 
                    
                    # 计算与数据库中所有人脸的欧式距离 
                    distances = []
                    for known_feature in self.face_feature_known_list:   
                        distances.append(self.return_euclidean_distance(
                            self.current_frame_face_feature_list[i], known_feature))
                    
                    # 找到最相似的人脸 
                    min_dist = min(distances)
                    if min_dist < 0.4:  # 阈值 
                        idx = distances.index(min_dist)   
                        self.current_frame_face_name_list[i] = self.face_name_known_list[idx]   
                        self.current_frame_face_known_list[i] = True  # 标记为已知人脸 
                    
                    # 记录人脸位置 
                    self.current_frame_face_position_list.append((
                        face.left(), face.top(), face.right(), face.bottom()))   
            
            # 绘制结果
            self.draw_results()
            
            # 更新FPS 
            self.update_fps()
            
            time.sleep(0.03)  # 控制帧率
    
    def draw_results(self):
        """绘制识别结果"""
        # 清空画布
        win32gui.FillRect(self.mem_dc, (0, 0, self.screen_width, self.screen_height), self.background_brush)
        
        # 绘制人脸框和名称
        for i, (left, top, right, bottom) in enumerate(self.current_frame_face_position_list):
            # 选择笔和字体
            if self.current_frame_face_known_list[i]:
                win32gui.SelectObject(self.mem_dc, self.box_pen_known)
                win32gui.SetTextColor(self.mem_dc, win32api.RGB(255, 255, 0))  # 黄色
            else:
                win32gui.SelectObject(self.mem_dc, self.box_pen_unknown)
                win32gui.SetTextColor(self.mem_dc, win32api.RGB(0, 255, 255))  # 青色
            
            # 选择空画刷，避免填充矩形内部
            old_brush = win32gui.SelectObject(self.mem_dc, self.null_brush)
            
            # 绘制人脸框
            win32gui.Rectangle(self.mem_dc, left, top, right, bottom)
            
            # 恢复原始画刷
            win32gui.SelectObject(self.mem_dc, old_brush)
            
            # 绘制人名
            win32gui.SelectObject(self.mem_dc, self.name_font.GetSafeHandle())
            win32gui.SetBkMode(self.mem_dc, win32con.TRANSPARENT)
            name = self.current_frame_face_name_list[i]
            self.cdc.TextOut(left, bottom + 5, name)
        
        # 显示FPS信息
        win32gui.SelectObject(self.mem_dc, self.text_font.GetSafeHandle())
        win32gui.SetTextColor(self.mem_dc, win32api.RGB(0, 255, 0))  # 绿色
        info_text = f"Faces: {self.current_frame_face_cnt} | FPS: {self.fps_show:.1f}"
        self.cdc.TextOut(20, 20, info_text)
        
        # 更新窗口
        win32gui.BitBlt(self.hdc, 0, 0, self.screen_width, self.screen_height, self.mem_dc, 0, 0, win32con.SRCCOPY)
        win32gui.PumpWaitingMessages()

def main():
    logging.basicConfig(level=logging.INFO)   
    recognizer = TransparentFaceRecognizer()
    recognizer.run()   

if __name__ == '__main__':
    import threading
    main()