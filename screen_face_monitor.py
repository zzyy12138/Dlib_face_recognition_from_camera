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
 
# 创建logs目录
if not os.path.exists('logs'):
    os.makedirs('logs')

class DailyLogManager:
    """每日日志管理器"""
    def __init__(self):
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.log_filename = f"logs/face_monitor_{self.current_date}.log"
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志配置"""
        # 清除现有的处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 配置新的日志处理器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def check_and_rotate(self):
        """检查是否需要轮转日志文件"""
        current_date = datetime.now().strftime('%Y%m%d')
        if current_date != self.current_date:
            logging.info(f"日期变更，从 {self.current_date} 到 {current_date}，开始新的日志文件")
            self.current_date = current_date
            self.log_filename = f"logs/face_monitor_{self.current_date}.log"
            self.setup_logging()
            logging.info(f"日志文件已切换到: {self.log_filename}")

# 创建全局日志管理器
log_manager = DailyLogManager()
 
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
        
        # 识别阈值设置
        self.recognition_threshold = 0.4  # 默认识别阈值
        
        # 根据GPU可用性自动调整设置
        if gpu_available:
            # GPU模式 - 高性能设置
            self.cpu_optimization = False
            self.process_interval = 50  # 更快的处理间隔
            self.image_scale = 0.5  # 更高的图像质量
            logging.info("使用GPU模式，启用高性能设置")
        else:
            # CPU模式 - 优化设置
            self.cpu_optimization = True
            self.process_interval = 100  # 较慢的处理间隔
            self.image_scale = 0.3  # 更小的图像以节省CPU
            logging.info("使用CPU模式，启用优化设置")
        
        # 状态显示控制
        self.show_status_display = True  # 是否显示左上角状态信息
        
        # 新面孔管理 - 增强弹窗控制
        self.last_new_face_time  = 0 
        self.new_face_cooldown  = 5  # 新面孔检测冷却时间(秒)
        self.current_new_face  = None  # 当前正在处理的新面孔 
        self.is_processing_new_face  = False  # 是否正在处理新面孔 
        self.new_face_popup_window  = None  # 当前新面孔弹窗窗口对象
        self.shown_faces  = set()  # 已显示的面孔 
        self.show_popup  = False  # 是否显示弹窗 - 默认关闭
        self.auto_add_new_faces  = False  # 是否自动识别添加新面孔 - 默认关闭
        self.processed_features  = set()  # 已处理的特征集合
        
        # 初始化 
        self.set_window_clickthrough() 
        self.get_face_database() 
        self.setup_exit_controls() 
        self.create_system_tray_icon() 
        
        logging.info("人脸识别监控系统初始化完成")
        logging.info(f"屏幕分辨率: {self.screen_width}x{self.screen_height}")
        logging.info(f"GPU状态: {'可用' if gpu_available else '不可用'}")
        if gpu_available:
            logging.info(f"GPU设备数量: {gpu_count}")
        logging.info(f"运行模式: {'GPU加速' if gpu_available else 'CPU优化'}")
        logging.info(f"识别阈值: {self.recognition_threshold}")
        logging.info(f"处理间隔: {self.process_interval}ms")
        logging.info(f"图像缩放: {self.image_scale}")
        logging.info(f"弹窗显示: {'开启' if self.show_popup else '关闭'}")
        logging.info(f"自动发现新面孔: {'开启' if self.auto_add_new_faces else '关闭'}")
        logging.info(f"状态显示: {'开启' if self.show_status_display else '关闭'}")
 
    def create_system_tray_icon(self):
        """创建系统托盘图标"""
        # 创建一个更美观的图标
        image = Image.new('RGBA', (32, 32), (0, 0, 0, 0))  # 透明背景
        draw = ImageDraw.Draw(image)
        
        # 绘制圆形背景
        draw.ellipse((2, 2, 30, 30), fill=(52, 152, 219, 255), outline=(41, 128, 185, 255), width=2)
        
        # 绘制人脸轮廓
        # 头部轮廓
        draw.ellipse((8, 6, 24, 22), fill=(255, 255, 255, 255), outline=(200, 200, 200, 255), width=1)
        
        # 眼睛
        draw.ellipse((11, 10, 15, 14), fill=(70, 130, 180, 255))  # 左眼
        draw.ellipse((17, 10, 21, 14), fill=(70, 130, 180, 255))  # 右眼
        
        # 鼻子
        draw.ellipse((15, 14, 17, 18), fill=(255, 182, 193, 255))
        
        # 嘴巴
        draw.arc((13, 18, 19, 22), start=0, end=180, fill=(220, 20, 60, 255), width=2)
        
        # 添加一些装饰性元素
        draw.ellipse((4, 4, 8, 8), fill=(255, 255, 0, 180))  # 左上角小圆点
        draw.ellipse((24, 4, 28, 8), fill=(255, 255, 0, 180))  # 右上角小圆点
        
        # 缩放到16x16用于系统托盘
        image = image.resize((16, 16), Image.Resampling.LANCZOS)
        
        self.toggle_popup_item  = pystray.MenuItem(
            lambda item: f"{'关闭' if self.show_popup  else '开启'}弹窗显示",
            self.toggle_popup  
        )
        
        self.toggle_auto_add_item  = pystray.MenuItem(
            lambda item: f"{'关闭' if self.auto_add_new_faces  else '开启'}自动发现新面孔",
            self.toggle_auto_add_new_faces  
        )
        
        self.threshold_item = pystray.MenuItem(
            lambda item: f"识别阈值: {self.recognition_threshold:.2f}",
            self.adjust_threshold
        )
        
        self.toggle_status_item = pystray.MenuItem(
            lambda item: f"{'关闭' if self.show_status_display else '开启'}状态显示",
            self.toggle_status_display
        )
        
        menu = pystray.Menu(
            self.toggle_popup_item, 
            self.toggle_auto_add_item,
            self.threshold_item,
            self.toggle_status_item,
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
 
    def toggle_auto_add_new_faces(self, icon=None, item=None):
        """切换自动发现新面孔状态"""
        self.auto_add_new_faces  = not self.auto_add_new_faces  
        if icon:
            icon.update_menu() 
        logging.info(f" 自动发现新面孔已 {'开启' if self.auto_add_new_faces  else '关闭'}")
 
    def adjust_threshold(self, icon=None, item=None):
        """调整识别阈值"""
        def show_threshold_dialog():
            dialog = Toplevel(self.root)
            dialog.title("调整识别阈值")
            dialog.geometry("500x500")
            dialog.attributes("-topmost", True)
            dialog.grab_set()
            
            # 说明文字
            info_text = """识别阈值说明：
• 阈值越小，识别越严格，误识别率越低
• 阈值越大，识别越宽松，漏识别率越低
• 推荐范围：0.3 - 0.6
• 当前阈值：{:.2f}""".format(self.recognition_threshold)
            
            Label(dialog, text=info_text, font=('Arial', 10), justify='left').pack(pady=10)
            
            # 预设按钮 - 使用两行布局
            preset_frame = tk.Frame(dialog)
            preset_frame.pack(pady=10)
            
            # 第一行按钮
            row1_frame = tk.Frame(preset_frame)
            row1_frame.pack(pady=5)
            
            # 第二行按钮
            row2_frame = tk.Frame(preset_frame)
            row2_frame.pack(pady=5)
            
            def set_preset_threshold(value):
                self.recognition_threshold = value
                logging.info(f" 识别阈值已设置为: {value:.2f}")
                # 更新菜单显示
                if hasattr(self, 'tray_icon') and self.tray_icon:
                    self.tray_icon.update_menu()
                # 更新对话框中的当前阈值显示
                current_threshold_label.config(text=f"当前阈值：{value:.2f}")
            
            # 第一行：严格和标准
            tk.Button(row1_frame, text="严格 (0.3)", width=12,
                     command=lambda: set_preset_threshold(0.3)).pack(side=tk.LEFT, padx=5)
            tk.Button(row1_frame, text="标准 (0.4)", width=12,
                     command=lambda: set_preset_threshold(0.4)).pack(side=tk.LEFT, padx=5)
            
            # 第二行：宽松和很宽松
            tk.Button(row2_frame, text="宽松 (0.5)", width=12,
                     command=lambda: set_preset_threshold(0.5)).pack(side=tk.LEFT, padx=5)
            tk.Button(row2_frame, text="很宽松 (0.6)", width=12,
                     command=lambda: set_preset_threshold(0.6)).pack(side=tk.LEFT, padx=5)
            
            # 当前阈值显示
            current_threshold_label = Label(dialog, text=f"当前阈值：{self.recognition_threshold:.2f}", 
                                          font=('Arial', 12, 'bold'), fg='blue')
            current_threshold_label.pack(pady=10)
            
            # 关闭按钮
            tk.Button(dialog, text="关闭", command=dialog.destroy, width=10).pack(pady=10)
        
        self.root.after(0, show_threshold_dialog)
 
    def toggle_status_display(self, icon=None, item=None):
        """切换状态显示开关"""
        self.show_status_display = not self.show_status_display
        if icon:
            icon.update_menu()
        logging.info(f" 状态显示已 {'开启' if self.show_status_display else '关闭'}")
 
    def setup_exit_controls(self):
        """设置退出控制"""
        self.root.bind('<Escape>',  lambda e: self.quit_program()) 
        menu = tk.Menu(self.root,  tearoff=0)
        menu.add_command(label=" 退出", command=self.quit_program) 
        self.root.bind("<Button-3>",  lambda e: menu.tk_popup(e.x_root,  e.y_root))
 
    def quit_program(self):
        """退出程序"""
        # 清理系统托盘图标
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.stop()
        
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
            # 每30秒记录一次详细性能统计
            if int(now) % 30 == 0:
                mode = "GPU加速" if gpu_available else "CPU优化"
                logging.info(f"性能统计 - 模式:{mode}, FPS:{self.fps_show:.1f}, 检测到人脸:{self.current_frame_face_cnt}, 处理间隔:{self.process_interval}ms, 图像缩放:{self.image_scale}")
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
        # 检查自动发现新面孔功能是否开启
        if not self.auto_add_new_faces:
            logging.debug(" 自动发现新面孔功能已关闭，跳过")
            return
            
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
            
        # 检查是否正在处理新面孔（包括弹窗是否还在显示）
        if self.is_processing_new_face or self.new_face_popup_window is not None: 
            logging.debug(" 已有正在处理的新面孔或弹窗还在显示，跳过")
            return 
            
        # 记录新面孔检测
        logging.info(f"发现新面孔，准备处理 (位置: {face_rect.left()},{face_rect.top()}-{face_rect.right()},{face_rect.bottom()})")
            
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
            self.new_face_popup_window  = None
            return 
            
        face_data = self.current_new_face  
 
        def ask_name_with_preview():
            popup = Toplevel(self.root) 
            popup.title("发现新的人脸")
            popup.geometry("400x600") 
            popup.attributes("-topmost",  True)
            popup.grab_set() 
            popup.resizable(False, False)  # 禁止调整大小
            
            # 保存弹窗窗口对象引用
            self.new_face_popup_window = popup

            # 创建主框架
            main_frame = tk.Frame(popup)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # 标题
            title_label = Label(main_frame, text="发现新的人脸", font=('Arial', 16, 'bold'))
            title_label.pack(pady=(0, 20))

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
                pil_img = Image.fromarray(face_img).resize((250,  250))
                tk_img = ImageTk.PhotoImage(pil_img)
                img_label = Label(main_frame, image=tk_img)
                img_label.image  = tk_img 
                img_label.pack(pady=(0, 20)) 
            except Exception as e:
                error_label = Label(main_frame, text="图像显示失败", font=('Arial', 12))
                error_label.pack(pady=(0, 20))

            # 输入提示
            prompt_label = Label(main_frame, text="请输入此人姓名：", font=('Arial', 12))
            prompt_label.pack(pady=(0, 10))
            
            # 输入框
            name_entry = tk.Entry(main_frame, font=('Arial', 12), width=25)
            name_entry.pack(pady=(0, 20)) 
            name_entry.focus_set() 

            # 按钮框架
            button_frame = tk.Frame(main_frame)
            button_frame.pack(pady=(0, 10))
 
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
                
                # 处理完成，清理状态
                self.is_processing_new_face  = False 
                self.current_new_face  = None 
                self.new_face_popup_window = None

            def on_cancel():
                logging.info("用户取消添加新面孔")
                popup.destroy() 
                # 处理完成，清理状态
                self.is_processing_new_face  = False 
                self.current_new_face  = None 
                self.new_face_popup_window = None

            # 确认按钮
            confirm_btn = tk.Button(button_frame, text="确认添加", command=on_confirm, 
                                  font=('Arial', 11), width=12, bg='green', fg='white')
            confirm_btn.pack(side=tk.LEFT, padx=5)
            
            # 跳过按钮
            cancel_btn = tk.Button(button_frame, text="跳过", command=on_cancel, 
                                 font=('Arial', 11), width=12, bg='gray', fg='white')
            cancel_btn.pack(side=tk.LEFT, padx=5)
            
            # 绑定回车键
            popup.bind('<Return>',  lambda e: on_confirm())
            popup.protocol("WM_DELETE_WINDOW",  on_cancel)
 
        self.root.after(0,  ask_name_with_preview)
 
    def check_popup_status(self):
        """检查弹窗状态，确保状态一致性"""
        # 如果弹窗窗口对象存在但窗口已关闭，清理状态
        if self.new_face_popup_window is not None:
            try:
                # 尝试获取窗口状态，如果窗口已关闭会抛出异常
                self.new_face_popup_window.winfo_exists()
            except:
                # 窗口已关闭，清理状态
                self.is_processing_new_face = False
                self.current_new_face = None
                self.new_face_popup_window = None
                logging.debug("检测到弹窗已关闭，已清理状态")

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
        # 检查日志轮转
        log_manager.check_and_rotate()
        
        # 检查弹窗状态，确保状态一致性
        self.check_popup_status()
        
        self.canvas.delete("all") 
        img = self.get_screen() 
        scale = self.image_scale if self.cpu_optimization else 0.5
        img_small = cv2.resize(img,  (0, 0), fx=scale, fy=scale)
        
        # 人脸检测 
        faces = cnn_face_detector(img_small, 0)
        
        # 记录检测到的人脸数量
        if len(faces) > 0:
            logging.debug(f"检测到 {len(faces)} 个人脸")
        
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
                
                if min_dist < self.recognition_threshold:  # 识别阈值 
                    idx = distances.index(min_dist) 
                    name = self.face_name_known_list[idx] 
                    known = True 
                    
                    # 记录识别结果
                    logging.info(f"识别到已知人脸: {name} (距离: {min_dist:.3f})")
                    
                    if name not in self.shown_faces  and self.show_popup: 
                        self.root.after(100,  lambda n=name, i=idx: self.show_face_info(n,  i))
                else:
                    # 记录未知人脸
                    logging.debug(f"检测到未知人脸 (最小距离: {min_dist:.3f}, 阈值: {self.recognition_threshold})")
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
        self.root.after(self.process_interval,  self.process_frame) 
 
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
        
        # 只在开启状态显示时绘制状态信息
        if self.show_status_display:
            # 创建半透明背景
            bg_width = 310
            bg_height = 180  # 增加高度以容纳GPU信息
            self.canvas.create_rectangle(
                10, 10, 10 + bg_width, 10 + bg_height,
                fill='black', outline='white', width=2, stipple='gray50'
            )
            
            # 显示统计信息 
            info = f"检测到人脸: {self.current_frame_face_cnt}  | FPS: {self.fps_show:.1f}" 
            self.canvas.create_text( 
                20, 25, 
                text=info, 
                fill='lime', 
                font=('Arial', 12, 'bold'), 
                anchor='nw'
            )
            
            # 显示GPU状态
            gpu_status = f"GPU: {'可用' if gpu_available else '不可用'}"
            if gpu_available:
                gpu_status += f" ({gpu_count}个设备)"
            gpu_color = 'green' if gpu_available else 'red'
            self.canvas.create_text( 
                20, 50, 
                text=gpu_status, 
                fill=gpu_color, 
                font=('Arial', 11, 'bold'), 
                anchor='nw'
            )
            
            # 显示弹窗状态
            popup_status = "弹窗显示: 开启" if self.show_popup else "弹窗显示: 关闭"
            if self.is_processing_new_face or self.new_face_popup_window is not None:
                popup_status += " | 新面孔处理中"
            self.canvas.create_text( 
                20, 75, 
                text=popup_status, 
                fill='orange', 
                font=('Arial', 11, 'bold'), 
                anchor='nw'
            )
            
            # 显示自动发现新面孔状态
            auto_add_status = "自动发现新面孔: 开启" if self.auto_add_new_faces else "自动发现新面孔: 关闭"
            self.canvas.create_text( 
                20, 100, 
                text=auto_add_status, 
                fill='cyan', 
                font=('Arial', 11, 'bold'), 
                anchor='nw'
            )
            
            # 显示当前识别阈值
            threshold_status = f"识别阈值: {self.recognition_threshold:.2f}"
            self.canvas.create_text( 
                20, 125, 
                text=threshold_status, 
                fill='magenta', 
                font=('Arial', 11, 'bold'), 
                anchor='nw'
            )
            
            # 显示运行模式
            mode_status = f"运行模式: {'GPU加速' if gpu_available else 'CPU优化'} (间隔:{self.process_interval}ms)"
            self.canvas.create_text( 
                20, 150, 
                text=mode_status, 
                fill='yellow', 
                font=('Arial', 10, 'bold'), 
                anchor='nw'
            )
 
    def run(self):
        """运行主循环"""
        self.process_frame() 
        self.root.mainloop() 
 
    def regenerate_csv_from_images(self):
        """从图像文件夹重新生成CSV文件"""
        print("正在从图像文件夹重新生成人脸特征文件...")
        try:
            # 清空现有数据
            self.face_name_known_list.clear()
            self.face_feature_known_list.clear()
            self.face_image_path_list.clear()
            self.processed_features.clear()
            
            # 获取所有人员文件夹
            person_folders = []
            data_faces_path = "data/data_faces_from_camera/"
            if os.path.exists(data_faces_path):
                person_folders = [f for f in os.listdir(data_faces_path) if f.startswith("person_")]
            
            if not person_folders:
                print("没有找到任何人脸图像文件夹")
                return False
            
            # 创建新的CSV文件
            csv_path = "data/features_all.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                for person_folder in person_folders:
                    person_name = person_folder.split('_', 1)[1] if '_' in person_folder else person_folder
                    folder_path = os.path.join(data_faces_path, person_folder)
                    
                    # 获取该人员的所有图像
                    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if not image_files:
                        print(f"警告: {person_folder} 文件夹中没有找到图像文件")
                        continue
                    
                    # 提取该人员的所有特征
                    features_list = []
                    for img_file in image_files:
                        img_path = os.path.join(folder_path, img_file)
                        try:
                            # 读取图像
                            img = cv2.imread(img_path)
                            if img is None:
                                print(f"警告: 无法读取图像 {img_path}")
                                continue
                            
                            # 检测人脸 - 尝试不同尺寸
                            faces = cnn_face_detector(img, 0)
                            
                            # 如果原始尺寸没有检测到，尝试缩小图像
                            if not faces:
                                img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                                faces = cnn_face_detector(img_small, 0)
                                if faces:
                                    # 如果缩小后检测到，将坐标放大回原始尺寸
                                    for face in faces:
                                        rect = face.rect
                                        face.rect = dlib.rectangle(
                                            int(rect.left() * 2),
                                            int(rect.top() * 2),
                                            int(rect.right() * 2),
                                            int(rect.bottom() * 2)
                                        )
                            
                            # 如果还是没有检测到，尝试放大图像
                            if not faces:
                                img_large = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)
                                faces = cnn_face_detector(img_large, 0)
                                if faces:
                                    # 如果放大后检测到，将坐标缩小回原始尺寸
                                    for face in faces:
                                        rect = face.rect
                                        face.rect = dlib.rectangle(
                                            int(rect.left() / 2),
                                            int(rect.top() / 2),
                                            int(rect.right() / 2),
                                            int(rect.bottom() / 2)
                                        )
                            
                            if not faces:
                                print(f"警告: 在图像 {img_path} 中没有检测到人脸")
                                continue
                            
                            # 提取特征
                            shape = predictor(img, faces[0].rect)
                            feature = face_reco_model.compute_face_descriptor(img, shape)
                            features_list.append(feature)
                            
                        except Exception as e:
                            print(f"警告: 处理图像 {img_path} 时出错: {str(e)}")
                            continue
                    
                    if features_list:
                        # 计算平均特征
                        avg_feature = np.mean(features_list, axis=0)
                        
                        # 写入CSV
                        writer.writerow([person_name] + list(avg_feature))
                        
                        # 更新内存数据
                        self.face_name_known_list.append(person_name)
                        self.face_feature_known_list.append(list(avg_feature))
                        self.face_image_path_list.append(os.path.join(folder_path, image_files[0]))
                        
                        feature_str = ','.join(map(str, avg_feature))
                        self.processed_features.add(feature_str)
                        
                        print(f"已处理 {person_name}: {len(features_list)} 张图像")
                    else:
                        print(f"警告: {person_name} 没有成功提取到特征")
            
            print(f"CSV文件重新生成完成，共处理 {len(self.face_name_known_list)} 个人")
            return True
            
        except Exception as e:
            print(f"重新生成CSV文件时出错: {str(e)}")
            return False

def detect_gpu_availability():
    """检测GPU可用性"""
    try:
        # 检查dlib是否支持CUDA
        if hasattr(dlib, 'DLIB_USE_CUDA') and dlib.DLIB_USE_CUDA:
            # 检查CUDA设备数量
            if hasattr(dlib, 'cuda') and hasattr(dlib.cuda, 'get_num_devices'):
                num_devices = dlib.cuda.get_num_devices()
                if num_devices > 0:
                    logging.info(f"检测到 {num_devices} 个CUDA设备，启用GPU加速")
                    return True, num_devices
                else:
                    logging.warning("dlib支持CUDA但未检测到可用的GPU设备")
                    return False, 0
            else:
                logging.warning("dlib支持CUDA但无法获取设备信息")
                return False, 0
        else:
            logging.info("dlib未编译CUDA支持，使用CPU模式")
            return False, 0
    except Exception as e:
        logging.error(f"GPU检测过程中出错: {e}")
        return False, 0

# 检测GPU可用性
gpu_available, gpu_count = detect_gpu_availability()

def main():
    try:
        logging.info("=" * 50)
        logging.info("人脸识别监控系统启动")
        logging.info("=" * 50)
        
        # 启动时重做CSV文件
        print("正在重新生成人脸特征文件...")
        logging.info("开始重新生成人脸特征文件...")
        
        # 创建主实例
        recognizer = TransparentFaceRecognizer()
        
        # 重新生成CSV文件
        if recognizer.regenerate_csv_from_images():
            print("人脸特征文件重新生成成功！")
            logging.info("人脸特征文件重新生成成功")
        else:
            print("人脸特征文件重新生成失败，继续使用现有特征文件...")
            logging.warning("人脸特征文件重新生成失败，继续使用现有特征文件")
        
        print("启动人脸识别监控系统...")
        logging.info("开始运行人脸识别监控系统")
        recognizer.run() 
    except Exception as e:
        logging.error(f" 程序运行出错: {str(e)}")
        logging.error("程序异常退出")
        raise
    finally:
        logging.info("=" * 50)
        logging.info("人脸识别监控系统退出")
        logging.info("=" * 50)
 
if __name__ == '__main__':
    main()