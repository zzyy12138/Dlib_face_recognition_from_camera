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
from tkinter import ttk
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
        
        self.manual_add_item = pystray.MenuItem(
            "手动添加人脸",
            self.manual_add_face
        )
        
        menu = pystray.Menu(
            self.toggle_popup_item, 
            self.toggle_auto_add_item,
            self.threshold_item,
            self.toggle_status_item,
            pystray.MenuItem('手动添加人脸', self.manual_add_face),
            pystray.MenuItem('打开人脸数据文件夹', self.open_faces_folder),
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
 
    def manual_add_face(self, icon=None, item=None):
        """手动添加人脸"""
        def run_face_collector():
            try:
                logging.info("启动手动添加人脸工具")
                
                # 导入并运行人脸采集工具
                import subprocess
                import sys
                
                # 获取当前脚本的目录
                script_dir = os.path.dirname(os.path.abspath(__file__))
                face_collector_script = os.path.join(script_dir, "face_collector_from_image.py")
                
                if os.path.exists(face_collector_script):
                    # 使用subprocess启动人脸采集工具
                    process = subprocess.Popen([sys.executable, face_collector_script])
                    
                    # 等待进程完成
                    process.wait()
                    
                    # 进程完成后，显示进度条并重新加载人脸数据库
                    self.show_loading_progress("正在更新人脸库...", self.reload_face_database)
                    
                else:
                    logging.error(f"找不到人脸采集工具脚本: {face_collector_script}")
                    # 如果找不到脚本，显示错误信息
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("错误", f"找不到人脸采集工具脚本:\n{face_collector_script}")
                    
            except Exception as e:
                logging.error(f"启动人脸采集工具时出错: {str(e)}")
                import tkinter.messagebox as messagebox
                messagebox.showerror("错误", f"启动人脸采集工具时出错:\n{str(e)}")
        
        # 在新线程中运行，避免阻塞主程序
        threading.Thread(target=run_face_collector, daemon=True).start()
 
    def open_faces_folder(self, icon=None, item=None):
        """打开存放人脸图像的文件夹"""
        folder_path = "data/data_faces_from_camera"
        try:
            abs_path = os.path.abspath(folder_path)
            if not os.path.exists(abs_path):
                os.makedirs(abs_path)
                logging.info(f"文件夹不存在，已创建: {abs_path}")
            
            os.startfile(abs_path)
            logging.info(f"已请求打开文件夹: {abs_path}")
        except Exception as e:
            logging.error(f"无法打开文件夹 {folder_path}: {e}")
            import tkinter.messagebox as messagebox
            messagebox.showerror("错误", f"无法打开文件夹:\n{os.path.abspath(folder_path)}\n\n错误: {e}")
 
    def show_loading_progress(self, message, task_function):
        """显示加载进度条（直接在主窗口上显示）"""
        def show_progress():
            # 清空画布
            self.canvas.delete("all")
            
            # 创建半透明背景
            bg_width = 500
            bg_height = 200
            x = (self.screen_width - bg_width) // 2
            y = (self.screen_height - bg_height) // 2
            
            # 绘制背景
            self.canvas.create_rectangle(
                x, y, x + bg_width, y + bg_height,
                fill='black', outline='white', width=2, stipple='gray50'
            )
            
            # 标题
            self.canvas.create_text(
                self.screen_width // 2, y + 30,
                text="人脸识别监控系统初始化", 
                fill='lime', 
                font=('Arial', 16, 'bold')
            )
            
            # 消息
            self.canvas.create_text(
                self.screen_width // 2, y + 70,
                text=message, 
                fill='white', 
                font=('Arial', 12)
            )
            
            # 进度条背景
            progress_width = 400
            progress_height = 20
            progress_x = (self.screen_width - progress_width) // 2
            progress_y = y + 110
            
            self.canvas.create_rectangle(
                progress_x, progress_y, progress_x + progress_width, progress_y + progress_height,
                fill='gray', outline='white', width=1
            )
            
            # 进度条（动画效果）
            def animate_progress():
                # 创建移动的进度条
                bar_width = 50
                bar_x = progress_x + 5
                bar_y = progress_y + 2
                bar_height = progress_height - 4
                
                # 删除之前的进度条
                self.canvas.delete("progress_bar")
                
                # 绘制新的进度条
                self.canvas.create_rectangle(
                    bar_x, bar_y, bar_x + bar_width, bar_y + bar_height,
                    fill='lime', outline='', tags="progress_bar"
                )
                
                # 更新位置
                bar_x += 10
                if bar_x > progress_x + progress_width - bar_width - 5:
                    bar_x = progress_x + 5
                
                # 状态文本
                self.canvas.create_text(
                    self.screen_width // 2, y + 150,
                    text="请稍候...", 
                    fill='cyan', 
                    font=('Arial', 10),
                    tags="status_text"
                )
                
                # 继续动画
                if hasattr(self, 'progress_active') and self.progress_active:
                    self.root.after(100, animate_progress)
            
            # 开始动画
            self.progress_active = True
            animate_progress()
            
            def update_status(text):
                # 更新状态文本
                self.canvas.delete("status_text")
                self.canvas.create_text(
                    self.screen_width // 2, y + 150,
                    text=text, 
                    fill='cyan', 
                    font=('Arial', 10),
                    tags="status_text"
                )
                self.root.update()
            
            def run_task():
                try:
                    update_status("正在重新加载人脸数据库...")
                    task_function()
                    update_status("初始化完成！")
                    # 停止动画
                    self.progress_active = False
                    # 1秒后清除进度条
                    self.root.after(1000, lambda: self.canvas.delete("all"))
                except Exception as e:
                    update_status(f"初始化失败: {str(e)}")
                    self.progress_active = False
                    # 2秒后清除进度条
                    self.root.after(2000, lambda: self.canvas.delete("all"))
            
            # 在新线程中运行任务
            threading.Thread(target=run_task, daemon=True).start()
        
        # 在主线程中显示进度条
        self.root.after(0, show_progress)
 
    def reload_face_database(self):
        """重新加载人脸数据库"""
        try:
            logging.info("开始重新加载人脸数据库")
            
            # 清空现有数据
            self.face_name_known_list.clear()
            self.face_feature_known_list.clear()
            self.face_image_path_list.clear()
            self.processed_features.clear()
            
            # 重新加载数据库
            self.get_face_database()
            
            # 重新生成CSV文件
            if self.regenerate_csv_from_images():
                logging.info("人脸数据库重新加载成功")
            else:
                logging.warning("人脸数据库重新加载失败")
                
        except Exception as e:
            logging.error(f"重新加载人脸数据库时出错: {str(e)}")
 
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
                        
                        # 构建图像路径 - 支持新旧格式
                        if '_' in name and name.count('_') >= 1:
                            # 新格式: 姓名_身份证号
                            folder_name = f"person_{name}"
                        else:
                            # 旧格式: 数字编号
                            folder_name = f"person_{name}"
                        
                        self.face_image_path_list.append(f"data/data_faces_from_camera/{folder_name}/img_face_1.jpg")  
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
                    # 使用imdecode读取图像，避免中文路径问题
                    img_path_abs = os.path.abspath(img_path)
                    img_array = np.fromfile(img_path_abs, dtype=np.uint8)
                    pil_img = Image.open(img_path).resize((200,  200))
                    tk_img = ImageTk.PhotoImage(pil_img)
                    label_img = Label(popup, image=tk_img)
                    label_img.image  = tk_img 
                    label_img.pack(pady=10) 
                except Exception as e:
                    Label(popup, text=f"图片加载失败: {str(e)}").pack(pady=10)
            else:
                Label(popup, text="无图片").pack(pady=10)
            
            # 解析姓名和身份证号
            if '_' in name and name.count('_') >= 1:
                # 新格式: 姓名_身份证号
                name_parts = name.split('_', 1)
                display_name = name_parts[0]
                id_number = name_parts[1]
                info = f"姓名: {display_name}\n身份证号: {id_number}\n识别时间: {time.strftime('%Y-%m-%d  %H:%M:%S')}"
            else:
                # 旧格式或其他格式
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
            popup.geometry("800x800") 
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
 
            # 显示新检测到的人脸图像 
            try:
                pil_img = Image.fromarray(face_img).resize((200,  200))
                tk_img = ImageTk.PhotoImage(pil_img)
                img_label = Label(main_frame, image=tk_img)
                img_label.image  = tk_img 
                img_label.pack(pady=(0, 20)) 
            except Exception as e:
                error_label = Label(main_frame, text="图像显示失败", font=('Arial', 12))
                error_label.pack(pady=(0, 20))

            # 创建选项卡
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

            # 选项卡1：创建新文件夹
            new_folder_frame = ttk.Frame(notebook)
            notebook.add(new_folder_frame, text="创建新文件夹")

            # 选项卡2：保存到已有文件夹
            existing_folder_frame = ttk.Frame(notebook)
            notebook.add(existing_folder_frame, text="保存到已有文件夹")

            # ===== 选项卡1：创建新文件夹 =====
            # 创建一个容器框架，并使用grid布局以更好地对齐
            input_container = tk.Frame(new_folder_frame)
            input_container.pack(pady=40, padx=30, fill=tk.X, expand=True)
            
            prompt_label = Label(input_container, text="请输入此人信息：", font=('Arial', 12, 'bold'))
            prompt_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 20))

            # 姓名
            name_label = Label(input_container, text="姓名:", font=('Arial', 11))
            name_label.grid(row=1, column=0, sticky='e', pady=5)
            name_entry = tk.Entry(input_container, font=('Arial', 11))
            name_entry.grid(row=1, column=1, sticky='ew', pady=5, padx=(10, 0))
            name_entry.focus_set()

            # 身份证号
            id_label = Label(input_container, text="身份证号:", font=('Arial', 11))
            id_label.grid(row=2, column=0, sticky='e', pady=5)
            id_entry = tk.Entry(input_container, font=('Arial', 11))
            id_entry.grid(row=2, column=1, sticky='ew', pady=5, padx=(10, 0))

            # 配置列的权重，使输入框可以水平扩展
            input_container.columnconfigure(1, weight=1)

            # ===== 选项卡2：保存到已有文件夹 =====
            # 获取已有文件夹列表
            existing_folders = []
            data_faces_path = "data/data_faces_from_camera/"
            if os.path.exists(data_faces_path):
                existing_folders = [f for f in os.listdir(data_faces_path) if f.startswith("person_")]
            
            if existing_folders:
                # 创建左右分栏布局
                content_frame = tk.Frame(existing_folder_frame)
                content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
                
                # 左侧：文件夹列表
                left_frame = tk.Frame(content_frame)
                left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
                
                # 文件夹选择提示
                select_label = Label(left_frame, text="选择要保存到的文件夹：", font=('Arial', 12, 'bold'))
                select_label.pack(pady=(0, 10))
                
                # 创建列表框和滚动条的容器，并设置固定高度
                listbox_container = tk.Frame(left_frame, height=250)
                listbox_container.pack(fill=tk.X, expand=False)
                listbox_container.pack_propagate(False)
                
                scrollbar = tk.Scrollbar(listbox_container)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # 列表框，不单独设置高度，它将填充父容器
                folder_listbox = tk.Listbox(listbox_container, yscrollcommand=scrollbar.set, font=('Arial', 10))
                folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.config(command=folder_listbox.yview)
                
                # 填充文件夹列表
                for folder in sorted(existing_folders):
                    # 解析文件夹名称显示
                    folder_parts = folder.split('_', 2)
                    if len(folder_parts) >= 3:
                        display_name = f"{folder_parts[1]}_{folder_parts[2]}"
                    elif len(folder_parts) == 2:
                        display_name = f"未知_{folder_parts[1]}"
                    else:
                        display_name = folder
                    
                    folder_listbox.insert(tk.END, display_name)
                
                # 右侧：预览区域
                right_frame = tk.Frame(content_frame)
                right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
                
                preview_label = Label(right_frame, text="预览：", font=('Arial', 11, 'bold'))
                preview_label.pack(anchor='w', pady=(0, 5))
                
                # 创建一个固定大小的框架来容纳预览图像
                preview_img_frame = tk.Frame(right_frame, width=204, height=204, relief=tk.SUNKEN, borderwidth=2)
                preview_img_frame.pack(pady=(0, 10))
                preview_img_frame.pack_propagate(False) # 防止框架收缩

                # 预览图像标签 - 放置在固定大小的框架中
                preview_img_label = Label(preview_img_frame, text="请选择一个文件夹查看预览", font=('Arial', 10))
                preview_img_label.pack(fill=tk.BOTH, expand=True)
                
                def on_folder_select(event):
                    """当选择文件夹时更新预览"""
                    selection = folder_listbox.curselection()
                    if selection:
                        selected_index = selection[0]
                        selected_folder = existing_folders[selected_index]
                        folder_path = os.path.join(data_faces_path, selected_folder)
                        
                        # 查找第一张图片
                        try:
                            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if image_files:
                                first_img_path = os.path.join(folder_path, image_files[0])
                                # 读取并显示预览图像
                                try:
                                    img_path_abs = os.path.abspath(first_img_path)
                                    img = cv2.imdecode(np.fromfile(img_path_abs, dtype=np.uint8), cv2.IMREAD_COLOR)
                                    if img is not None:
                                        # 转换为RGB并调整大小
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        pil_img = Image.fromarray(img_rgb).resize((200, 200), Image.Resampling.LANCZOS)
                                        tk_img = ImageTk.PhotoImage(pil_img)
                                        preview_img_label.configure(image=tk_img, text="")
                                        preview_img_label.image = tk_img
                                    else:
                                        preview_img_label.configure(image="", text="无法读取图像")
                                except Exception as e:
                                    preview_img_label.configure(image="", text=f"图像加载失败: {str(e)}")
                            else:
                                preview_img_label.configure(image="", text="文件夹中没有图像")
                        except Exception as e:
                            preview_img_label.configure(image="", text=f"无法访问文件夹: {str(e)}")
                
                # 绑定选择事件
                folder_listbox.bind('<<ListboxSelect>>', on_folder_select)
                
            else:
                # 没有现有文件夹时的提示
                no_folder_label = Label(existing_folder_frame, text="没有找到现有的人脸文件夹", font=('Arial', 12))
                no_folder_label.pack(pady=50)

            def on_confirm_new_folder():
                """确认创建新文件夹"""
                name = name_entry.get().strip()
                id_number = id_entry.get().strip()
                
                # 验证输入
                if not name:
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("错误", "请输入姓名")
                    return
                
                if not id_number:
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("错误", "请输入身份证号")
                    return
                
                # 验证身份证号格式（简单验证）
                if len(id_number) != 18:
                    import tkinter.messagebox as messagebox
                    result = messagebox.askyesno("警告", "身份证号长度不是18位，是否继续？")
                    if not result:
                        return
                
                # 创建文件夹名称：姓名_身份证号
                folder_name = f"person_{name}_{id_number}"
                folder = f"data/data_faces_from_camera/{folder_name}"
                os.makedirs(folder,  exist_ok=True)
                
                # 查找可用的文件名 
                img_index = 1 
                while os.path.exists(os.path.join(folder,  f"img_face_{img_index}.jpg")):
                    img_index += 1 
                    
                img_filename = os.path.join(folder,  f"img_face_{img_index}.jpg")
                
                # 保存图像 - 使用imencode避免中文路径问题
                try:
                    # 转换颜色空间
                    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    # 使用imencode保存图像
                    success, encoded_img = cv2.imencode('.jpg', face_img_bgr)
                    if success:
                        with open(img_filename, 'wb') as f:
                            f.write(encoded_img.tobytes())
                    else:
                        raise Exception("图像编码失败")
                except Exception as e:
                    logging.error(f"保存图像失败: {str(e)}")
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("错误", f"保存图像失败: {str(e)}")
                    return
                
                # 更新内存中的数据库 
                self.face_name_known_list.append(f"{name}_{id_number}") 
                self.face_feature_known_list.append(face_data['feature']) 
                self.face_image_path_list.append(img_filename) 
                
                # 更新CSV文件 
                self.update_face_database_csv(f"{name}_{id_number}",  face_data['feature'])
                
                logging.info(f" 新增人脸: {name}_{id_number}，图像保存为 {img_filename}")
                popup.destroy() 
                
                # 处理完成，清理状态
                self.is_processing_new_face  = False 
                self.current_new_face  = None 
                self.new_face_popup_window = None

            def on_confirm_existing_folder():
                """确认保存到已有文件夹"""
                selection = folder_listbox.curselection()
                if not selection:
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("错误", "请选择一个文件夹")
                    return
                
                selected_index = selection[0]
                selected_folder = existing_folders[selected_index]
                folder_path = os.path.join(data_faces_path, selected_folder)
                
                # 解析文件夹名称获取人员信息
                folder_parts = selected_folder.split('_', 2)
                if len(folder_parts) >= 3:
                    person_name = f"{folder_parts[1]}_{folder_parts[2]}"
                elif len(folder_parts) == 2:
                    person_name = folder_parts[1]
                else:
                    person_name = selected_folder
                
                # 查找可用的文件名 
                img_index = 1 
                while os.path.exists(os.path.join(folder_path,  f"img_face_{img_index}.jpg")):
                    img_index += 1 
                    
                img_filename = os.path.join(folder_path,  f"img_face_{img_index}.jpg")
                
                # 保存图像
                try:
                    # 转换颜色空间
                    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    # 使用imencode保存图像
                    success, encoded_img = cv2.imencode('.jpg', face_img_bgr)
                    if success:
                        with open(img_filename, 'wb') as f:
                            f.write(encoded_img.tobytes())
                    else:
                        raise Exception("图像编码失败")
                except Exception as e:
                    logging.error(f"保存图像失败: {str(e)}")
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("错误", f"保存图像失败: {str(e)}")
                    return
                
                # 更新内存中的数据库 - 如果该人员已存在，更新特征；否则添加新记录
                if person_name in self.face_name_known_list:
                    # 更新现有记录的特征（取平均值）
                    idx = self.face_name_known_list.index(person_name)
                    old_feature = self.face_feature_known_list[idx]
                    new_feature = np.mean([old_feature, face_data['feature']], axis=0)
                    self.face_feature_known_list[idx] = list(new_feature)
                    logging.info(f" 更新人脸特征: {person_name}")
                else:
                    # 添加新记录
                    self.face_name_known_list.append(person_name) 
                    self.face_feature_known_list.append(face_data['feature']) 
                    self.face_image_path_list.append(img_filename) 
                    logging.info(f" 新增人脸: {person_name}，图像保存为 {img_filename}")
                
                # 更新CSV文件 
                self.update_face_database_csv(person_name, face_data['feature'])
                
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

            # 按钮框架
            button_frame = tk.Frame(main_frame)
            button_frame.pack(pady=(0, 10))
            
            # 创建新文件夹按钮
            new_folder_btn = tk.Button(button_frame, text="创建新文件夹", command=on_confirm_new_folder, 
                                     font=('Arial', 11), width=15, bg='green', fg='white')
            new_folder_btn.pack(side=tk.LEFT, padx=5)
            
            # 保存到已有文件夹按钮（仅在有现有文件夹时显示）
            if existing_folders:
                existing_folder_btn = tk.Button(button_frame, text="保存到选中文件夹", command=on_confirm_existing_folder, 
                                             font=('Arial', 11), width=15, bg='blue', fg='white')
                existing_folder_btn.pack(side=tk.LEFT, padx=5)
            
            # 跳过按钮
            cancel_btn = tk.Button(button_frame, text="跳过", command=on_cancel, 
                                 font=('Arial', 11), width=12, bg='gray', fg='white')
            cancel_btn.pack(side=tk.LEFT, padx=5)
            
            # 绑定回车键到新文件夹确认
            popup.bind('<Return>',  lambda e: on_confirm_new_folder())
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
        
        # 如果正在显示进度条，跳过人脸检测
        if hasattr(self, 'progress_active') and self.progress_active:
            self.root.after(self.process_interval, self.process_frame)
            return
        
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
            feature = face_reco_model.compute_face_descriptor(img, shape)
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
                    # logging.info(f"识别到已知人脸: {name} (距离: {min_dist:.3f})")
                    
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
            
            # 解析显示名称
            display_name = self.current_frame_face_name_list[i]
            if '_' in display_name and display_name.count('_') >= 1:
                # 新格式: 姓名_身份证号，只显示姓名
                name_parts = display_name.split('_', 1)
                display_name = name_parts[0]
            
            self.canvas.create_text( 
                left, bottom + 20, 
                text=display_name, 
                fill='yellow', 
                font=('Arial', 12, 'bold'), 
                anchor='nw'
            )
        
        # 只在开启状态显示时绘制状态信息
        if self.show_status_display:
            # 创建半透明背景
            bg_width = 310
            bg_height = 180  # 恢复原来的高度
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
                    # 解析文件夹名称
                    # 格式1: person_姓名_身份证号
                    # 格式2: person_数字编号 (兼容旧格式)
                    folder_parts = person_folder.split('_', 2)  # 最多分割2次
                    
                    if len(folder_parts) >= 3:
                        # 新格式: person_姓名_身份证号
                        person_name = f"{folder_parts[1]}_{folder_parts[2]}"
                        display_name = f"{folder_parts[1]}_{folder_parts[2]}"
                    elif len(folder_parts) == 2:
                        # 旧格式: person_数字编号
                        person_name = folder_parts[1]
                        display_name = f"未知_{folder_parts[1]}"
                    else:
                        # 异常格式，跳过
                        print(f"警告: 跳过异常格式的文件夹 {person_folder}")
                        continue
                    
                    folder_path = os.path.join(data_faces_path, person_folder)
                    
                    # 获取该人员的所有图像
                    image_files = []
                    try:
                        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    except Exception as e:
                        print(f"警告: 无法读取文件夹 {person_folder}: {str(e)}")
                        # 即使无法读取文件夹，也继续处理，生成空数据
                    
                    # 提取该人员的所有特征
                    features_list = []
                    processed_images = 0
                    
                    if image_files:
                        for img_file in image_files:
                            img_path = os.path.join(folder_path, img_file)
                            try:
                                # 读取图像 - 使用绝对路径避免编码问题
                                img_path_abs = os.path.abspath(img_path)
                                img = cv2.imdecode(np.fromfile(img_path_abs, dtype=np.uint8), cv2.IMREAD_COLOR)
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
                                processed_images += 1
                                
                            except Exception as e:
                                print(f"警告: 处理图像 {img_path} 时出错: {str(e)}")
                                continue
                    
                    # 无论是否成功提取到特征，都生成CSV数据
                    if features_list:
                        # 计算平均特征
                        avg_feature = np.mean(features_list, axis=0)
                        print(f"已处理 {display_name}: {processed_images} 张图像成功提取特征")
                    else:
                        # 生成128维的零向量作为默认特征
                        avg_feature = np.zeros(128)
                        print(f"警告: {display_name} 没有成功提取到特征，使用默认零向量")
                    
                    # 写入CSV - 无论是否有特征都写入
                    writer.writerow([person_name] + list(avg_feature))
                    
                    # 更新内存数据
                    self.face_name_known_list.append(person_name)
                    self.face_feature_known_list.append(list(avg_feature))
                    
                    # 设置图像路径 - 如果有图像文件则使用第一个，否则使用默认路径
                    if image_files:
                        self.face_image_path_list.append(os.path.join(folder_path, image_files[0]))
                    else:
                        # 创建一个默认的图像路径，即使文件不存在
                        default_img_path = os.path.join(folder_path, "img_face_1.jpg")
                        self.face_image_path_list.append(default_img_path)
                    
                    feature_str = ','.join(map(str, avg_feature))
                    self.processed_features.add(feature_str)
            
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
        
        # 创建主实例
        recognizer = TransparentFaceRecognizer()
        
        # 启动时重做CSV文件，显示进度条
        print("正在重新生成人脸特征文件...")
        logging.info("开始重新生成人脸特征文件...")
        
        # 显示启动进度条
        recognizer.show_loading_progress("正在初始化人脸库...", lambda: recognizer.regenerate_csv_from_images())
        
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