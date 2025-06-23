"""
基于Dlib的人脸识别监控程序
功能：
1. 实时捕获屏幕画面并进行人脸检测 
2. 对检测到的人脸进行识别和跟踪 
3. 支持透明窗口显示，不影响其他操作 
4. 系统托盘图标控制 
5. 智能管理新面孔弹窗，避免过多弹窗 
6. 自动API调用获取真实身份信息
7. 使用SQLite数据库存储人脸数据
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
import base64
import json
import io
import sqlite3

# 导入数据库管理器
from face_database_manager import FaceDatabaseManager

# 尝试导入requests库，如果失败则禁用API功能
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("警告: requests库未安装，API功能将被禁用")
    print("如需启用API功能，请运行: pip install requests flask flask-cors")

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
        # 初始化数据库管理器
        self.db_manager = FaceDatabaseManager()
        
        # 人脸数据库相关 - 从数据库加载
        self.face_feature_known_list = []
        self.face_name_known_list = []
        self.face_image_data_list = []  # 存储图像数据而不是路径
        self.real_name_known_list = []  # 存储真实姓名
        
        # 当前帧数据 
        self.current_frame_face_feature_list = []
        self.current_frame_face_cnt = 0 
        self.current_frame_face_name_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_known_list = []
        
        # 性能统计 
        self.fps_show = 0 
        self.frame_cnt = 0 
        self.start_time = time.time() 
        
        # 屏幕捕获 
        self.sct = mss.mss() 
        self.screen_width = pyautogui.size().width  
        self.screen_height = pyautogui.size().height  
        self.monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height} 
        
        # 界面相关 
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)
        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.7)
        self.root.attributes("-transparentcolor", "white")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True) 
        self.canvas = Canvas(self.root, bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 识别阈值设置
        self.recognition_threshold = 0.5  # 默认识别阈值
        
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
        self.last_new_face_time = 0 
        self.new_face_cooldown = 5  # 新面孔检测冷却时间(秒)
        self.current_new_face = None  # 当前正在处理的新面孔 
        self.is_processing_new_face = False  # 是否正在处理新面孔 
        self.new_face_popup_window = None  # 当前新面孔弹窗窗口对象
        self.shown_faces = set()  # 已显示的面孔 
        self.show_popup = False  # 是否显示弹窗 - 默认关闭
        self.auto_add_new_faces = False  # 是否自动识别添加新面孔 - 默认关闭
        self.processed_features = set()  # 已处理的特征集合
        
        # API相关配置
        self.api_enabled = REQUESTS_AVAILABLE  # 是否启用API调用
        self.api_url = "http://localhost:5000/api/recognize_face"  # API地址
        self.api_timeout = 10  # API请求超时时间(秒)
        self.api_retry_count = 3  # API重试次数
        self.temp_faces = {}  # 临时存储的人脸信息 {feature_str: {'temp_name': 'xxx', 'temp_id': 'xxx', 'face_img': img_array}}
        self.temp_user_counter = 1  # 临时用户计数器
        
        # 清理计时器
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 3600  # 每小时清理一次临时文件
        
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
        logging.info(f"API库可用: {'是' if REQUESTS_AVAILABLE else '否'}")
        logging.info(f"API调用: {'开启' if self.api_enabled else '关闭'}")
        if self.api_enabled:
            logging.info(f"API地址: {self.api_url}")
            logging.info(f"API超时: {self.api_timeout}秒")
 
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
        
        # 删除API调用相关菜单项
        # self.toggle_api_item = pystray.MenuItem(
        #     lambda item: f"{'关闭' if self.api_enabled else '开启'}API调用",
        #     self.toggle_api_enabled
        # )
        
        menu = pystray.Menu(
            self.toggle_popup_item, 
            self.toggle_auto_add_item,
            self.threshold_item,
            self.toggle_status_item,
            # self.toggle_api_item,  # 删除API调用菜单项
            pystray.MenuItem('手动添加人脸', self.manual_add_face),
            pystray.MenuItem('人脸库管理器', self.open_faces_folder),
            pystray.MenuItem('清理所有临时身份', self.clear_all_temp_identities),
            pystray.MenuItem('清空数据库', self.clear_database),
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
 
    def toggle_api_enabled(self, icon=None, item=None):
        """切换API调用状态"""
        self.api_enabled = not self.api_enabled
        if icon:
            icon.update_menu()
        logging.info(f" API调用已 {'开启' if self.api_enabled else '关闭'}")

    def generate_temp_identity(self):
        """生成临时身份信息"""
        # 生成类似 unknown1, unknown2 的临时姓名
        temp_name = f"unknown{self.temp_user_counter}"
        
        # 生成临时身份证号
        temp_id = "TEMP" + str(random.randint(100000, 999999))
        
        # 增加计数器
        self.temp_user_counter += 1
        
        return temp_name, temp_id

    def image_to_base64(self, img_array):
        """将图像数组转换为base64编码"""
        try:
            # 确保图像是BGR格式（OpenCV默认格式）
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # 转换为RGB格式
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img_array
            
            # 编码为JPEG格式
            success, buffer = cv2.imencode('.jpg', img_rgb)
            if success:
                # 转换为base64
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return img_base64
            else:
                logging.error("图像编码失败")
                return None
        except Exception as e:
            logging.error(f"图像转base64失败: {str(e)}")
            return None

    def call_face_recognition_api(self, face_img):
        """调用人脸识别API"""
        if not self.api_enabled:
            logging.debug("API调用已禁用")
            return None
        
        if not REQUESTS_AVAILABLE:
            logging.warning("requests库不可用，无法调用API")
            return None
        
        try:
            # 转换图像为base64
            img_base64 = self.image_to_base64(face_img)
            if not img_base64:
                logging.error("图像转base64失败")
                return None
            
            # 准备请求数据
            request_data = {
                'image_base64': img_base64
            }
            
            # 发送API请求
            logging.info("正在调用人脸识别API...")
            response = requests.post(
                self.api_url,
                json=request_data,
                timeout=self.api_timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    data = result.get('data', {})
                    name = data.get('name', '')
                    id_card = data.get('id_card', '')
                    confidence = data.get('confidence', 0.0)
                    processing_time = data.get('processing_time', 0.0)
                    
                    logging.info(f"API识别成功: {name} - {id_card} (置信度: {confidence:.3f}, 耗时: {processing_time:.2f}s)")
                    return {
                        'name': name,
                        'id_card': id_card,
                        'confidence': confidence,
                        'processing_time': processing_time
                    }
                else:
                    error_msg = result.get('error', '未知错误')
                    logging.warning(f"API识别失败: {error_msg}")
                    return None
            else:
                logging.error(f"API请求失败，状态码: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logging.error("API请求超时")
            return None
        except requests.exceptions.ConnectionError:
            logging.error("无法连接到API服务器")
            return None
        except Exception as e:
            logging.error(f"API调用出错: {str(e)}")
            return None

    def update_face_with_api_result(self, feature_str, api_result):
        """使用API结果更新人脸信息"""
        if not api_result:
            logging.warning("API结果为空，无法更新人脸信息")
            return False
        
        try:
            # 检查临时人脸是否存在
            if feature_str not in self.temp_faces:
                logging.warning(f"临时人脸 {feature_str} 不存在")
                return False
            
            temp_face = self.temp_faces[feature_str]
            real_name = api_result['name']
            real_id_card = api_result['id_card']
            person_id = temp_face['person_id']
            
            # 检查是否已存在相同身份
            existing_person = self.db_manager.get_person_by_name_id(real_name, real_id_card)
            
            if existing_person:
                # 身份已存在，更新现有记录
                logging.info(f"发现相同身份 {real_name} - {real_id_card}，更新现有记录")
                
                # 删除临时人员记录
                self.db_manager.delete_person(person_id)
                
                # 更新内存数据库
                temp_person_name = f"{temp_face['temp_name']}_{temp_face['temp_id']}"
                if temp_person_name in self.face_name_known_list:
                    temp_index = self.face_name_known_list.index(temp_person_name)
                    self.face_name_known_list.pop(temp_index)
                    self.face_feature_known_list.pop(temp_index)
                    self.face_image_data_list.pop(temp_index)
                    self.real_name_known_list.pop(temp_index)  # 同时清理真实姓名列表
                    logging.debug(f"已从内存数据库中移除临时身份: {temp_person_name}")
                
                # 更新现有人员的特征
                self.db_manager.add_face_feature(existing_person['id'], temp_face['feature'])
                logging.info(f"已更新 {real_name} 的特征")
                
            else:
                # 新身份，更新临时记录为真实身份
                logging.info(f"发现新身份 {real_name} - {real_id_card}，更新为真实身份")
                
                # 更新数据库中的身份信息
                success = self.db_manager.update_person_real_info(person_id, real_name, real_id_card)
                if not success:
                    logging.error("更新数据库身份信息失败")
                    return False
                
                # 更新内存数据库
                temp_person_name = f"{temp_face['temp_name']}_{temp_face['temp_id']}"
                real_person_name = f"{real_name}_{real_id_card}"
                
                if temp_person_name in self.face_name_known_list:
                    temp_index = self.face_name_known_list.index(temp_person_name)
                    self.face_name_known_list[temp_index] = real_person_name
                    # 更新真实姓名列表
                    if temp_index < len(self.real_name_known_list):
                        self.real_name_known_list[temp_index] = real_name
                    else:
                        self.real_name_known_list.append(real_name)
                    logging.debug(f"已更新内存数据库中的身份: {temp_person_name} -> {real_person_name}")
                else:
                    # 如果临时身份不在内存数据库中，直接添加真实身份
                    self.face_name_known_list.append(real_person_name)
                    self.face_feature_known_list.append(temp_face['feature'])
                    self.real_name_known_list.append(real_name)  # 添加真实姓名
                    # 获取图像数据
                    image_data = self.db_manager.get_face_image(person_id)
                    if image_data:
                        self.face_image_data_list.append(image_data)
                    else:
                        # 如果没有图像数据，添加一个占位符
                        self.face_image_data_list.append(None)
                    logging.debug(f"已添加真实身份到内存数据库: {real_person_name}")
            
            # 从临时存储中移除
            del self.temp_faces[feature_str]
            
            logging.info(f"成功更新人脸信息: {real_name} - {real_id_card}")
            return True
            
        except Exception as e:
            logging.error(f"更新人脸信息失败: {str(e)}")
            return False

    def cleanup_temp_files(self, max_age_hours=24):
        """清理过期的临时文件"""
        try:
            # 使用数据库管理器清理过期的临时人员
            deleted_count = self.db_manager.delete_temp_persons(max_age_hours)
            
            if deleted_count > 0:
                logging.info(f"已清理 {deleted_count} 个过期的临时人员")
                
                # 重新加载数据库以更新内存中的数据
                self.face_name_known_list.clear()
                self.face_feature_known_list.clear()
                self.face_image_data_list.clear()
                self.real_name_known_list.clear()  # 清空真实姓名列表
                self.processed_features.clear()
                self.get_face_database()
                
        except Exception as e:
            logging.error(f"清理临时文件时出错: {str(e)}")

    def show_api_update_notification(self, temp_name, real_name, real_id):
        """显示API更新通知"""
        def show_notification():
            popup = Toplevel(self.root)
            popup.title("身份信息更新")
            popup.geometry("400x300")
            popup.attributes("-topmost", True)
            popup.grab_set()
            
            # 主框架
            main_frame = tk.Frame(popup)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # 标题
            title_label = Label(main_frame, text="身份信息已更新", font=('Arial', 14, 'bold'), fg='green')
            title_label.pack(pady=(0, 20))
            
            # 更新信息
            info_text = f"""临时身份: {temp_name}
真实身份: {real_name}
身份证号: {real_id}
更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
            
            info_label = Label(main_frame, text=info_text, font=('Arial', 11), justify='left')
            info_label.pack(pady=(0, 20))
            
            # 关闭按钮
            close_btn = tk.Button(main_frame, text="确定", command=popup.destroy, 
                                font=('Arial', 11), width=10, bg='green', fg='white')
            close_btn.pack(pady=(0, 10))
            
            # 5秒后自动关闭
            popup.after(5000, popup.destroy)
        
        # 在主线程中显示通知
        self.root.after(0, show_notification)

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
        """打开人脸库管理器 - 显示数据库内容并预览图片"""
        try:
            def show_face_library():
                # 创建主窗口
                main_window = Toplevel(self.root)
                main_window.title("人脸库管理器")
                main_window.geometry("800x600")
                main_window.attributes("-topmost", True)
                main_window.grab_set()
                
                # 主框架
                main_frame = tk.Frame(main_window)
                main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # 标题
                title_label = Label(main_frame, text="人脸库管理器", font=('Arial', 16, 'bold'), fg='blue')
                title_label.pack(pady=(0, 10))
                
                # 创建左右分栏
                paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
                paned_window.pack(fill=tk.BOTH, expand=True)
                
                # 左侧：人员列表
                left_frame = ttk.Frame(paned_window)
                paned_window.add(left_frame, weight=1)
                
                # 人员列表标题
                list_title = Label(left_frame, text="人员列表", font=('Arial', 12, 'bold'))
                list_title.pack(pady=(0, 5))
                
                # 创建Treeview显示人员列表
                columns = ('ID', '姓名', '身份证号', '类型', '创建时间')
                tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=15)
                
                # 设置列标题
                for col in columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=100)
                
                # 添加滚动条
                scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=tree.yview)
                tree.configure(yscrollcommand=scrollbar.set)
                
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # 右侧：详细信息
                right_frame = ttk.Frame(paned_window)
                paned_window.add(right_frame, weight=1)
                
                # 详细信息标题
                detail_title = Label(right_frame, text="详细信息", font=('Arial', 12, 'bold'))
                detail_title.pack(pady=(0, 5))
                
                # 详细信息显示区域
                detail_frame = ttk.Frame(right_frame)
                detail_frame.pack(fill=tk.BOTH, expand=True)
                
                # 图片显示区域
                image_frame = ttk.LabelFrame(detail_frame, text="人脸图片")
                image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
                
                image_label = Label(image_frame, text="请选择人员查看图片", font=('Arial', 10))
                image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # 信息显示区域
                info_frame = ttk.LabelFrame(detail_frame, text="人员信息")
                info_frame.pack(fill=tk.X, pady=(0, 10))
                
                info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, font=('Arial', 9))
                info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                # 按钮区域
                button_frame = ttk.Frame(detail_frame)
                button_frame.pack(fill=tk.X, pady=(0, 10))
                
                def load_person_data():
                    """加载人员数据到列表"""
                    try:
                        # 清空现有数据
                        for item in tree.get_children():
                            tree.delete(item)
                        
                        # 从数据库获取所有人员
                        conn = sqlite3.connect(self.db_manager.db_path)
                        cursor = conn.cursor()
                        
                        cursor.execute('''
                            SELECT id, name, id_card, is_temp, created_time, real_name, real_id_card
                            FROM persons
                            ORDER BY created_time DESC
                        ''')
                        
                        for row in cursor.fetchall():
                            person_id, name, id_card, is_temp, created_time, real_name, real_id_card = row
                            
                            # 确定显示名称
                            display_name = name
                            if real_name:
                                display_name = real_name
                            
                            # 确定身份证号
                            display_id = id_card
                            if real_id_card:
                                display_id = real_id_card
                            
                            # 确定类型
                            person_type = "临时" if is_temp else "真实"
                            
                            # 格式化时间
                            if created_time:
                                try:
                                    dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                                except:
                                    formatted_time = created_time
                            else:
                                formatted_time = "未知"
                            
                            tree.insert('', 'end', values=(person_id, display_name, display_id, person_type, formatted_time))
                        
                        conn.close()
                        
                        # 更新统计信息
                        stats = self.db_manager.get_statistics()
                        status_text = f"总人员: {stats['total_persons']} | 真实身份: {stats['real_persons']} | 临时身份: {stats['temp_persons']}"
                        status_label.config(text=status_text)
                        
                    except Exception as e:
                        logging.error(f"加载人员数据失败: {str(e)}")
                        import tkinter.messagebox as messagebox
                        messagebox.showerror("错误", f"加载人员数据失败:\n{str(e)}")
                
                def on_person_select(event):
                    """当选择人员时显示详细信息"""
                    selection = tree.selection()
                    if not selection:
                        return
                    
                    try:
                        # 获取选中的人员ID
                        person_id = tree.item(selection[0])['values'][0]
                        
                        # 获取人员详细信息
                        person_info = self.db_manager.get_person_by_id(person_id)
                        if not person_info:
                            return
                        
                        # 显示人员信息
                        info_text.delete(1.0, tk.END)
                        info_content = f"""人员ID: {person_info['id']}
姓名: {person_info['name']}
身份证号: {person_info['id_card'] or '无'}
真实姓名: {person_info['real_name'] or '无'}
真实身份证号: {person_info['real_id_card'] or '无'}
身份类型: {'临时身份' if person_info['is_temp'] else '真实身份'}
创建时间: {person_info['created_time']}
更新时间: {person_info['updated_time']}"""
                        
                        info_text.insert(1.0, info_content)
                        
                        # 获取并显示人脸图片
                        image_data = self.db_manager.get_face_image(person_id)
                        if image_data:
                            try:
                                # 将二进制数据转换为PIL图像
                                pil_img = Image.open(io.BytesIO(image_data))
                                
                                # 计算合适的显示尺寸
                                display_width = 200
                                display_height = 200
                                
                                # 保持宽高比
                                img_width, img_height = pil_img.size
                                ratio = min(display_width / img_width, display_height / img_height)
                                new_width = int(img_width * ratio)
                                new_height = int(img_height * ratio)
                                
                                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                tk_img = ImageTk.PhotoImage(pil_img)
                                
                                # 更新图片显示
                                image_label.config(image=tk_img, text="")
                                image_label.image = tk_img  # 保持引用
                                
                            except Exception as e:
                                image_label.config(image="", text=f"图片加载失败: {str(e)}")
                        else:
                            image_label.config(image="", text="无图片数据")
                        
                    except Exception as e:
                        logging.error(f"显示人员详细信息失败: {str(e)}")
                        info_text.delete(1.0, tk.END)
                        info_text.insert(1.0, f"获取详细信息失败: {str(e)}")
                
                def delete_selected_person():
                    """删除选中的人员"""
                    selection = tree.selection()
                    if not selection:
                        import tkinter.messagebox as messagebox
                        messagebox.showwarning("警告", "请先选择要删除的人员")
                        return
                    
                    person_id = tree.item(selection[0])['values'][0]
                    person_name = tree.item(selection[0])['values'][1]
                    
                    # 确认删除
                    import tkinter.messagebox as messagebox
                    result = messagebox.askyesno("确认删除", f"确定要删除人员 '{person_name}' 吗？\n此操作不可恢复！")
                    
                    if result:
                        try:
                            if self.db_manager.delete_person(person_id):
                                messagebox.showinfo("成功", f"已删除人员 '{person_name}'")
                                load_person_data()  # 重新加载数据
                            else:
                                messagebox.showerror("错误", "删除失败")
                        except Exception as e:
                            messagebox.showerror("错误", f"删除失败: {str(e)}")
                
                def refresh_data():
                    """刷新数据"""
                    load_person_data()
                
                # 绑定选择事件
                tree.bind('<<TreeviewSelect>>', on_person_select)
                
                # 按钮
                ttk.Button(button_frame, text="刷新", command=refresh_data).pack(side=tk.LEFT, padx=5)
                ttk.Button(button_frame, text="删除选中", command=delete_selected_person).pack(side=tk.LEFT, padx=5)
                ttk.Button(button_frame, text="关闭", command=main_window.destroy).pack(side=tk.RIGHT, padx=5)
                
                # 状态栏
                status_frame = ttk.Frame(main_frame)
                status_frame.pack(fill=tk.X, pady=(10, 0))
                
                status_label = Label(status_frame, text="正在加载数据...", font=('Arial', 9))
                status_label.pack(side=tk.LEFT)
                
                # 加载数据
                load_person_data()
            
            # 在主线程中显示窗口
            self.root.after(0, show_face_library)
            
        except Exception as e:
            logging.error(f"打开人脸库管理器失败: {str(e)}")
            import tkinter.messagebox as messagebox
            messagebox.showerror("错误", f"打开人脸库管理器失败:\n{str(e)}")

    def clear_all_temp_identities(self, icon=None, item=None):
        """清理所有临时身份（包括内存中的）"""
        try:
            logging.info("开始清理所有临时身份...")
            
            # 清理数据库中的临时人员
            before_count = self.db_manager.get_statistics()['temp_persons']
            self.cleanup_temp_files(max_age_hours=0)  # 清理所有临时文件
            after_count = self.db_manager.get_statistics()['temp_persons']
            db_cleaned_count = before_count - after_count
            
            # 清理内存中的临时身份
            temp_names_to_remove = []
            for i, name in enumerate(self.face_name_known_list):
                if name.startswith('unknown') or name.startswith('TEMP'):
                    temp_names_to_remove.append(i)
            
            # 从后往前删除，避免索引变化
            for i in reversed(temp_names_to_remove):
                self.face_name_known_list.pop(i)
                self.face_feature_known_list.pop(i)
                self.face_image_data_list.pop(i)
                self.real_name_known_list.pop(i)  # 同时清理真实姓名列表
            
            memory_cleaned_count = len(temp_names_to_remove)
            
            # 清理临时存储
            temp_faces_count = len(self.temp_faces)
            self.temp_faces.clear()
            
            # 清理已处理特征集合中的临时特征
            temp_features_to_remove = []
            for feature_str in self.processed_features:
                if 'unknown' in feature_str or 'TEMP' in feature_str:
                    temp_features_to_remove.append(feature_str)
            
            for feature_str in temp_features_to_remove:
                self.processed_features.discard(feature_str)
            
            # 重置临时用户计数器
            self.temp_user_counter = 1
            
            # 显示清理结果
            import tkinter.messagebox as messagebox
            total_cleaned = db_cleaned_count + memory_cleaned_count + temp_faces_count
            if total_cleaned > 0:
                result_msg = f"清理完成！\n\n数据库临时人员: {db_cleaned_count} 个\n内存临时身份: {memory_cleaned_count} 个\n临时存储: {temp_faces_count} 个\n总计: {total_cleaned} 个"
                messagebox.showinfo("清理完成", result_msg)
                logging.info(f"清理所有临时身份完成，总计清理了 {total_cleaned} 个临时项目")
            else:
                messagebox.showinfo("清理完成", "没有需要清理的临时身份")
                logging.info("清理所有临时身份完成，没有需要清理的项目")
                
        except Exception as e:
            logging.error(f"清理所有临时身份时出错: {str(e)}")
            import tkinter.messagebox as messagebox
            messagebox.showerror("错误", f"清理临时身份时出错:\n{str(e)}")

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
                # 删除之前的状态文本，防止重叠
                self.canvas.delete("status_text")
                
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
            self.face_image_data_list.clear()
            self.real_name_known_list.clear()  # 清空真实姓名列表
            self.processed_features.clear()
            
            # 重新加载数据库
            self.get_face_database()
            
            logging.info("人脸数据库重新加载成功")
                
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
        """从SQLite数据库加载人脸数据库"""
        try:
            # 从数据库获取所有特征
            features = self.db_manager.get_face_features()
            
            if features:
                for person_id, feature_vector, person_name, real_name in features:
                    self.face_name_known_list.append(person_name)
                    self.face_feature_known_list.append(feature_vector)
                    self.real_name_known_list.append(real_name)  # 存储真实姓名
                    
                    # 获取该人员的图像数据
                    image_data = self.db_manager.get_face_image(person_id)
                    self.face_image_data_list.append(image_data)
                    
                    # 生成特征字符串用于去重
                    feature_str = ','.join(map(str, feature_vector))
                    self.processed_features.add(feature_str)
                
                logging.info(f"已从数据库加载 {len(self.face_feature_known_list)} 张人脸")
            else:
                logging.info("数据库中没有找到人脸数据")
                
        except Exception as e:
            logging.error(f"从数据库加载人脸数据时出错: {str(e)}")
            # 如果数据库加载失败，尝试从CSV文件导入
            logging.info("尝试从CSV文件导入数据到数据库...")
            if self.db_manager.import_from_csv():
                # 重新加载
                self.get_face_database()

    def update_face_database_csv(self, person_name, feature):
        """更新人脸数据库CSV文件"""
        try:
            csv_path = "data/features_all.csv"
            
            # 检查是否已存在该身份
            existing_data = []
            person_exists = False
            updated = False
            
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            if row and row[0] == person_name:
                                # 身份已存在，更新特征
                                person_exists = True
                                new_row = [person_name] + list(feature)
                                existing_data.append(new_row)
                                logging.info(f"更新身份 {person_name} 的特征")
                                updated = True
                            else:
                                existing_data.append(row)
                except Exception as e:
                    logging.warning(f"读取CSV文件时出错: {str(e)}")
                    return False
            
            if not person_exists:
                # 新身份，追加到文件末尾
                if not os.path.exists(csv_path):
                    # 创建目录
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    row_data = [person_name] + list(feature)
                    writer.writerow(row_data)
                logging.info(f"已添加新身份到CSV文件: {person_name}")
                return True
            else:
                # 身份已存在，重写整个文件
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in existing_data:
                        writer.writerow(row)
                if updated:
                    logging.info(f"已更新CSV文件中 {person_name} 的特征")
                return True
            
        except Exception as e:
            logging.error(f"更新CSV文件失败: {str(e)}")
            return False
 
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
        # 过滤掉临时身份，不显示unknown1、unknown2等临时身份的弹窗
        if name.startswith('unknown') or name.startswith('TEMP'):
            logging.debug(f"跳过临时身份弹窗显示: {name}")
            return
            
        if not self.show_popup or name in self.shown_faces: 
            return 
            
        self.shown_faces.add(name) 
 
        def show():
            popup = Toplevel(self.root) 
            popup.title(f"识别到: {name}")
            popup.attributes("-topmost", True)
            
            # 从数据库获取图像数据
            image_data = self.face_image_data_list[idx] if idx < len(self.face_image_data_list) else None
            
            if image_data: 
                try:
                    # 将二进制数据转换为PIL图像
                    pil_img = Image.open(io.BytesIO(image_data)).resize((200, 200))
                    tk_img = ImageTk.PhotoImage(pil_img)
                    label_img = Label(popup, image=tk_img)
                    label_img.image = tk_img 
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
                
                # 如果有真实姓名，优先显示真实姓名
                if idx < len(self.real_name_known_list) and self.real_name_known_list[idx]:
                    display_name = self.real_name_known_list[idx]
                
                info = f"姓名: {display_name}\n身份证号: {id_number}\n识别时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            else:
                # 旧格式或其他格式
                display_name = name
                
                # 如果有真实姓名，优先显示真实姓名
                if idx < len(self.real_name_known_list) and self.real_name_known_list[idx]:
                    display_name = self.real_name_known_list[idx]
                
                info = f"姓名: {display_name}\n识别时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                
            Label(popup, text=info, font=('Arial', 12)).pack(pady=10)
            
            def close():
                self.shown_faces.discard(name) 
                popup.destroy() 
                
            tk.Button(popup, text="关闭", command=close).pack(pady=10)
            popup.after(5000, close)
 
        threading.Thread(target=show).start()
 
    def create_new_face_data(self, img, face_rect, shape, feature):
        """处理新检测到的人脸 - 保存到数据库"""
        # 检查自动发现新面孔功能是否开启
        if not self.auto_add_new_faces:
            logging.debug("自动发现新面孔功能已关闭，跳过")
            return
            
        current_time = time.time() 
        
        # 检查冷却时间 
        if current_time - self.last_new_face_time < self.new_face_cooldown: 
            logging.debug("新面孔检测过于频繁，已忽略")
            return 
            
        # 检查是否已经处理过这个特征 
        feature_str = ','.join(map(str, feature))
        if feature_str in self.processed_features: 
            logging.debug("已处理过此特征的人脸，跳过")
            return 
            
        # 检查是否正在处理新面孔
        if self.is_processing_new_face or self.new_face_popup_window is not None: 
            logging.debug("已有正在处理的新面孔或弹窗还在显示，跳过")
            return 
            
        # 记录新面孔检测
        logging.info(f"发现新面孔，准备处理 (位置: {face_rect.left()},{face_rect.top()}-{face_rect.right()},{face_rect.bottom()})")
        
        # 截取人脸图像
        top = max(0, face_rect.top()) 
        bottom = min(img.shape[0], face_rect.bottom()) 
        left = max(0, face_rect.left()) 
        right = min(img.shape[1], face_rect.right()) 
        face_img = img[top:bottom, left:right]
        
        # 生成临时身份信息
        temp_name, temp_id = self.generate_temp_identity()
        
        try:
            # 将图像转换为JPEG格式的二进制数据
            face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            success, encoded_img = cv2.imencode('.jpg', face_img_bgr)
            if not success:
                logging.error("图像编码失败")
                return
            image_data = encoded_img.tobytes()
            
            # 添加到数据库
            person_id = self.db_manager.add_person(temp_name, temp_id, is_temp=True)
            self.db_manager.add_face_image(person_id, image_data, 'jpg')
            self.db_manager.add_face_feature(person_id, feature)
            
            # 添加到临时存储
            self.temp_faces[feature_str] = {
                'temp_name': temp_name,
                'temp_id': temp_id,
                'person_id': person_id,
                'face_img': face_img,
                'feature': feature,
                'detect_time': current_time
            }
            
            # 不添加到内存数据库，避免被识别为已知人脸
            # temp_person_name = f"{temp_name}_{temp_id}"
            # self.face_name_known_list.append(temp_person_name)
            # self.face_feature_known_list.append(feature)
            # self.face_image_data_list.append(image_data)
            
            # 标记为已处理
            self.processed_features.add(feature_str)
            self.last_new_face_time = current_time
            
            logging.info(f"已创建临时身份: {temp_name} - {temp_id} (数据库ID: {person_id})")
            
        except Exception as e:
            logging.error(f"保存新面孔到数据库失败: {str(e)}")
            return
        
        # 在新线程中调用API
        def api_call_thread():
            try:
                logging.info(f"开始为 {temp_name} 调用API获取真实身份...")
                api_result = self.call_face_recognition_api(face_img)
                
                if api_result:
                    # API调用成功，更新身份信息
                    success = self.update_face_with_api_result(feature_str, api_result)
                    if success:
                        logging.info(f"成功更新 {temp_name} 为真实身份: {api_result['name']} - {api_result['id_card']}")
                        
                        # 显示更新通知
                        if self.show_popup:
                            self.show_api_update_notification(temp_name, api_result['name'], api_result['id_card'])
                    else:
                        logging.error(f"更新 {temp_name} 身份信息失败")
                else:
                    logging.warning(f"API调用失败，保持临时身份: {temp_name}")
                    
            except Exception as e:
                logging.error(f"API调用线程出错: {str(e)}")
        
        # 启动API调用线程
        threading.Thread(target=api_call_thread, daemon=True).start()
 
    def process_frame(self):
        """处理每一帧图像"""
        # 检查日志轮转
        log_manager.check_and_rotate()
        
        # 定期清理临时文件
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.cleanup_temp_files(max_age_hours=24)
            self.last_cleanup_time = current_time
        
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
            
            # 使用数据库查找相似人脸
            match_result = self.db_manager.find_similar_face(feature, self.recognition_threshold)
            
            if match_result:
                # 找到匹配的人脸
                person_id, distance, person_name, real_name = match_result
                
                # 优先使用real_name，如果没有则使用person_name
                display_name = real_name if real_name else person_name
                name = display_name
                known = True
                
                # 记录识别结果
                # logging.info(f"识别到已知人脸: {name} (距离: {distance:.3f})")
                
                # 在内存列表中找到对应的索引（使用person_name查找，因为内存中存储的是person_name）
                if person_name in self.face_name_known_list:
                    idx = self.face_name_known_list.index(person_name)
                    if name not in self.shown_faces and self.show_popup: 
                        self.root.after(100, lambda n=name, i=idx: self.show_face_info(n, i))
            else:
                # 未找到匹配的人脸，标记为未知
                name = "Unknown"
                known = False
                logging.debug(f"检测到未知人脸")
                # 未知人脸，尝试添加到处理 
                self.create_new_face_data(img, rect, shape, feature)
                
            # 更新当前帧数据 
            self.current_frame_face_feature_list.append(feature) 
            self.current_frame_face_position_list.append((rect.left(), rect.top(), rect.right(), rect.bottom())) 
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
            
            # 处理显示名称
            if self.current_frame_face_known_list[i]:
                # 已知人脸 - 已经优先使用了real_name
                if display_name.startswith('unknown') or display_name.startswith('TEMP'):
                    # 临时身份，显示为未知
                    display_name = "未知"
                elif '_' in display_name and display_name.count('_') >= 1:
                    # 新格式: 姓名_身份证号，只显示姓名
                    name_parts = display_name.split('_', 1)
                    display_name = name_parts[0]
                # 其他情况直接显示原始名称
            else:
                # 未知人脸
                if display_name == "Unknown":
                    display_name = "未知"
                elif display_name.startswith('unknown') or display_name.startswith('TEMP'):
                    display_name = "未知"
            
            # 设置文本颜色
            text_color = 'lime' if self.current_frame_face_known_list[i] else 'yellow'
            
            self.canvas.create_text( 
                left, bottom + 20, 
                text=display_name, 
                fill=text_color, 
                font=('Arial', 12, 'bold'), 
                anchor='nw'
            )
        
        # 只在开启状态显示时绘制状态信息
        if self.show_status_display:
            # 创建半透明背景
            bg_width = 310
            bg_height = 210  # 增加高度以容纳API状态
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
            
            # 显示API状态
            api_status = "API调用: 开启" if self.api_enabled else "API调用: 关闭"
            if self.temp_faces:
                api_status += f" | 临时面孔: {len(self.temp_faces)}"
            api_color = 'green' if self.api_enabled else 'red'
            self.canvas.create_text( 
                20, 125, 
                text=api_status, 
                fill=api_color, 
                font=('Arial', 11, 'bold'), 
                anchor='nw'
            )
            
            # 显示当前识别阈值
            threshold_status = f"识别阈值: {self.recognition_threshold:.2f}"
            self.canvas.create_text( 
                20, 150, 
                text=threshold_status, 
                fill='magenta', 
                font=('Arial', 11, 'bold'), 
                anchor='nw'
            )
            
            # 显示运行模式
            mode_status = f"运行模式: {'GPU加速' if gpu_available else 'CPU优化'} (间隔:{self.process_interval}ms)"
            self.canvas.create_text( 
                20, 175, 
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
        logging.info("正在从图像文件夹重新生成人脸特征文件...")
        try:
            # 清空现有数据
            self.face_name_known_list.clear()
            self.face_feature_known_list.clear()
            self.face_image_data_list.clear()
            self.processed_features.clear()
            
            # 获取所有人员文件夹
            person_folders = []
            data_faces_path = "data/data_faces_from_camera/"
            if os.path.exists(data_faces_path):
                # 只获取真实身份的文件夹，过滤掉unknown临时文件夹
                person_folders = [f for f in os.listdir(data_faces_path) 
                                if f.startswith("person_") and not f.startswith("person_unknown")]
            
            if not person_folders:
                print("没有找到任何真实身份的人脸图像文件夹")
                logging.warning("没有找到任何真实身份的人脸图像文件夹")
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
                        msg = f"警告: 跳过异常格式的文件夹 {person_folder}"
                        print(msg)
                        logging.warning(msg)
                        continue
                    
                    folder_path = os.path.join(data_faces_path, person_folder)
                    
                    # 获取该人员的所有图像
                    image_files = []
                    try:
                        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    except Exception as e:
                        msg = f"警告: 无法读取文件夹 {person_folder}: {str(e)}"
                        print(msg)
                        logging.warning(msg)
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
                                    msg = f"警告: 无法读取图像 {img_path}"
                                    print(msg)
                                    logging.warning(msg)
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
                                    msg = f"警告: 在图像 {img_path} 中没有检测到人脸"
                                    print(msg)
                                    logging.warning(msg)
                                    continue
                                
                                # 提取特征
                                shape = predictor(img, faces[0].rect)
                                feature = face_reco_model.compute_face_descriptor(img, shape)
                                features_list.append(feature)
                                processed_images += 1
                                
                            except Exception as e:
                                msg = f"警告: 处理图像 {img_path} 时出错: {str(e)}"
                                print(msg)
                                logging.warning(msg)
                                continue
                    
                    # 无论是否成功提取到特征，都生成CSV数据
                    if features_list:
                        # 计算平均特征
                        avg_feature = np.mean(features_list, axis=0)
                        msg = f"已处理 {display_name}: {processed_images} 张图像成功提取特征"
                        print(msg)
                        logging.info(msg)
                    else:
                        # 生成128维的零向量作为默认特征
                        avg_feature = np.zeros(128)
                        msg = f"警告: {display_name} 没有成功提取到特征，使用默认零向量"
                        print(msg)
                        logging.warning(msg)
                    
                    # 写入CSV - 无论是否有特征都写入
                    writer.writerow([person_name] + list(avg_feature))
                    
                    # 更新内存数据
                    self.face_name_known_list.append(person_name)
                    self.face_feature_known_list.append(list(avg_feature))
                    
                    # 设置图像路径 - 如果有图像文件则使用第一个，否则使用默认路径
                    if image_files:
                        self.face_image_data_list.append(os.path.join(folder_path, image_files[0]))
                    else:
                        # 创建一个默认的图像路径，即使文件不存在
                        default_img_path = os.path.join(folder_path, "img_face_1.jpg")
                        self.face_image_data_list.append(default_img_path)
                    
                    feature_str = ','.join(map(str, avg_feature))
                    self.processed_features.add(feature_str)
            
            msg = f"CSV文件重新生成完成，共处理 {len(self.face_name_known_list)} 个人"
            print(msg)
            logging.info(msg)
            return True
            
        except Exception as e:
            msg = f"重新生成CSV文件时出错: {str(e)}"
            print(msg)
            logging.error(msg)
            return False

    def debug_database(self, icon=None, item=None):
        """调试数据库 - 查看当前数据库状态"""
        try:
            def show_debug_info():
                # 创建调试窗口
                debug_window = Toplevel(self.root)
                debug_window.title("数据库调试信息")
                debug_window.geometry("600x500")
                debug_window.attributes("-topmost", True)
                debug_window.grab_set()
                
                # 主框架
                main_frame = tk.Frame(debug_window)
                main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # 标题
                title_label = Label(main_frame, text="数据库调试信息", font=('Arial', 14, 'bold'), fg='blue')
                title_label.pack(pady=(0, 10))
                
                # 创建文本框
                text_frame = tk.Frame(main_frame)
                text_frame.pack(fill=tk.BOTH, expand=True)
                
                text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
                scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)
                
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # 获取数据库信息
                try:
                    # 数据库统计信息
                    stats = self.db_manager.get_statistics()
                    text_widget.insert(tk.END, "=== 数据库统计信息 ===\n")
                    text_widget.insert(tk.END, f"总人员: {stats['total_persons']} 个\n")
                    text_widget.insert(tk.END, f"真实身份: {stats['real_persons']} 个\n")
                    text_widget.insert(tk.END, f"临时身份: {stats['temp_persons']} 个\n")
                    text_widget.insert(tk.END, f"人脸特征: {stats['total_features']} 个\n")
                    text_widget.insert(tk.END, f"人脸图像: {stats['total_images']} 个\n\n")
                    
                    # 内存数据库信息
                    text_widget.insert(tk.END, "=== 内存数据库信息 ===\n")
                    text_widget.insert(tk.END, f"加载的人员数量: {len(self.face_name_known_list)}\n")
                    text_widget.insert(tk.END, f"加载的特征数量: {len(self.face_feature_known_list)}\n")
                    text_widget.insert(tk.END, f"加载的图像数量: {len(self.face_image_data_list)}\n\n")
                    
                    # 真实身份列表
                    text_widget.insert(tk.END, "=== 真实身份列表 ===\n")
                    if self.face_name_known_list:
                        for i, name in enumerate(self.face_name_known_list):
                            text_widget.insert(tk.END, f"{i+1}. {name}\n")
                    else:
                        text_widget.insert(tk.END, "没有加载任何真实身份\n")
                    
                    text_widget.insert(tk.END, "\n=== 临时身份信息 ===\n")
                    text_widget.insert(tk.END, f"临时面孔数量: {len(self.temp_faces)}\n")
                    if self.temp_faces:
                        for feature_str, temp_info in self.temp_faces.items():
                            text_widget.insert(tk.END, f"临时身份: {temp_info['temp_name']}_{temp_info['temp_id']}\n")
                    
                    text_widget.insert(tk.END, "\n=== 识别设置 ===\n")
                    text_widget.insert(tk.END, f"识别阈值: {self.recognition_threshold}\n")
                    text_widget.insert(tk.END, f"自动发现新面孔: {'开启' if self.auto_add_new_faces else '关闭'}\n")
                    text_widget.insert(tk.END, f"API调用: {'开启' if self.api_enabled else '关闭'}\n")
                    
                    # 当前帧信息
                    text_widget.insert(tk.END, "\n=== 当前帧信息 ===\n")
                    text_widget.insert(tk.END, f"检测到人脸: {self.current_frame_face_cnt} 个\n")
                    if self.current_frame_face_name_list:
                        for i, name in enumerate(self.current_frame_face_name_list):
                            known = self.current_frame_face_known_list[i]
                            status = "已知" if known else "未知"
                            text_widget.insert(tk.END, f"人脸 {i+1}: {name} ({status})\n")
                    
                except Exception as e:
                    text_widget.insert(tk.END, f"获取调试信息时出错: {str(e)}\n")
                
                # 按钮
                button_frame = tk.Frame(main_frame)
                button_frame.pack(fill=tk.X, pady=(10, 0))
                
                def refresh_info():
                    text_widget.delete(1.0, tk.END)
                    # 重新获取信息（这里可以调用相同的逻辑）
                    debug_window.destroy()
                    self.debug_database()
                
                def reload_database():
                    try:
                        self.reload_face_database()
                        text_widget.insert(tk.END, "\n=== 重新加载完成 ===\n")
                        text_widget.insert(tk.END, f"重新加载后的人员数量: {len(self.face_name_known_list)}\n")
                    except Exception as e:
                        text_widget.insert(tk.END, f"\n重新加载失败: {str(e)}\n")
                
                tk.Button(button_frame, text="刷新", command=refresh_info).pack(side=tk.LEFT, padx=5)
                tk.Button(button_frame, text="重新加载数据库", command=reload_database).pack(side=tk.LEFT, padx=5)
                tk.Button(button_frame, text="关闭", command=debug_window.destroy).pack(side=tk.RIGHT, padx=5)
            
            # 在主线程中显示调试窗口
            self.root.after(0, show_debug_info)
            
        except Exception as e:
            logging.error(f"显示调试信息时出错: {str(e)}")
            import tkinter.messagebox as messagebox
            messagebox.showerror("错误", f"显示调试信息时出错:\n{str(e)}")

    def clear_database(self, icon=None, item=None):
        """清空数据库"""
        def show_clear_dialog():
            # 创建密码确认对话框
            dialog = Toplevel(self.root)
            dialog.title("清空数据库")
            dialog.geometry("450x350")
            dialog.attributes("-topmost", True)
            dialog.grab_set()
            
            # 主框架
            main_frame = tk.Frame(dialog)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # 警告标题
            warning_label = Label(main_frame, text="⚠️ 危险操作警告", font=('Arial', 14, 'bold'), fg='red')
            warning_label.pack(pady=(0, 10))
            
            # 警告信息
            warning_text = """此操作将清空所有数据：
• 所有人员信息
• 所有人脸特征
• 所有人脸图像
• 所有识别记录

此操作不可恢复！"""
            
            warning_info = Label(main_frame, text=warning_text, font=('Arial', 11), fg='red', justify='left')
            warning_info.pack(pady=(0, 20))
            
            # 确认文字输入框
            confirm_frame = tk.Frame(main_frame)
            confirm_frame.pack(pady=(0, 20))
            
            confirm_label = Label(confirm_frame, text="请输入确认文字:", font=('Arial', 11))
            confirm_label.pack(side=tk.LEFT, padx=(0, 10))
            
            confirm_var = tk.StringVar()
            confirm_entry = tk.Entry(confirm_frame, textvariable=confirm_var, font=('Arial', 11), width=25)
            confirm_entry.pack(side=tk.LEFT)
            
            # 提示文字
            hint_label = Label(main_frame, text="请输入：确认清空数据库", font=('Arial', 10), fg='blue')
            hint_label.pack(pady=(0, 20))
            
            # 按钮框架
            button_frame = tk.Frame(main_frame)
            button_frame.pack(pady=(0, 10))
            
            def confirm_clear():
                """确认清空"""
                confirm_text = confirm_var.get().strip()
                
                # 验证确认文字
                if confirm_text != "确认清空数据库":
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("错误", "确认文字不正确！\n请输入：确认清空数据库")
                    confirm_var.set("")
                    confirm_entry.focus()
                    return
                
                # 再次确认
                confirm_result = tk.messagebox.askyesno("最终确认", 
                    "确认文字正确！\n\n这是最后一次确认，确定要清空所有数据吗？\n\n此操作不可恢复！")
                
                if confirm_result:
                    try:
                        # 执行清空操作
                        logging.info("用户确认清空数据库")
                        
                        # 清空数据库
                        success = self.db_manager.clear_database()
                        
                        if success:
                            # 清空内存数据
                            self.face_name_known_list.clear()
                            self.face_feature_known_list.clear()
                            self.face_image_data_list.clear()
                            self.real_name_known_list.clear()
                            self.processed_features.clear()
                            self.temp_faces.clear()
                            self.shown_faces.clear()
                            
                            # 重置临时用户计数器
                            self.temp_user_counter = 1
                            
                            logging.info("数据库清空成功")
                            tk.messagebox.showinfo("成功", "数据库已成功清空！\n\n所有数据已被删除。")
                        else:
                            tk.messagebox.showerror("错误", "清空数据库失败！")
                        
                        dialog.destroy()
                        
                    except Exception as e:
                        logging.error(f"清空数据库时出错: {str(e)}")
                        tk.messagebox.showerror("错误", f"清空数据库时出错:\n{str(e)}")
                        dialog.destroy()
            
            def cancel_clear():
                """取消清空"""
                dialog.destroy()
            
            # 确认按钮
            confirm_btn = tk.Button(button_frame, text="确认清空", command=confirm_clear, 
                                  font=('Arial', 11), width=12, bg='red', fg='white')
            confirm_btn.pack(side=tk.LEFT, padx=5)
            
            # 取消按钮
            cancel_btn = tk.Button(button_frame, text="取消", command=cancel_clear, 
                                 font=('Arial', 11), width=12, bg='gray', fg='white')
            cancel_btn.pack(side=tk.LEFT, padx=5)
            
            # 绑定回车键
            confirm_entry.bind('<Return>', lambda e: confirm_clear())
            
            # 设置焦点
            confirm_entry.focus()
        
        # 在主线程中显示对话框
        self.root.after(0, show_clear_dialog)

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
    recognizer = None
    try:
        logging.info("=" * 50)
        logging.info("人脸识别监控系统启动")
        logging.info("=" * 50)
        
        # 创建主实例
        recognizer = TransparentFaceRecognizer()
        
        # 显示加载进度条
        recognizer.show_loading_progress("正在加载人脸库...", lambda: recognizer.get_face_database())
        
        print("启动人脸识别监控系统...")
        logging.info("开始运行人脸识别监控系统")
        recognizer.run() 
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}")
        logging.error("程序异常退出")
        raise
    finally:
        # 关闭数据库连接
        if recognizer and hasattr(recognizer, 'db_manager'):
            recognizer.db_manager.close()
        
        logging.info("=" * 50)
        logging.info("人脸识别监控系统退出")
        logging.info("=" * 50)
 
if __name__ == '__main__':
    main()