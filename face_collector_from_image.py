import cv2
import dlib
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import shutil
import sys
import subprocess
import threading
from face_database_manager import FaceDatabaseManager
import random

# 设置系统编码以支持中文路径
if sys.platform.startswith('win'):
    # Windows系统
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'Chinese_China.UTF8')
        except:
            pass

class FaceCollector:
    def __init__(self):
        # 初始化人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
        self.face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # 初始化数据库管理器
        self.db_manager = FaceDatabaseManager("data/face_database.db")
        
        # 设置保存路径（保留用于临时文件）
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        
        # 创建保存目录（用于临时文件）
        if not os.path.exists(self.path_photos_from_camera):
            os.makedirs(self.path_photos_from_camera)
        
        # 初始化GUI
        self.win = tk.Tk()
        self.win.title("智能人脸采集工具 - SQLite版本")
        self.win.geometry("1200x800")
        self.win.configure(bg='#f0f0f0')
        
        # 设置窗口图标（如果有的话）
        try:
            self.win.iconbitmap('icon.ico')
        except:
            pass
        
        # 创建界面元素
        self.create_widgets()
        
        # 初始化变量
        self.current_image = None
        self.current_faces = []
        self.registered_names = []
        self.current_image_path = ""  # 存储当前选择的图片路径
        self.load_registered_names()
        
    def create_widgets(self):
        # 主容器
        main_container = tk.Frame(self.win, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 左侧图片显示区域
        self.frame_left = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=2)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 图片显示标题
        image_title = tk.Label(self.frame_left, text="图片预览", 
                              font=('Microsoft YaHei UI', 14, 'bold'), 
                              fg='#2c3e50', bg='white')
        image_title.pack(pady=10)
        
        # 图片显示区域
        self.label_image = tk.Label(self.frame_left, bg='white', relief=tk.SUNKEN, bd=1)
        self.label_image.pack(padx=20, pady=(0, 20), fill=tk.BOTH, expand=True)
        
        # 右侧控制区域
        self.frame_right = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=2)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 控制区域标题
        control_title = tk.Label(self.frame_right, text="操作控制", 
                                font=('Microsoft YaHei UI', 14, 'bold'), 
                                fg='#2c3e50', bg='white')
        control_title.pack(pady=10)
        
        # 选择图片按钮
        self.btn_select = tk.Button(self.frame_right, text="📁 选择图片", 
                                   command=self.select_image,
                                   font=('Microsoft YaHei UI', 12),
                                   bg='#3498db', fg='white',
                                   relief=tk.FLAT, padx=20, pady=8,
                                   cursor='hand2')
        self.btn_select.pack(pady=8, padx=20, fill=tk.X)
        
        # 人脸选择区域
        self.face_selection_frame = tk.Frame(self.frame_right, bg='white')
        self.face_selection_frame.pack(pady=8, padx=20, fill=tk.X)
        
        tk.Label(self.face_selection_frame, text="选择人脸:", 
                font=('Microsoft YaHei UI', 11, 'bold'), 
                fg='#2c3e50', bg='white').pack(anchor=tk.W, pady=(0, 5))
        
        # 人脸选择变量
        self.selected_faces = []
        self.face_vars = []
        
        # 姓名输入区域
        name_frame = tk.Frame(self.frame_right, bg='white')
        name_frame.pack(pady=8, padx=20, fill=tk.X)
        
        tk.Label(name_frame, text="输入姓名:", 
                font=('Microsoft YaHei UI', 11, 'bold'), 
                fg='#2c3e50', bg='white').pack(anchor=tk.W, pady=(0, 5))
        
        # 创建输入框框架
        self.entry_frame = tk.Frame(name_frame, bg='white')
        self.entry_frame.pack(fill=tk.X)
        
        self.entry_name = tk.Entry(self.entry_frame, font=('Microsoft YaHei UI', 11),
                                  relief=tk.SOLID, bd=1)
        self.entry_name.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(0, 8))
        
        # 身份证号输入区域
        id_frame = tk.Frame(self.frame_right, bg='white')
        id_frame.pack(pady=8, padx=20, fill=tk.X)
        
        tk.Label(id_frame, text="身份证号:", 
                font=('Microsoft YaHei UI', 11, 'bold'), 
                fg='#2c3e50', bg='white').pack(anchor=tk.W, pady=(0, 5))
        
        self.entry_id = tk.Entry(id_frame, font=('Microsoft YaHei UI', 11),
                                relief=tk.SOLID, bd=1)
        self.entry_id.pack(fill=tk.X, pady=(0, 8))
        
        # 保存按钮
        self.btn_save = tk.Button(self.frame_right, text="💾 保存所有人脸", 
                                 command=self.save_selected_face,
                                 font=('Microsoft YaHei UI', 12),
                                 bg='#27ae60', fg='white',
                                 relief=tk.FLAT, padx=20, pady=8,
                                 cursor='hand2')
        self.btn_save.pack(pady=8, padx=20, fill=tk.X)
        
        # 批量保存按钮
        self.btn_batch_save = tk.Button(self.frame_right, text="📝 批量保存不同人员", 
                                       command=self.save_multiple_faces_with_names,
                                       font=('Microsoft YaHei UI', 11),
                                       bg='#9b59b6', fg='white',
                                       relief=tk.FLAT, padx=15, pady=6,
                                       cursor='hand2')
        self.btn_batch_save.pack(pady=5, padx=20, fill=tk.X)
        
        # 分隔线
        separator = ttk.Separator(self.frame_right, orient='horizontal')
        separator.pack(fill=tk.X, padx=20, pady=12)
        
        # 已注册人名区域
        names_title = tk.Label(self.frame_right, text="已注册的人员信息", 
                              font=('Microsoft YaHei UI', 12, 'bold'), 
                              fg='#2c3e50', bg='white')
        names_title.pack(pady=(0, 8))
        
        # 人名列表框架
        list_frame = tk.Frame(self.frame_right, bg='white')
        list_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox_names = tk.Listbox(list_frame, 
                                       font=('Microsoft YaHei UI', 10),
                                       yscrollcommand=scrollbar.set,
                                       selectmode=tk.SINGLE,
                                       relief=tk.SOLID, bd=1,
                                       bg='#f8f9fa', fg='#2c3e50',
                                       selectbackground='#3498db',
                                       selectforeground='white',
                                       height=8)  # 限制列表高度
        self.listbox_names.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox_names.yview)
        
        # 绑定点击事件，点击时自动填充信息
        self.listbox_names.bind('<Double-Button-1>', self.on_name_list_click)
        self.listbox_names.bind('<Return>', self.on_name_list_click)
        
        # 删除按钮
        self.btn_delete = tk.Button(self.frame_right, text="🗑️ 删除选中的人员信息", 
                                   command=self.delete_selected_name,
                                   font=('Microsoft YaHei UI', 11),
                                   bg='#e74c3c', fg='white',
                                   relief=tk.FLAT, padx=15, pady=6,
                                   cursor='hand2')
        self.btn_delete.pack(pady=8, padx=20, fill=tk.X)
        
        # 状态栏
        self.status_label = tk.Label(self.win, text="就绪", 
                                    font=('Microsoft YaHei UI', 9),
                                    fg='#7f8c8d', bg='#ecf0f1',
                                    relief=tk.SUNKEN, bd=1, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定按钮悬停效果
        self.bind_hover_effects()
    
    def bind_hover_effects(self):
        """绑定按钮悬停效果"""
        buttons = [self.btn_select, self.btn_save, self.btn_batch_save, self.btn_delete]
        for button in buttons:
            button.bind('<Enter>', lambda e, btn=button: self.on_button_hover(btn, '#2980b9'))
            button.bind('<Leave>', lambda e, btn=button: self.on_button_hover(btn, button.cget('bg')))
    
    def on_button_hover(self, button, color):
        """按钮悬停效果"""
        if color == '#2980b9':
            # 悬停时变暗
            if button == self.btn_save:
                button.config(bg='#229954')
            elif button == self.btn_batch_save:
                button.config(bg='#8e44ad')
            elif button == self.btn_delete:
                button.config(bg='#c0392b')
            else:
                button.config(bg=color)
        else:
            # 恢复原色
            if button == self.btn_save:
                button.config(bg='#27ae60')
            elif button == self.btn_batch_save:
                button.config(bg='#9b59b6')
            elif button == self.btn_delete:
                button.config(bg='#e74c3c')
            else:
                button.config(bg='#3498db')
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        self.win.update_idletasks()
    
    def generate_temp_identity(self):
        """生成临时身份信息，参考screen_face_monitor.py中的实现"""
        # 生成类似 unknown1, unknown2 的临时姓名
        temp_name = f"unknown{random.randint(1000, 9999)}"
        
        # 生成临时身份证号
        temp_id = "TEMP" + str(random.randint(100000, 999999))
        
        return temp_name, temp_id

    def load_registered_names(self):
        """从数据库加载已注册的人名，优先显示real_name和real_id_card信息"""
        try:
            # 从数据库获取所有非临时人员信息
            persons = self.db_manager.get_all_persons(include_temp=False)
            
            self.registered_names = []
            for person in persons:
                # 优先使用real_name和real_id_card，如果没有则使用name和id_card
                display_name = person.get('real_name') or person['name']
                display_id = person.get('real_id_card') or person.get('id_card')
                
                if display_id:
                    self.registered_names.append(f"{display_name}_{display_id}")
                else:
                    self.registered_names.append(display_name)
            
            self.update_name_list()
            print(f"从数据库加载了 {len(self.registered_names)} 个已注册人员")
        except Exception as e:
            print(f"加载已注册人名时出错: {str(e)}")
            self.registered_names = []
            self.update_name_list()
    
    def update_name_list(self):
        """更新人名列表显示"""
        self.listbox_names.delete(0, tk.END)
        for name in self.registered_names:
            self.listbox_names.insert(tk.END, name)
    
    def decode_path(self, file_path):
        """处理中文路径编码问题"""
        try:
            # 尝试直接使用路径
            if os.path.exists(file_path):
                return file_path
            
            # 如果直接路径不存在，尝试不同的编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'cp936']
            for encoding in encodings:
                try:
                    decoded_path = file_path.encode(encoding).decode(encoding)
                    if os.path.exists(decoded_path):
                        return decoded_path
                except:
                    continue
            
            # 如果还是不行，尝试使用原始路径
            return file_path
        except:
            return file_path
    
    def select_image(self):
        """选择图片文件"""
        self.update_status("正在选择图片...")
        
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("所有支持的图片格式", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.webp *.ico"),
                ("JPEG 图片", "*.jpg *.jpeg"),
                ("PNG 图片", "*.png"),
                ("BMP 图片", "*.bmp"),
                ("GIF 图片", "*.gif"),
                ("TIFF 图片", "*.tiff *.tif"),
                ("WebP 图片", "*.webp"),
                ("ICO 图标", "*.ico"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.update_status(f"正在处理图片: {os.path.basename(file_path)}")
            print(f"选择的图片路径: {file_path}")
            
            # 处理中文路径编码问题
            processed_path = self.decode_path(file_path)
            if processed_path != file_path:
                print(f"路径已处理: {processed_path}")
            
            # 尝试使用OpenCV读取图片
            image = cv2.imread(processed_path)
            
            # 如果OpenCV读取失败，尝试使用PIL读取
            if image is None:
                print(f"OpenCV无法读取图片，尝试使用PIL读取: {processed_path}")
                try:
                    # 使用PIL读取图片
                    pil_image = Image.open(processed_path)
                    # 转换为RGB模式
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    # 转换为numpy数组
                    image = np.array(pil_image)
                    # 转换为BGR格式（OpenCV格式）
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    print(f"PIL成功读取图片: {processed_path}")
                except Exception as pil_error:
                    print(f"PIL也无法读取图片: {str(pil_error)}")
                    self.update_status("图片读取失败")
                    messagebox.showerror("错误", f"无法读取图片文件: {file_path}\n请确保文件存在且格式正确\n错误信息: {str(pil_error)}")
                    return
            
            if image is None:
                print(f"无法读取图片，请检查文件是否存在且格式正确: {processed_path}")
                self.update_status("图片读取失败")
                messagebox.showerror("错误", f"无法读取图片文件: {processed_path}\n请确保文件存在且格式正确")
                return

            print(f"原始图片尺寸: {image.shape}")

            try:
                # 计算缩放比例，确保图片宽度不超过1000像素
                max_width = 1000
                scale = min(1.0, max_width / image.shape[1])
                if scale < 1.0:
                    new_width = int(image.shape[1] * scale)
                    new_height = int(image.shape[0] * scale)
                    image = cv2.resize(image, (new_width, new_height))
                    print(f"缩放后图片尺寸: {image.shape}")

                # 确保图像是 uint8 类型
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # 转换为 RGB 图像用于 Dlib 检测
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"RGB图像类型: {rgb_image.dtype}, 形状: {rgb_image.shape}")

                # 使用 Dlib 检测人脸
                self.current_faces = self.detector(rgb_image, 1)
                print(f"检测到的人脸数量: {len(self.current_faces)}")

                # 保存 RGB 图像以便后续处理
                self.current_image = rgb_image
                self.current_image_path = processed_path
                
                # 清除之前的选择（加载新图片时）
                self.selected_faces.clear()
                
                # 显示图片和人脸框
                self.display_image()
                
                # 更新状态栏
                if len(self.current_faces) > 0:
                    self.update_status(f"检测到 {len(self.current_faces)} 个人脸 - 点击人脸框或使用复选框选择")
                else:
                    self.update_status("未检测到人脸")
                    
            except Exception as e:
                print(f"处理错误: {str(e)}")
                print(f"图像类型: {image.dtype if 'image' in locals() else 'unknown'}")
                print(f"图像形状: {image.shape if 'image' in locals() else 'unknown'}")
                self.update_status("图片处理失败")
                messagebox.showerror("错误", f"图像处理失败: {str(e)}")
                return

    
    def display_image(self):
        """显示图片和人脸框"""
        if self.current_image is None:
            return
        
        # 创建图片副本用于绘制
        display_image = self.current_image.copy()
        
        # 清除之前的人脸选择界面
        self.clear_face_selection()
        
        # 绘制人脸框
        for i, face in enumerate(self.current_faces):
            # 根据是否选中使用不同颜色
            if i in self.selected_faces:
                color = (255, 0, 0)  # 红色表示选中
                thickness = 3
            else:
                color = (0, 255, 0)  # 绿色表示未选中
                thickness = 2
            
            # 绘制边框
            cv2.rectangle(display_image, 
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        color, thickness)
            
            # 添加人脸编号
            cv2.putText(display_image, f"Face {i+1}", 
                       (face.left(), face.top()-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 转换为PIL图像
        image = Image.fromarray(display_image)
        # 调整大小以适应显示
        image.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(image)
        
        # 更新显示
        self.label_image.configure(image=photo)
        self.label_image.image = photo
        
        # 更新人脸选择界面
        self.update_face_selection()
        
        # 绑定点击事件
        self.label_image.bind('<Button-1>', self.on_image_click)
    
    def clear_face_selection(self):
        """清除人脸选择界面"""
        for widget in self.face_selection_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                widget.destroy()
        self.face_vars.clear()
    
    def update_face_selection(self):
        """更新人脸选择界面"""
        # 清除之前的复选框
        for widget in self.face_selection_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                widget.destroy()
        self.face_vars.clear()
        
        # 为每个检测到的人脸创建复选框
        for i, face in enumerate(self.current_faces):
            var = tk.BooleanVar()
            # 设置复选框状态与selected_faces同步
            if i in self.selected_faces:
                var.set(True)
            self.face_vars.append(var)
            
            # 创建复选框
            cb = tk.Checkbutton(self.face_selection_frame, 
                               text=f"人脸 {i+1}", 
                               variable=var,
                               font=('Microsoft YaHei UI', 10),
                               fg='#2c3e50', bg='white',
                               selectcolor='#3498db',
                               command=lambda idx=i, v=var: self.on_face_selection_change(idx, v))
            cb.pack(anchor=tk.W, pady=2)
        
        # 更新保存按钮文本
        self.update_save_button_text()
    
    def on_face_selection_change(self, face_index, var):
        """处理人脸选择变化"""
        if var.get():
            if face_index not in self.selected_faces:
                self.selected_faces.append(face_index)
        else:
            if face_index in self.selected_faces:
                self.selected_faces.remove(face_index)
        
        # 只更新图片显示，不重新创建复选框
        self.update_image_display()
        
        # 更新保存按钮文本
        self.update_save_button_text()
    
    def update_image_display(self):
        """更新图片显示，不重新创建复选框"""
        if self.current_image is None:
            return
        
        # 创建图片副本用于绘制
        display_image = self.current_image.copy()
        
        # 绘制人脸框
        for i, face in enumerate(self.current_faces):
            # 根据是否选中使用不同颜色
            if i in self.selected_faces:
                color = (255, 0, 0)  # 红色表示选中
                thickness = 3
            else:
                color = (0, 255, 0)  # 绿色表示未选中
                thickness = 2
            
            # 绘制边框
            cv2.rectangle(display_image, 
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        color, thickness)
            
            # 添加人脸编号
            cv2.putText(display_image, f"Face {i+1}", 
                       (face.left(), face.top()-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 转换为PIL图像
        image = Image.fromarray(display_image)
        # 调整大小以适应显示
        image.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(image)
        
        # 更新显示
        self.label_image.configure(image=photo)
        self.label_image.image = photo
    
    def on_image_click(self, event):
        """处理图片点击事件"""
        if not self.current_faces:
            return
        
        # 获取点击位置相对于图片的坐标
        widget = event.widget
        image_width = widget.winfo_width()
        image_height = widget.winfo_height()
        
        # 获取图片的实际显示尺寸
        if hasattr(widget, 'image') and widget.image:
            # 这里需要根据实际显示比例计算点击位置
            # 简化处理：假设图片居中显示
            click_x = event.x
            click_y = event.y
            
            # 检查点击是否在任何人脸框内
            for i, face in enumerate(self.current_faces):
                # 这里需要根据实际显示比例调整坐标
                # 简化处理：直接使用原始坐标
                if (face.left() <= click_x <= face.right() and 
                    face.top() <= click_y <= face.bottom()):
                    # 切换选中状态
                    if i in self.selected_faces:
                        self.selected_faces.remove(i)
                        if i < len(self.face_vars):
                            self.face_vars[i].set(False)
                    else:
                        self.selected_faces.append(i)
                        if i < len(self.face_vars):
                            self.face_vars[i].set(True)
                    
                    # 只更新图片显示，不重新创建复选框
                    self.update_image_display()
                    break
    
    def get_next_available_filename(self, save_dir, base_name, extension=".jpg"):
        """获取下一个可用的文件名"""
        counter = 1
        while True:
            filename = f"{base_name}_{counter:03d}{extension}"
            filepath = os.path.join(save_dir, filename)
            if not os.path.exists(filepath):
                return filename
            counter += 1
    
    def get_image_extension(self, file_path):
        """获取图片文件扩展名"""
        _, ext = os.path.splitext(file_path)
        return ext.lower()
    
    def save_selected_face(self):
        """保存选中的人脸到数据库"""
        if not self.current_faces:
            messagebox.showwarning("警告", "请先选择包含人脸的图片")
            return
        
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("警告", "请输入姓名")
            return
        
        id_number = self.entry_id.get().strip()
        if not id_number:
            messagebox.showwarning("警告", "请输入身份证号")
            return
        
        self.update_status(f"正在保存 {name} 的人脸数据到数据库...")
        
        # 确定要保存的人脸：优先保存选中的，如果没有选中则保存所有人脸
        if self.selected_faces:
            faces_to_save = [self.current_faces[i] for i in self.selected_faces]
        else:
            faces_to_save = self.current_faces
        
        # 生成临时身份信息（参考screen_face_monitor.py中的自动发现新面孔）
        temp_name, temp_id = self.generate_temp_identity()
        
        # 添加人员到数据库，将输入框中的信息保存为real_name和real_id_card
        try:
            person_id = self.db_manager.add_person(
                name=temp_name, 
                id_card=temp_id, 
                is_temp=False,  # 手动注册的不标记为临时
                real_name=name,  # 输入框中的姓名作为真实姓名
                real_id_card=id_number  # 输入框中的身份证号作为真实身份证号
            )
            print(f"添加人员到数据库成功: 临时身份 {temp_name}_{temp_id} -> 真实身份 {name}_{id_number} (ID: {person_id})")
        except Exception as e:
            print(f"添加人员到数据库失败: {str(e)}")
            self.update_status("添加人员到数据库失败")
            messagebox.showerror("错误", f"添加人员到数据库失败: {str(e)}")
            return
        
        # 保存每个人脸
        saved_count = 0
        
        for i, face in enumerate(faces_to_save):
            try:
                # 提取人脸区域
                face_image = self.current_image[face.top():face.bottom(), face.left():face.right()]
                
                # 确保图片是uint8类型
                if face_image.dtype != np.uint8:
                    face_image = face_image.astype(np.uint8)
                
                # 提取人脸特征向量
                try:
                    # 使用dlib提取68个关键点
                    shape = self.predictor(self.current_image, face)
                    # 计算128维特征向量
                    feature = self.face_reco_model.compute_face_descriptor(self.current_image, shape)
                    print(f"成功提取人脸 {i+1} 的特征向量")
                except Exception as feature_error:
                    print(f"提取人脸 {i+1} 特征向量失败: {str(feature_error)}")
                    feature = None
                
                # 转换为BGR格式并编码为JPEG
                bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                success, encoded_image = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    # 保存到数据库
                    image_data = encoded_image.tobytes()
                    image_id = self.db_manager.add_face_image(person_id, image_data, 'jpg')
                    print(f"成功保存人脸图像到数据库: 图像ID {image_id}")
                    
                    # 保存特征向量
                    if feature is not None:
                        feature_id = self.db_manager.add_face_feature(person_id, feature)
                        print(f"成功保存人脸特征向量到数据库: 特征ID {feature_id}")
                    
                    saved_count += 1
                else:
                    print(f"编码人脸图像失败")
                    # 尝试使用PIL作为备用方案
                    try:
                        pil_image = Image.fromarray(face_image)
                        import io
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format='JPEG', quality=95)
                        image_data = img_buffer.getvalue()
                        image_id = self.db_manager.add_face_image(person_id, image_data, 'jpg')
                        print(f"使用PIL成功保存人脸图像到数据库: 图像ID {image_id}")
                        
                        # 保存特征向量
                        if feature is not None:
                            feature_id = self.db_manager.add_face_feature(person_id, feature)
                            print(f"成功保存人脸特征向量到数据库: 特征ID {feature_id}")
                        
                        saved_count += 1
                    except Exception as pil_error:
                        print(f"PIL保存也失败: {str(pil_error)}")
                        
            except Exception as e:
                print(f"保存第 {i+1} 张人脸图像时出错: {str(e)}")
                continue
        
        if saved_count > 0:
            self.update_status(f"已保存 {saved_count} 张人脸图像到数据库")
            messagebox.showinfo("成功", f"已保存 {saved_count} 张人脸图像到数据库\n临时身份: {temp_name}_{temp_id}\n真实身份: {name}_{id_number}")
            # 更新已注册人名列表
            self.load_registered_names()
        else:
            self.update_status("保存失败")
            messagebox.showerror("错误", "没有成功保存任何图像到数据库")
    
    def save_multiple_faces_with_names(self):
        """为多个人脸分别指定姓名保存到数据库"""
        if not self.current_faces:
            messagebox.showwarning("警告", "请先选择包含人脸的图片")
            return
        
        if not self.selected_faces:
            messagebox.showwarning("警告", "请先选择要保存的人脸")
            return
        
        # 创建批量保存对话框
        dialog = tk.Toplevel(self.win)
        dialog.title("批量保存人脸到数据库")
        dialog.geometry("500x600")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.win)
        dialog.grab_set()
        
        # 对话框标题
        title_label = tk.Label(dialog, text="为每个人脸指定姓名和身份证号", 
                              font=('Microsoft YaHei UI', 14, 'bold'), 
                              fg='#2c3e50', bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # 创建输入框和自动完成列表
        name_entries = []
        id_entries = []
        
        for i, face_idx in enumerate(self.selected_faces):
            # 为每个人脸创建一个框架
            face_frame = tk.Frame(dialog, bg='#f0f0f0')
            face_frame.pack(pady=8, padx=20, fill=tk.X)
            
            # 标签
            tk.Label(face_frame, text=f"人脸 {face_idx+1}:", 
                    font=('Microsoft YaHei UI', 10, 'bold'), 
                    fg='#2c3e50', bg='#f0f0f0').pack(anchor=tk.W, pady=(0, 5))
            
            # 姓名输入框架
            name_input_frame = tk.Frame(face_frame, bg='#f0f0f0')
            name_input_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(name_input_frame, text="姓名:", 
                    font=('Microsoft YaHei UI', 9), 
                    fg='#2c3e50', bg='#f0f0f0', width=8).pack(side=tk.LEFT)
            
            # 姓名输入框
            name_entry = tk.Entry(name_input_frame, font=('Microsoft YaHei UI', 9))
            name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
            name_entries.append(name_entry)
            
            # 身份证号输入框架
            id_input_frame = tk.Frame(face_frame, bg='#f0f0f0')
            id_input_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(id_input_frame, text="身份证号:", 
                    font=('Microsoft YaHei UI', 9), 
                    fg='#2c3e50', bg='#f0f0f0', width=8).pack(side=tk.LEFT)
            
            # 身份证号输入框
            id_entry = tk.Entry(id_input_frame, font=('Microsoft YaHei UI', 9))
            id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
            id_entries.append(id_entry)
        
        # 保存按钮
        def save_batch():
            names = [entry.get().strip() for entry in name_entries]
            id_numbers = [entry.get().strip() for entry in id_entries]
            
            if not all(names):
                messagebox.showwarning("警告", "请为所有人脸输入姓名")
                return
            
            if not all(id_numbers):
                messagebox.showwarning("警告", "请为所有人脸输入身份证号")
                return
            
            dialog.destroy()
            self.batch_save_faces(names, id_numbers)
        
        save_btn = tk.Button(dialog, text="保存到数据库", command=save_batch,
                            font=('Microsoft YaHei UI', 12),
                            bg='#27ae60', fg='white',
                            relief=tk.FLAT, padx=20, pady=8)
        save_btn.pack(pady=20)
        
        # 存储对话框引用以便后续使用
        dialog.name_entries = name_entries
        dialog.id_entries = id_entries
    
    def batch_save_faces(self, names, id_numbers):
        """批量保存不同姓名的人脸到数据库"""
        self.update_status("正在批量保存人脸到数据库...")
        
        saved_count = 0
        for i, (face_idx, name, id_number) in enumerate(zip(self.selected_faces, names, id_numbers)):
            if not name or not id_number:
                continue
            
            try:
                # 生成临时身份信息
                temp_name, temp_id = self.generate_temp_identity()
                
                # 添加人员到数据库，将输入框中的信息保存为real_name和real_id_card
                person_id = self.db_manager.add_person(
                    name=temp_name, 
                    id_card=temp_id, 
                    is_temp=False,  # 手动注册的不标记为临时
                    real_name=name,  # 输入框中的姓名作为真实姓名
                    real_id_card=id_number  # 输入框中的身份证号作为真实身份证号
                )
                print(f"添加人员到数据库成功: 临时身份 {temp_name}_{temp_id} -> 真实身份 {name}_{id_number} (ID: {person_id})")
                
                # 保存人脸
                face = self.current_faces[face_idx]
                face_image = self.current_image[face.top():face.bottom(), face.left():face.right()]
                
                # 确保图片是uint8类型
                if face_image.dtype != np.uint8:
                    face_image = face_image.astype(np.uint8)
                
                # 提取人脸特征向量
                try:
                    # 使用dlib提取68个关键点
                    shape = self.predictor(self.current_image, face)
                    # 计算128维特征向量
                    feature = self.face_reco_model.compute_face_descriptor(self.current_image, shape)
                    print(f"成功提取人脸 {i+1} 的特征向量")
                except Exception as feature_error:
                    print(f"提取人脸 {i+1} 特征向量失败: {str(feature_error)}")
                    feature = None
                
                # 转换为BGR格式并编码为JPEG
                bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                success, encoded_image = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    # 保存到数据库
                    image_data = encoded_image.tobytes()
                    image_id = self.db_manager.add_face_image(person_id, image_data, 'jpg')
                    print(f"成功保存 {name} 的人脸图像到数据库: 图像ID {image_id}")
                    
                    # 保存特征向量
                    if feature is not None:
                        feature_id = self.db_manager.add_face_feature(person_id, feature)
                        print(f"成功保存人脸特征向量到数据库: 特征ID {feature_id}")
                    
                    saved_count += 1
                else:
                    # 尝试PIL保存
                    try:
                        pil_image = Image.fromarray(face_image)
                        import io
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format='JPEG', quality=95)
                        image_data = img_buffer.getvalue()
                        image_id = self.db_manager.add_face_image(person_id, image_data, 'jpg')
                        print(f"使用PIL成功保存 {name} 的人脸图像到数据库: 图像ID {image_id}")
                        
                        # 保存特征向量
                        if feature is not None:
                            feature_id = self.db_manager.add_face_feature(person_id, feature)
                            print(f"成功保存人脸特征向量到数据库: 特征ID {feature_id}")
                        
                        saved_count += 1
                    except Exception as pil_error:
                        print(f"保存 {name} 的人脸图像失败: {str(pil_error)}")
                
            except Exception as e:
                print(f"保存 {name} 的人脸时出错: {str(e)}")
                continue
        
        if saved_count > 0:
            self.load_registered_names()
            self.update_status(f"批量保存完成，共保存 {saved_count} 张人脸到数据库")
            messagebox.showinfo("成功", f"批量保存完成，共保存 {saved_count} 张人脸到数据库")
        else:
            self.update_status("批量保存失败")
            messagebox.showerror("错误", "批量保存失败")
    
    def delete_selected_name(self):
        """从数据库删除选中的人员信息"""
        selection = self.listbox_names.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的人员")
            return
        
        person_id_str = self.listbox_names.get(selection[0])
        if messagebox.askyesno("确认", f"确定要删除 {person_id_str} 的所有人脸数据吗？"):
            self.update_status(f"正在删除 {person_id_str} 的数据...")
            
            try:
                # 解析姓名和身份证号
                if '_' in person_id_str:
                    parts = person_id_str.split('_', 1)
                    display_name = parts[0]
                    display_id = parts[1]
                else:
                    display_name = person_id_str
                    display_id = None
                
                # 从数据库获取人员信息，优先查找real_name和real_id_card
                person_info = self.db_manager.get_person_by_name_id(display_name, display_id)
                
                # 如果没找到，尝试查找real_name和real_id_card
                if not person_info:
                    # 获取所有人员信息进行匹配
                    persons = self.db_manager.get_all_persons(include_temp=False)
                    for person in persons:
                        real_name = person.get('real_name') or person['name']
                        real_id = person.get('real_id_card') or person.get('id_card')
                        
                        if real_name == display_name and real_id == display_id:
                            person_info = person
                            break
                
                if person_info:
                    # 删除人员（会级联删除相关的人脸图像和特征）
                    success = self.db_manager.delete_person(person_info['id'])
                    if success:
                        self.update_status(f"已删除 {person_id_str} 的所有数据")
                        messagebox.showinfo("成功", f"已删除 {person_id_str} 的所有数据")
                        # 更新列表
                        self.load_registered_names()
                    else:
                        self.update_status("删除失败")
                        messagebox.showerror("错误", "删除失败")
                else:
                    self.update_status("未找到该人员")
                    messagebox.showerror("错误", "未找到该人员")
                    
            except Exception as e:
                print(f"删除人员时出错: {str(e)}")
                self.update_status("删除失败")
                messagebox.showerror("错误", f"删除失败: {str(e)}")
    
    def update_save_button_text(self):
        """更新保存按钮文本"""
        if self.selected_faces:
            self.btn_save.config(text="💾 保存选中的人脸")
        else:
            self.btn_save.config(text="💾 保存所有人脸")
    
    def on_name_list_click(self, event):
        """处理已注册人名列表的点击事件"""
        selection = self.listbox_names.curselection()
        if not selection:
            return
        
        person_id = self.listbox_names.get(selection[0])
        if not person_id:
            return
        
        # 解析"姓名_身份证号"格式
        if '_' in person_id:
            parts = person_id.split('_', 1)  # 最多分割1次，保留身份证号中的下划线
            if len(parts) >= 2:
                display_name = parts[0]
                display_id = parts[1]
                
                # 从数据库查找对应的真实身份信息
                persons = self.db_manager.get_all_persons(include_temp=False)
                for person in persons:
                    real_name = person.get('real_name') or person['name']
                    real_id = person.get('real_id_card') or person.get('id_card')
                    
                    if real_name == display_name and real_id == display_id:
                        # 填充真实身份信息
                        self.entry_name.delete(0, tk.END)
                        self.entry_name.insert(0, real_name)
                        self.entry_id.delete(0, tk.END)
                        self.entry_id.insert(0, real_id)
                        return
                
                # 如果没找到匹配的真实身份，使用显示的信息
                self.entry_name.delete(0, tk.END)
                self.entry_name.insert(0, display_name)
                self.entry_id.delete(0, tk.END)
                self.entry_id.insert(0, display_id)
            else:
                # 如果格式不正确，只填充姓名
                self.entry_name.delete(0, tk.END)
                self.entry_name.insert(0, person_id)
        else:
            # 兼容旧格式，只填充姓名
            self.entry_name.delete(0, tk.END)
            self.entry_name.insert(0, person_id)
    
    def run(self):
        """运行程序"""
        self.win.mainloop()
    
    def __del__(self):
        """析构函数，确保数据库连接正确关闭"""
        if hasattr(self, 'db_manager'):
            self.db_manager.close()

if __name__ == "__main__":
    collector = FaceCollector()
    collector.run() 