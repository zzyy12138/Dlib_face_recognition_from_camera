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
        
        # 设置保存路径
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        
        # 创建保存目录
        if not os.path.exists(self.path_photos_from_camera):
            os.makedirs(self.path_photos_from_camera)
        
        # 初始化GUI
        self.win = tk.Tk()
        self.win.title("智能人脸采集工具")
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
        
        # 创建输入框和自动完成框架
        self.entry_frame = tk.Frame(name_frame, bg='white')
        self.entry_frame.pack(fill=tk.X)
        
        self.entry_name = tk.Entry(self.entry_frame, font=('Microsoft YaHei UI', 11),
                                  relief=tk.SOLID, bd=1)
        self.entry_name.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(0, 8))
        
        # 创建自动完成下拉列表
        self.autocomplete_listbox = tk.Listbox(name_frame, 
                                              font=('Microsoft YaHei UI', 10),
                                              relief=tk.SOLID, bd=1,
                                              bg='white', fg='#2c3e50',
                                              selectbackground='#3498db',
                                              selectforeground='white',
                                              height=4)
        
        # 绑定输入事件
        self.entry_name.bind('<KeyRelease>', self.on_name_input)
        self.entry_name.bind('<KeyPress>', self.on_name_keypress)
        self.entry_name.bind('<FocusOut>', self.hide_autocomplete)
        self.entry_name.bind('<FocusIn>', self.on_name_focus)
        
        # 绑定下拉列表事件
        self.autocomplete_listbox.bind('<Double-Button-1>', self.select_autocomplete)
        self.autocomplete_listbox.bind('<Return>', self.select_autocomplete)
        
        # 自动完成相关变量
        self.autocomplete_visible = False
        self.filtered_names = []
        
        # 保存按钮
        self.btn_save = tk.Button(self.frame_right, text="💾 保存所有人脸", 
                                 command=self.save_selected_face,
                                 font=('Microsoft YaHei UI', 12),
                                 bg='#27ae60', fg='white',
                                 relief=tk.FLAT, padx=20, pady=8,
                                 cursor='hand2')
        self.btn_save.pack(pady=8, padx=20, fill=tk.X)
        
        # 批量保存按钮
        self.btn_batch_save = tk.Button(self.frame_right, text="📝 批量保存不同姓名", 
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
        names_title = tk.Label(self.frame_right, text="已注册的人名", 
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
        
        # 删除按钮
        self.btn_delete = tk.Button(self.frame_right, text="🗑️ 删除选中的人名", 
                                   command=self.delete_selected_name,
                                   font=('Microsoft YaHei UI', 11),
                                   bg='#e74c3c', fg='white',
                                   relief=tk.FLAT, padx=15, pady=6,
                                   cursor='hand2')
        self.btn_delete.pack(pady=8, padx=20, fill=tk.X)
        
        # 状态栏
        self.status_frame = tk.Frame(self.win, bg='#34495e', height=30)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="就绪", 
                                    font=('Microsoft YaHei UI', 9),
                                    fg='white', bg='#34495e')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 绑定鼠标悬停效果
        self.bind_hover_effects()
        
    def bind_hover_effects(self):
        """绑定按钮悬停效果"""
        buttons = [self.btn_select, self.btn_save, self.btn_batch_save, self.btn_delete]
        colors = ['#3498db', '#27ae60', '#9b59b6', '#e74c3c']
        hover_colors = ['#2980b9', '#229954', '#8e44ad', '#c0392b']
        
        for btn, color, hover_color in zip(buttons, colors, hover_colors):
            btn.bind('<Enter>', lambda e, b=btn, c=hover_color: self.on_button_hover(b, c))
            btn.bind('<Leave>', lambda e, b=btn, c=color: self.on_button_hover(b, c))
    
    def on_button_hover(self, button, color):
        """按钮悬停效果"""
        button.configure(bg=color)
        
    def update_status(self, message):
        """更新状态栏信息"""
        self.status_label.config(text=message)
        self.win.update_idletasks()

    def load_registered_names(self):
        """加载已注册的人名"""
        if os.path.exists(self.path_photos_from_camera):
            self.registered_names = [d.split('_')[1] for d in os.listdir(self.path_photos_from_camera) 
                                  if os.path.isdir(os.path.join(self.path_photos_from_camera, d))]
            self.update_name_list()
    
    def update_name_list(self):
        """更新人名列表显示"""
        self.listbox_names.delete(0, tk.END)
        for name in self.registered_names:
            self.listbox_names.insert(tk.END, name)
        
        # 如果当前有自动完成列表显示，重新过滤
        if self.autocomplete_visible:
            current_text = self.entry_name.get().strip()
            if current_text:
                self.filtered_names = [name for name in self.registered_names 
                                      if name.lower().startswith(current_text.lower())]
                if self.filtered_names:
                    self.show_autocomplete()
                else:
                    self.hide_autocomplete()
    
    def on_name_input(self, event):
        """处理姓名输入事件"""
        # 忽略特殊键
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'Return', 'Tab']:
            return
        
        current_text = self.entry_name.get().strip()
        
        if not current_text:
            self.hide_autocomplete()
            return
        
        # 过滤匹配的人名
        self.filtered_names = [name for name in self.registered_names 
                              if name.lower().startswith(current_text.lower())]
        
        if self.filtered_names:
            self.show_autocomplete()
        else:
            self.hide_autocomplete()
    
    def on_name_keypress(self, event):
        """处理姓名输入框按键事件"""
        if not self.autocomplete_visible:
            return
        
        if event.keysym == 'Up':
            self.navigate_autocomplete(-1)
            return 'break'
        elif event.keysym == 'Down':
            self.navigate_autocomplete(1)
            return 'break'
        elif event.keysym == 'Return':
            if self.autocomplete_listbox.curselection():
                self.select_autocomplete()
                return 'break'
        elif event.keysym == 'Escape':
            self.hide_autocomplete()
            return 'break'
    
    def on_name_focus(self, event):
        """处理姓名输入框获得焦点事件"""
        current_text = self.entry_name.get().strip()
        if current_text:
            self.on_name_input(event)
    
    def show_autocomplete(self):
        """显示自动完成下拉列表"""
        if not self.filtered_names:
            return
        
        # 清空并填充列表
        self.autocomplete_listbox.delete(0, tk.END)
        for name in self.filtered_names:
            self.autocomplete_listbox.insert(tk.END, name)
        
        # 显示下拉列表
        if not self.autocomplete_visible:
            self.autocomplete_listbox.pack(fill=tk.X, pady=(0, 8))
            self.autocomplete_visible = True
        
        # 选中第一项
        if self.autocomplete_listbox.size() > 0:
            self.autocomplete_listbox.selection_set(0)
    
    def hide_autocomplete(self, event=None):
        """隐藏自动完成下拉列表"""
        if self.autocomplete_visible:
            self.autocomplete_listbox.pack_forget()
            self.autocomplete_visible = False
    
    def navigate_autocomplete(self, direction):
        """在自动完成列表中导航"""
        if not self.autocomplete_visible or not self.filtered_names:
            return
        
        current_selection = self.autocomplete_listbox.curselection()
        if not current_selection:
            self.autocomplete_listbox.selection_set(0)
            return
        
        current_index = current_selection[0]
        new_index = current_index + direction
        
        # 循环选择
        if new_index < 0:
            new_index = len(self.filtered_names) - 1
        elif new_index >= len(self.filtered_names):
            new_index = 0
        
        self.autocomplete_listbox.selection_clear(0, tk.END)
        self.autocomplete_listbox.selection_set(new_index)
        self.autocomplete_listbox.see(new_index)
    
    def select_autocomplete(self, event=None):
        """选择自动完成项"""
        if not self.autocomplete_visible:
            return
        
        selection = self.autocomplete_listbox.curselection()
        if selection:
            selected_name = self.autocomplete_listbox.get(selection[0])
            self.entry_name.delete(0, tk.END)
            self.entry_name.insert(0, selected_name)
            self.hide_autocomplete()
            # 将焦点设置到输入框
            self.entry_name.focus_set()
    
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
        if not self.current_faces:
            return
        
        for i, face in enumerate(self.current_faces):
            var = tk.BooleanVar(value=i in self.selected_faces)
            self.face_vars.append(var)
            
            cb = tk.Checkbutton(self.face_selection_frame, 
                              text=f"人脸 {i+1}", 
                              variable=var,
                              command=lambda idx=i, v=var: self.on_face_selection_change(idx, v),
                              font=('Microsoft YaHei UI', 10), 
                              bg='white', fg='#2c3e50',
                              selectcolor='#3498db')
            cb.pack(anchor=tk.W, pady=1)
    
    def on_face_selection_change(self, face_index, var):
        """人脸选择状态改变时的处理"""
        if var.get():
            if face_index not in self.selected_faces:
                self.selected_faces.append(face_index)
        else:
            if face_index in self.selected_faces:
                self.selected_faces.remove(face_index)
        
        # 更新保存按钮文本
        self.update_save_button_text()
        
        # 重新显示图片以更新颜色
        self.display_image()
    
    def on_image_click(self, event):
        """图片点击事件处理"""
        if not self.current_faces:
            return
        
        # 获取点击位置相对于图片的坐标
        img_width = self.label_image.winfo_width()
        img_height = self.label_image.winfo_height()
        
        # 计算缩放比例
        original_width = self.current_image.shape[1]
        original_height = self.current_image.shape[0]
        
        scale_x = original_width / img_width
        scale_y = original_height / img_height
        
        # 转换点击坐标
        click_x = int(event.x * scale_x)
        click_y = int(event.y * scale_y)
        
        # 检查点击是否在人脸框内
        for i, face in enumerate(self.current_faces):
            if (face.left() <= click_x <= face.right() and 
                face.top() <= click_y <= face.bottom()):
                # 切换选中状态
                if i in self.selected_faces:
                    self.selected_faces.remove(i)
                else:
                    self.selected_faces.append(i)
                
                # 更新复选框状态
                if i < len(self.face_vars):
                    self.face_vars[i].set(i in self.selected_faces)
                
                # 更新保存按钮文本
                self.update_save_button_text()
                
                # 重新显示图片
                self.display_image()
                break

    def get_next_available_filename(self, save_dir, base_name, extension=".jpg"):
        """获取下一个可用的文件名，避免覆盖现有文件"""
        counter = 1
        while True:
            filename = f"{base_name}_{counter}{extension}"
            filepath = os.path.join(save_dir, filename)
            if not os.path.exists(filepath):
                return filename
            counter += 1
    
    def get_image_extension(self, file_path):
        """根据文件路径获取图片扩展名"""
        _, ext = os.path.splitext(file_path.lower())
        # 支持的格式映射
        format_map = {
            '.jpg': '.jpg',
            '.jpeg': '.jpg',
            '.png': '.png',
            '.bmp': '.bmp',
            '.gif': '.gif',
            '.tiff': '.tiff',
            '.tif': '.tiff',
            '.webp': '.webp',
            '.ico': '.png'  # ICO格式转换为PNG保存
        }
        return format_map.get(ext, '.jpg')  # 默认使用jpg格式
    
    def save_selected_face(self):
        """保存选中的人脸"""
        if not self.current_faces:
            messagebox.showwarning("警告", "请先选择包含人脸的图片")
            return
        
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("警告", "请输入姓名")
            return
        
        self.update_status(f"正在保存 {name} 的人脸数据...")
        
        # 确定要保存的人脸：优先保存选中的，如果没有选中则保存所有人脸
        if self.selected_faces:
            faces_to_save = [self.current_faces[i] for i in self.selected_faces]
        else:
            faces_to_save = self.current_faces
        
        # 创建保存目录
        save_dir = os.path.join(self.path_photos_from_camera, f"person_{name}")
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"创建目录: {save_dir}")
        except Exception as e:
            print(f"创建目录失败: {str(e)}")
            self.update_status("创建目录失败")
            messagebox.showerror("错误", f"创建保存目录失败: {str(e)}")
            return
        
        # 保存每个人脸
        saved_count = 0
        extension = ".jpg"  # 统一使用jpg格式
        
        for i, face in enumerate(faces_to_save):
            try:
                # 提取人脸区域
                face_image = self.current_image[face.top():face.bottom(), face.left():face.right()]
                
                # 获取唯一的文件名
                filename = self.get_next_available_filename(save_dir, "img_face", extension)
                save_path = os.path.normpath(os.path.join(save_dir, filename))
                
                print(f"正在保存图片到: {save_path}")
                print(f"图片尺寸: {face_image.shape}")
                
                # 确保图片是uint8类型
                if face_image.dtype != np.uint8:
                    face_image = face_image.astype(np.uint8)
                
                # 转换为BGR格式并保存为JPEG
                bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(save_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                if success:
                    saved_count += 1
                    print(f"成功保存图片: {save_path}")
                else:
                    print(f"保存图片失败: {save_path}")
                    # 尝试使用PIL作为备用方案
                    try:
                        Image.fromarray(face_image).save(save_path, 'JPEG', quality=95)
                        saved_count += 1
                        print(f"使用PIL成功保存图片: {save_path}")
                    except Exception as pil_error:
                        print(f"PIL保存也失败: {str(pil_error)}")
                        # 最后尝试：使用绝对路径
                        try:
                            abs_save_path = os.path.abspath(save_path)
                            print(f"尝试使用绝对路径保存: {abs_save_path}")
                            Image.fromarray(face_image).save(abs_save_path, 'JPEG', quality=95)
                            saved_count += 1
                            print(f"使用绝对路径成功保存图片: {abs_save_path}")
                        except Exception as abs_error:
                            print(f"绝对路径保存也失败: {str(abs_error)}")
            except Exception as e:
                print(f"保存第 {i+1} 张图片时出错: {str(e)}")
                continue
        
        if saved_count > 0:
            self.update_status(f"已保存 {saved_count} 张人脸图片")
            messagebox.showinfo("成功", f"已保存 {saved_count} 张人脸图片")
            # 更新已注册人名列表
            if name not in self.registered_names:
                self.registered_names.append(name)
                self.update_name_list()
        else:
            self.update_status("保存失败")
            messagebox.showerror("错误", "没有成功保存任何图片")
    
    def save_multiple_faces_with_names(self):
        """为多个人脸分别指定姓名保存"""
        if not self.current_faces:
            messagebox.showwarning("警告", "请先选择包含人脸的图片")
            return
        
        if not self.selected_faces:
            messagebox.showwarning("警告", "请先选择要保存的人脸")
            return
        
        # 创建批量保存对话框
        dialog = tk.Toplevel(self.win)
        dialog.title("批量保存人脸")
        dialog.geometry("400x500")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.win)
        dialog.grab_set()
        
        # 对话框标题
        title_label = tk.Label(dialog, text="为每个人脸指定姓名", 
                              font=('Microsoft YaHei UI', 14, 'bold'), 
                              fg='#2c3e50', bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # 创建输入框和自动完成列表
        name_entries = []
        autocomplete_listboxes = []
        
        for i, face_idx in enumerate(self.selected_faces):
            # 为每个人脸创建一个框架
            face_frame = tk.Frame(dialog, bg='#f0f0f0')
            face_frame.pack(pady=5, padx=20, fill=tk.X)
            
            # 标签
            tk.Label(face_frame, text=f"人脸 {face_idx+1}:", 
                    font=('Microsoft YaHei UI', 10), 
                    fg='#2c3e50', bg='#f0f0f0').pack(side=tk.LEFT)
            
            # 输入框框架
            entry_frame = tk.Frame(face_frame, bg='#f0f0f0')
            entry_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
            
            # 输入框
            entry = tk.Entry(entry_frame, font=('Microsoft YaHei UI', 10))
            entry.pack(fill=tk.X)
            name_entries.append(entry)
            
            # 自动完成下拉列表
            autocomplete_listbox = tk.Listbox(entry_frame, 
                                            font=('Microsoft YaHei UI', 9),
                                            relief=tk.SOLID, bd=1,
                                            bg='white', fg='#2c3e50',
                                            selectbackground='#3498db',
                                            selectforeground='white',
                                            height=3)
            autocomplete_listboxes.append(autocomplete_listbox)
            
            # 绑定输入事件
            entry.bind('<KeyRelease>', lambda e, idx=i: self.on_batch_name_input(e, idx, name_entries, autocomplete_listboxes))
            entry.bind('<KeyPress>', lambda e, idx=i: self.on_batch_name_keypress(e, idx, autocomplete_listboxes))
            entry.bind('<FocusOut>', lambda e, idx=i: self.hide_batch_autocomplete(idx, autocomplete_listboxes))
            entry.bind('<FocusIn>', lambda e, idx=i: self.on_batch_name_focus(e, idx, name_entries, autocomplete_listboxes))
            
            # 绑定下拉列表事件
            autocomplete_listbox.bind('<Double-Button-1>', lambda e, idx=i: self.select_batch_autocomplete(e, idx, name_entries, autocomplete_listboxes))
            autocomplete_listbox.bind('<Return>', lambda e, idx=i: self.select_batch_autocomplete(e, idx, name_entries, autocomplete_listboxes))
        
        # 保存按钮
        def save_batch():
            names = [entry.get().strip() for entry in name_entries]
            if not all(names):
                messagebox.showwarning("警告", "请为所有人脸输入姓名")
                return
            
            dialog.destroy()
            self.batch_save_faces(names)
        
        save_btn = tk.Button(dialog, text="保存", command=save_batch,
                            font=('Microsoft YaHei UI', 12),
                            bg='#27ae60', fg='white',
                            relief=tk.FLAT, padx=20, pady=8)
        save_btn.pack(pady=20)
        
        # 存储对话框引用以便后续使用
        dialog.name_entries = name_entries
        dialog.autocomplete_listboxes = autocomplete_listboxes
    
    def batch_save_faces(self, names):
        """批量保存不同姓名的人脸"""
        self.update_status("正在批量保存人脸...")
        
        saved_count = 0
        for i, (face_idx, name) in enumerate(zip(self.selected_faces, names)):
            if not name:
                continue
            
            # 创建保存目录
            save_dir = os.path.join(self.path_photos_from_camera, f"person_{name}")
            try:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            except Exception as e:
                print(f"创建目录失败: {str(e)}")
                continue
            
            # 保存人脸
            try:
                face = self.current_faces[face_idx]
                face_image = self.current_image[face.top():face.bottom(), face.left():face.right()]
                
                filename = self.get_next_available_filename(save_dir, "img_face", ".jpg")
                save_path = os.path.join(save_dir, filename)
                
                # 确保图片是uint8类型
                if face_image.dtype != np.uint8:
                    face_image = face_image.astype(np.uint8)
                
                # 保存图片
                bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(save_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    saved_count += 1
                    print(f"成功保存 {name} 的人脸: {save_path}")
                else:
                    # 尝试PIL保存
                    try:
                        Image.fromarray(face_image).save(save_path, 'JPEG', quality=95)
                        saved_count += 1
                        print(f"使用PIL成功保存 {name} 的人脸: {save_path}")
                    except:
                        print(f"保存 {name} 的人脸失败")
                
                # 更新已注册人名列表
                if name not in self.registered_names:
                    self.registered_names.append(name)
                
            except Exception as e:
                print(f"保存 {name} 的人脸时出错: {str(e)}")
                continue
        
        if saved_count > 0:
            self.update_name_list()
            self.update_status(f"批量保存完成，共保存 {saved_count} 张人脸")
            messagebox.showinfo("成功", f"批量保存完成，共保存 {saved_count} 张人脸")
        else:
            self.update_status("批量保存失败")
            messagebox.showerror("错误", "批量保存失败")
    
    def delete_selected_name(self):
        """删除选中的人名"""
        selection = self.listbox_names.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的人名")
            return
        
        name = self.listbox_names.get(selection[0])
        if messagebox.askyesno("确认", f"确定要删除 {name} 的所有人脸数据吗？"):
            self.update_status(f"正在删除 {name} 的数据...")
            
            # 删除文件夹
            folder_path = os.path.join(self.path_photos_from_camera, f"person_{name}")
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            
            # 更新列表
            self.registered_names.remove(name)
            self.update_name_list()
            
            self.update_status(f"已删除 {name} 的所有人脸数据")
            messagebox.showinfo("成功", f"已删除 {name} 的所有人脸数据")
    
    def update_save_button_text(self):
        """更新保存按钮文本"""
        if self.selected_faces:
            self.btn_save.config(text="💾 保存选中的人脸")
        else:
            self.btn_save.config(text="💾 保存所有人脸")
    
    def on_batch_name_input(self, event, idx, name_entries, autocomplete_listboxes):
        """处理批量保存对话框中的姓名输入事件"""
        # 忽略特殊键
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'Return', 'Tab']:
            return
        
        current_text = name_entries[idx].get().strip()
        
        if not current_text:
            self.hide_batch_autocomplete(idx, autocomplete_listboxes)
            return
        
        # 过滤匹配的人名
        filtered_names = [name for name in self.registered_names 
                         if name.lower().startswith(current_text.lower())]
        
        if filtered_names:
            self.show_batch_autocomplete(idx, filtered_names, autocomplete_listboxes)
        else:
            self.hide_batch_autocomplete(idx, autocomplete_listboxes)
    
    def on_batch_name_keypress(self, event, idx, autocomplete_listboxes):
        """处理批量保存对话框中的按键事件"""
        listbox = autocomplete_listboxes[idx]
        
        if not listbox.winfo_viewable():
            return
        
        if event.keysym == 'Up':
            self.navigate_batch_autocomplete(idx, -1, autocomplete_listboxes)
            return 'break'
        elif event.keysym == 'Down':
            self.navigate_batch_autocomplete(idx, 1, autocomplete_listboxes)
            return 'break'
        elif event.keysym == 'Return':
            if listbox.curselection():
                self.select_batch_autocomplete(event, idx, None, autocomplete_listboxes)
                return 'break'
        elif event.keysym == 'Escape':
            self.hide_batch_autocomplete(idx, autocomplete_listboxes)
            return 'break'
    
    def on_batch_name_focus(self, event, idx, name_entries, autocomplete_listboxes):
        """处理批量保存对话框中的焦点事件"""
        current_text = name_entries[idx].get().strip()
        if current_text:
            self.on_batch_name_input(event, idx, name_entries, autocomplete_listboxes)
    
    def show_batch_autocomplete(self, idx, filtered_names, autocomplete_listboxes):
        """显示批量保存对话框中的自动完成列表"""
        listbox = autocomplete_listboxes[idx]
        
        if not filtered_names:
            return
        
        # 清空并填充列表
        listbox.delete(0, tk.END)
        for name in filtered_names:
            listbox.insert(tk.END, name)
        
        # 显示下拉列表
        if not listbox.winfo_viewable():
            listbox.pack(fill=tk.X, pady=(2, 0))
        
        # 选中第一项
        if listbox.size() > 0:
            listbox.selection_set(0)
    
    def hide_batch_autocomplete(self, idx, autocomplete_listboxes):
        """隐藏批量保存对话框中的自动完成列表"""
        listbox = autocomplete_listboxes[idx]
        if listbox.winfo_viewable():
            listbox.pack_forget()
    
    def navigate_batch_autocomplete(self, idx, direction, autocomplete_listboxes):
        """在批量保存对话框的自动完成列表中导航"""
        listbox = autocomplete_listboxes[idx]
        
        if not listbox.winfo_viewable():
            return
        
        current_selection = listbox.curselection()
        if not current_selection:
            listbox.selection_set(0)
            return
        
        current_index = current_selection[0]
        new_index = current_index + direction
        
        # 循环选择
        if new_index < 0:
            new_index = listbox.size() - 1
        elif new_index >= listbox.size():
            new_index = 0
        
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(new_index)
        listbox.see(new_index)
    
    def select_batch_autocomplete(self, event, idx, name_entries, autocomplete_listboxes):
        """选择批量保存对话框中的自动完成项"""
        listbox = autocomplete_listboxes[idx]
        
        if not listbox.winfo_viewable():
            return
        
        selection = listbox.curselection()
        if selection:
            selected_name = listbox.get(selection[0])
            # 获取对应的输入框
            if name_entries:
                entry = name_entries[idx]
                entry.delete(0, tk.END)
                entry.insert(0, selected_name)
            self.hide_batch_autocomplete(idx, autocomplete_listboxes)
            # 将焦点设置到输入框
            if name_entries:
                entry.focus_set()
    
    def run(self):
        """运行程序"""
        self.win.mainloop()

if __name__ == "__main__":
    collector = FaceCollector()
    collector.run() 