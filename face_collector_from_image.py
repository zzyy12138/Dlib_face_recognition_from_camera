import cv2
import dlib
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
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
        self.win.title("图片人脸采集")
        self.win.geometry("1000x600")
        
        # 创建界面元素
        self.create_widgets()
        
        # 初始化变量
        self.current_image = None
        self.current_faces = []
        self.registered_names = []
        self.current_image_path = ""  # 存储当前选择的图片路径
        self.load_registered_names()
        
    def create_widgets(self):
        # 左侧图片显示区域
        self.frame_left = tk.Frame(self.win)
        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.label_image = tk.Label(self.frame_left)
        self.label_image.pack()
        
        # 右侧控制区域
        self.frame_right = tk.Frame(self.win)
        self.frame_right.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 选择图片按钮
        self.btn_select = tk.Button(self.frame_right, text="选择图片", command=self.select_image)
        self.btn_select.pack(pady=10)
        
        # 姓名输入框
        tk.Label(self.frame_right, text="输入姓名:").pack(pady=5)
        self.entry_name = tk.Entry(self.frame_right)
        self.entry_name.pack(pady=5)
        
        # 保存格式选择
        tk.Label(self.frame_right, text="保存格式:").pack(pady=5)
        self.format_var = tk.StringVar(value="jpg")
        format_combo = tk.OptionMenu(self.frame_right, self.format_var, "jpg", "png", "bmp", "tiff")
        format_combo.pack(pady=5)
        
        # 保存按钮
        self.btn_save = tk.Button(self.frame_right, text="保存选中的人脸", command=self.save_selected_face)
        self.btn_save.pack(pady=10)
        
        # 显示已注册的人名
        tk.Label(self.frame_right, text="已注册的人名:").pack(pady=5)
        self.listbox_names = tk.Listbox(self.frame_right, width=30, height=10)
        self.listbox_names.pack(pady=5)
        
        # 删除按钮
        self.btn_delete = tk.Button(self.frame_right, text="删除选中的人名", command=self.delete_selected_name)
        self.btn_delete.pack(pady=10)
        
        # 刷新特征文件按钮
        self.btn_refresh = tk.Button(self.frame_right, text="刷新特征文件", command=lambda: self.refresh_features_csv())
        self.btn_refresh.pack(pady=10)
        
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
                    messagebox.showerror("错误", f"无法读取图片文件: {file_path}\n请确保文件存在且格式正确\n错误信息: {str(pil_error)}")
                    return
            
            if image is None:
                print(f"无法读取图片，请检查文件是否存在且格式正确: {processed_path}")
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
                
                # 自动检测原始格式并建议保存格式
                original_ext = os.path.splitext(processed_path.lower())[1]
                if original_ext in ['.jpg', '.jpeg']:
                    self.format_var.set('jpg')
                elif original_ext == '.png':
                    self.format_var.set('png')
                elif original_ext == '.bmp':
                    self.format_var.set('bmp')
                elif original_ext in ['.tiff', '.tif']:
                    self.format_var.set('tiff')
                elif original_ext == '.webp':
                    self.format_var.set('png')  # WebP转换为PNG
                elif original_ext == '.ico':
                    self.format_var.set('png')  # ICO转换为PNG
                else:
                    self.format_var.set('jpg')  # 默认使用jpg
                
                print(f"检测到原始格式: {original_ext}, 建议保存格式: {self.format_var.get()}")

                # 显示图片和人脸框
                self.display_image()
            except Exception as e:
                print(f"处理错误: {str(e)}")
                print(f"图像类型: {image.dtype if 'image' in locals() else 'unknown'}")
                print(f"图像形状: {image.shape if 'image' in locals() else 'unknown'}")
                messagebox.showerror("错误", f"图像处理失败: {str(e)}")
                return

    
    def display_image(self):
        """显示图片和人脸框"""
        if self.current_image is None:
            return
        
        # 创建图片副本用于绘制
        display_image = self.current_image.copy()
        
        # 绘制人脸框
        for face in self.current_faces:
            cv2.rectangle(display_image, 
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        (0, 255, 0), 2)
        
        # 转换为PIL图像
        image = Image.fromarray(display_image)
        # 调整大小以适应显示
        image.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(image)
        
        # 更新显示
        self.label_image.configure(image=photo)
        self.label_image.image = photo
    
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
    
    def refresh_features_csv(self, person_name=None):
        """刷新特征CSV文件"""
        try:
            print("开始刷新特征文件...")
            
            # 显示进度提示
            if person_name:
                messagebox.showinfo("处理中", f"正在为 {person_name} 提取人脸特征，请稍候...")
            else:
                messagebox.showinfo("处理中", "正在刷新特征文件，请稍候...")
            
            # 调用特征提取脚本
            result = subprocess.run(
                [sys.executable, "features_extraction_to_csv.py"], 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                timeout=60  # 设置60秒超时
            )
            
            if result.returncode == 0:
                print("特征文件刷新成功")
                print(f"输出信息: {result.stdout}")
                if person_name:
                    messagebox.showinfo("成功", f"已保存 {person_name} 的人脸图片并刷新特征文件")
                else:
                    messagebox.showinfo("成功", "特征文件已刷新")
            else:
                print(f"特征文件刷新失败: {result.stderr}")
                error_msg = result.stderr if result.stderr else "未知错误"
                messagebox.showwarning("警告", f"人脸图片保存成功，但特征文件刷新失败:\n{error_msg}")
                
        except subprocess.TimeoutExpired:
            print("特征提取超时")
            messagebox.showwarning("警告", "特征提取超时，请检查图片质量或稍后手动刷新")
        except FileNotFoundError:
            print("找不到特征提取脚本")
            messagebox.showwarning("警告", "找不到 features_extraction_to_csv.py 文件")
        except Exception as e:
            print(f"调用特征提取脚本时出错: {str(e)}")
            messagebox.showwarning("警告", f"人脸图片保存成功，但特征文件刷新失败:\n{str(e)}")

    def save_selected_face(self):
        """保存选中的人脸"""
        if not self.current_faces:
            messagebox.showwarning("警告", "请先选择包含人脸的图片")
            return
        
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("警告", "请输入姓名")
            return
        
        # 创建保存目录
        save_dir = os.path.join(self.path_photos_from_camera, f"person_{name}")
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"创建目录: {save_dir}")
        except Exception as e:
            print(f"创建目录失败: {str(e)}")
            messagebox.showerror("错误", f"创建保存目录失败: {str(e)}")
            return
        
        # 保存每个人脸
        saved_count = 0
        # 获取用户选择的保存格式
        save_format = self.format_var.get()
        extension = f".{save_format}"
        
        for i, face in enumerate(self.current_faces):
            try:
                # 提取人脸区域
                face_image = self.current_image[face.top():face.bottom(), face.left():face.right()]
                
                # 获取唯一的文件名，使用用户选择的格式
                filename = self.get_next_available_filename(save_dir, "img_face", extension)
                save_path = os.path.normpath(os.path.join(save_dir, filename))
                
                print(f"正在保存图片到: {save_path}")
                print(f"图片尺寸: {face_image.shape}")
                print(f"保存格式: {save_format}")
                
                # 确保图片是uint8类型
                if face_image.dtype != np.uint8:
                    face_image = face_image.astype(np.uint8)
                
                # 根据格式选择保存方法
                if save_format.lower() in ['jpg', 'jpeg']:
                    # 转换为BGR格式并保存
                    bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(save_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif save_format.lower() == 'png':
                    # PNG格式，使用PIL保存以保持透明度支持
                    success = False
                    try:
                        Image.fromarray(face_image).save(save_path, 'PNG', optimize=True)
                        success = True
                    except Exception as e:
                        print(f"PIL保存PNG失败: {str(e)}")
                elif save_format.lower() == 'bmp':
                    # BMP格式
                    bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(save_path, bgr_image)
                elif save_format.lower() == 'tiff':
                    # TIFF格式
                    bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(save_path, bgr_image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                else:
                    # 默认使用JPEG
                    bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(save_path, bgr_image)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                if success:
                    saved_count += 1
                    print(f"成功保存图片: {save_path}")
                else:
                    print(f"保存图片失败: {save_path}")
                    # 尝试使用PIL作为备用方案
                    try:
                        if save_format.lower() == 'jpg':
                            Image.fromarray(face_image).save(save_path, 'JPEG', quality=95)
                        elif save_format.lower() == 'png':
                            Image.fromarray(face_image).save(save_path, 'PNG', optimize=True)
                        elif save_format.lower() == 'bmp':
                            Image.fromarray(face_image).save(save_path, 'BMP')
                        elif save_format.lower() == 'tiff':
                            Image.fromarray(face_image).save(save_path, 'TIFF', compression='tiff_lzw')
                        else:
                            Image.fromarray(face_image).save(save_path, quality=95)
                        saved_count += 1
                        print(f"使用PIL成功保存图片: {save_path}")
                    except Exception as pil_error:
                        print(f"PIL保存也失败: {str(pil_error)}")
                        # 最后尝试：使用绝对路径
                        try:
                            abs_save_path = os.path.abspath(save_path)
                            print(f"尝试使用绝对路径保存: {abs_save_path}")
                            if save_format.lower() == 'jpg':
                                Image.fromarray(face_image).save(abs_save_path, 'JPEG', quality=95)
                            elif save_format.lower() == 'png':
                                Image.fromarray(face_image).save(abs_save_path, 'PNG', optimize=True)
                            elif save_format.lower() == 'bmp':
                                Image.fromarray(face_image).save(abs_save_path, 'BMP')
                            elif save_format.lower() == 'tiff':
                                Image.fromarray(face_image).save(abs_save_path, 'TIFF', compression='tiff_lzw')
                            else:
                                Image.fromarray(face_image).save(abs_save_path, quality=95)
                            saved_count += 1
                            print(f"使用绝对路径成功保存图片: {abs_save_path}")
                        except Exception as abs_error:
                            print(f"绝对路径保存也失败: {str(abs_error)}")
            except Exception as e:
                print(f"保存第 {i+1} 张图片时出错: {str(e)}")
                continue
        
        if saved_count > 0:
            messagebox.showinfo("成功", f"已保存 {saved_count} 张人脸图片")
            # 更新已注册人名列表
            if name not in self.registered_names:
                self.registered_names.append(name)
                self.update_name_list()
        else:
            messagebox.showerror("错误", "没有成功保存任何图片")
        
        # 自动调用features_extraction_to_csv.py的功能
        self.refresh_features_csv(name)
    
    def delete_selected_name(self):
        """删除选中的人名"""
        selection = self.listbox_names.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的人名")
            return
        
        name = self.listbox_names.get(selection[0])
        if messagebox.askyesno("确认", f"确定要删除 {name} 的所有人脸数据吗？"):
            # 删除文件夹
            folder_path = os.path.join(self.path_photos_from_camera, f"person_{name}")
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            
            # 更新列表
            self.registered_names.remove(name)
            self.update_name_list()
            
            # 刷新特征文件
            self.refresh_features_csv()
            
            messagebox.showinfo("成功", f"已删除 {name} 的所有人脸数据并刷新特征文件")
    
    def run(self):
        """运行程序"""
        self.win.mainloop()

if __name__ == "__main__":
    collector = FaceCollector()
    collector.run() 