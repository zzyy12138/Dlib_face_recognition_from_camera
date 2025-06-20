import cv2
import dlib
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import shutil

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
    
    def select_image(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            print(f"选择的图片路径: {file_path}")
            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                print(f"无法读取图片，请检查文件是否存在且格式正确: {file_path}")
                messagebox.showerror("错误", f"无法读取图片文件: {file_path}\n请确保文件存在且格式正确")
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
        for i, face in enumerate(self.current_faces):
            try:
                # 提取人脸区域
                face_image = self.current_image[face.top():face.bottom(), face.left():face.right()]
                # 保存图片
                save_path = os.path.normpath(os.path.join(save_dir, f"img_face_{i+1}.jpg"))
                print(f"正在保存图片到: {save_path}")
                print(f"图片尺寸: {face_image.shape}")
                
                # 确保图片是uint8类型
                if face_image.dtype != np.uint8:
                    face_image = face_image.astype(np.uint8)
                
                # 转换为BGR格式并保存
                bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 尝试保存图片
                success = cv2.imwrite(save_path, bgr_image)
                
                if success:
                    saved_count += 1
                    print(f"成功保存图片: {save_path}")
                else:
                    print(f"保存图片失败: {save_path}")
                    # 尝试使用PIL保存
                    try:
                        Image.fromarray(face_image).save(save_path)
                        saved_count += 1
                        print(f"使用PIL成功保存图片: {save_path}")
                    except Exception as pil_error:
                        print(f"PIL保存也失败: {str(pil_error)}")
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
            messagebox.showinfo("成功", f"已删除 {name} 的所有人脸数据")
    
    def run(self):
        """运行程序"""
        self.win.mainloop()

if __name__ == "__main__":
    collector = FaceCollector()
    collector.run() 