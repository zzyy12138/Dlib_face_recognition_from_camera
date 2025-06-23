"""
独立的人脸库管理器
功能：
1. 显示数据库中所有人员信息
2. 预览人脸图片
3. 删除人员记录
4. 刷新数据
5. 统计信息显示
"""

import tkinter as tk
from tkinter import ttk, Label, messagebox
import sqlite3
import os
import logging
from datetime import datetime
from PIL import Image, ImageTk
import io

# 导入数据库管理器
from face_database_manager import FaceDatabaseManager

class FaceLibraryManager:
    def __init__(self):
        """初始化人脸库管理器"""
        # 设置日志
        self.setup_logging()
        
        # 初始化数据库管理器
        self.db_manager = FaceDatabaseManager()
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("人脸库管理器")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # 创建界面
        self.create_interface()
        
        # 加载数据
        self.load_person_data()
        
        logging.info("人脸库管理器初始化完成")
    
    def setup_logging(self):
        """设置日志配置"""
        # 创建logs目录
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/face_library_manager.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def create_interface(self):
        """创建用户界面"""
        # 主框架
        main_frame = tk.Frame(self.root)
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
        self.tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=20)
        
        # 设置列标题和宽度
        column_widths = {
            'ID': 60,
            '姓名': 120,
            '身份证号': 150,
            '类型': 80,
            '创建时间': 120
        }
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 100))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
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
        
        self.image_label = Label(image_frame, text="请选择人员查看图片", font=('Arial', 10))
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(detail_frame, text="人员信息")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, font=('Arial', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 按钮区域
        button_frame = ttk.Frame(detail_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 按钮
        ttk.Button(button_frame, text="刷新", command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="删除选中", command=self.delete_selected_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="设置重点关注", command=self.toggle_important_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="导出数据", command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=self.close_window).pack(side=tk.RIGHT, padx=5)
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = Label(status_frame, text="正在加载数据...", font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)
        
        # 绑定选择事件
        self.tree.bind('<<TreeviewSelect>>', self.on_person_select)
        
        # 绑定键盘快捷键
        self.root.bind('<F5>', lambda e: self.refresh_data())
        self.root.bind('<Delete>', lambda e: self.delete_selected_person())
        self.root.bind('<Escape>', lambda e: self.close_window())
    
    def load_person_data(self):
        """加载人员数据到列表"""
        try:
            # 清空现有数据
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 从数据库获取所有人员
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, id_card, is_temp, is_important, created_time, real_name, real_id_card
                FROM persons
                ORDER BY created_time DESC
            ''')
            
            for row in cursor.fetchall():
                person_id, name, id_card, is_temp, is_important, created_time, real_name, real_id_card = row
                
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
                if is_important:
                    person_type += " ⭐"  # 添加星号标记重点关注人员
                
                # 格式化时间
                if created_time:
                    try:
                        dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_time = created_time
                else:
                    formatted_time = "未知"
                
                self.tree.insert('', 'end', values=(person_id, display_name, display_id, person_type, formatted_time))
            
            conn.close()
            
            # 更新统计信息
            stats = self.db_manager.get_statistics()
            important_count = len(self.db_manager.get_important_persons())
            status_text = f"总人员: {stats['total_persons']} | 真实身份: {stats['real_persons']} | 临时身份: {stats['temp_persons']} | 重点关注: {important_count}"
            self.status_label.config(text=status_text)
            
            logging.info(f"已加载 {stats['total_persons']} 个人员数据")
            
        except Exception as e:
            logging.error(f"加载人员数据失败: {str(e)}")
            messagebox.showerror("错误", f"加载人员数据失败:\n{str(e)}")
    
    def on_person_select(self, event):
        """当选择人员时显示详细信息"""
        selection = self.tree.selection()
        if not selection:
            return
        
        try:
            # 获取选中的人员ID
            person_id = self.tree.item(selection[0])['values'][0]
            
            # 获取人员详细信息
            person_info = self.db_manager.get_person_by_id(person_id)
            if not person_info:
                return
            
            # 显示人员信息
            self.info_text.delete(1.0, tk.END)
            info_content = f"""人员ID: {person_info['id']}
姓名: {person_info['name']}
身份证号: {person_info['id_card'] or '无'}
真实姓名: {person_info['real_name'] or '无'}
真实身份证号: {person_info['real_id_card'] or '无'}
身份类型: {'临时身份' if person_info['is_temp'] else '真实身份'}
重点关注: {'是 ⭐' if person_info['is_important'] else '否'}
创建时间: {person_info['created_time']}
更新时间: {person_info['updated_time']}"""
            
            self.info_text.insert(1.0, info_content)
            
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
                    self.image_label.config(image=tk_img, text="")
                    self.image_label.image = tk_img  # 保持引用
                    
                except Exception as e:
                    self.image_label.config(image="", text=f"图片加载失败: {str(e)}")
            else:
                self.image_label.config(image="", text="无图片数据")
            
        except Exception as e:
            logging.error(f"显示人员详细信息失败: {str(e)}")
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, f"获取详细信息失败: {str(e)}")
    
    def delete_selected_person(self):
        """删除选中的人员"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的人员")
            return
        
        person_id = self.tree.item(selection[0])['values'][0]
        person_name = self.tree.item(selection[0])['values'][1]
        
        # 确认删除
        result = messagebox.askyesno("确认删除", f"确定要删除人员 '{person_name}' 吗？\n此操作不可恢复！")
        
        if result:
            try:
                if self.db_manager.delete_person(person_id):
                    messagebox.showinfo("成功", f"已删除人员 '{person_name}'")
                    self.load_person_data()  # 重新加载数据
                else:
                    messagebox.showerror("错误", "删除失败")
            except Exception as e:
                messagebox.showerror("错误", f"删除失败: {str(e)}")
    
    def toggle_important_status(self):
        """切换选中人员的重点关注状态"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要设置的人员")
            return
        
        person_id = self.tree.item(selection[0])['values'][0]
        person_name = self.tree.item(selection[0])['values'][1]
        
        # 获取当前重点关注状态
        person_info = self.db_manager.get_person_by_id(person_id)
        if not person_info:
            messagebox.showerror("错误", "获取人员信息失败")
            return
        
        current_status = person_info['is_important']
        new_status = not current_status
        
        # 确认操作
        action = "设置为重点关注" if new_status else "取消重点关注"
        result = messagebox.askyesno("确认操作", f"确定要{action}人员 '{person_name}' 吗？")
        
        if result:
            try:
                if self.db_manager.set_important_status(person_id, new_status):
                    status_text = "重点关注 ⭐" if new_status else "普通人员"
                    messagebox.showinfo("成功", f"已{action}人员 '{person_name}'\n当前状态: {status_text}")
                    self.load_person_data()  # 重新加载数据
                    self.on_person_select(None)  # 刷新详细信息显示
                else:
                    messagebox.showerror("错误", f"{action}失败")
            except Exception as e:
                messagebox.showerror("错误", f"{action}失败: {str(e)}")
    
    def refresh_data(self):
        """刷新数据"""
        self.load_person_data()
        messagebox.showinfo("刷新", "数据已刷新")
    
    def export_data(self):
        """导出数据到CSV文件"""
        try:
            from tkinter import filedialog
            import csv
            
            # 选择保存路径
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="导出人员数据"
            )
            
            if not filename:
                return
            
            # 获取所有人员数据
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, id_card, is_temp, created_time, real_name, real_id_card
                FROM persons
                ORDER BY created_time DESC
            ''')
            
            data = cursor.fetchall()
            conn.close()
            
            # 写入CSV文件
            with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入表头
                writer.writerow(['ID', '姓名', '身份证号', '类型', '创建时间', '真实姓名', '真实身份证号'])
                
                # 写入数据
                for row in data:
                    person_id, name, id_card, is_temp, created_time, real_name, real_id_card = row
                    person_type = "临时" if is_temp else "真实"
                    writer.writerow([person_id, name, id_card, person_type, created_time, real_name, real_id_card])
            
            messagebox.showinfo("导出成功", f"数据已导出到:\n{filename}")
            logging.info(f"数据已导出到: {filename}")
            
        except Exception as e:
            logging.error(f"导出数据失败: {str(e)}")
            messagebox.showerror("错误", f"导出数据失败:\n{str(e)}")
    
    def close_window(self):
        """关闭窗口"""
        try:
            # 关闭数据库连接
            if hasattr(self, 'db_manager'):
                self.db_manager.close()
            
            # 检查窗口是否还存在
            if hasattr(self, 'root'):
                try:
                    if self.root.winfo_exists():
                        self.root.destroy()
                except tk.TclError:
                    # 窗口已经被销毁，忽略错误
                    pass
            
            logging.info("人脸库管理器已关闭")
            
        except Exception as e:
            logging.error(f"关闭窗口时出错: {str(e)}")
            # 如果出错，尝试强制关闭
            try:
                if hasattr(self, 'root'):
                    try:
                        self.root.quit()
                    except tk.TclError:
                        # 窗口已经被销毁，忽略错误
                        pass
            except:
                pass
    
    def run(self):
        """运行程序"""
        try:
            logging.info("启动人脸库管理器")
            self.root.mainloop()
        except Exception as e:
            logging.error(f"程序运行出错: {str(e)}")
            raise
        finally:
            # 确保在程序结束时正确关闭
            try:
                # 只在窗口还存在时才调用close_window
                if hasattr(self, 'root'):
                    try:
                        if self.root.winfo_exists():
                            self.close_window()
                    except tk.TclError:
                        # 窗口已经被销毁，只关闭数据库连接
                        if hasattr(self, 'db_manager'):
                            self.db_manager.close()
            except:
                pass

def main():
    """主函数"""
    try:
        # 检查数据库文件是否存在
        if not os.path.exists('data/face_database.db'):
            messagebox.showerror("错误", "找不到数据库文件！\n请确保 data/face_database.db 文件存在。")
            return
        
        # 创建并运行管理器
        manager = FaceLibraryManager()
        manager.run()
        
    except Exception as e:
        logging.error(f"程序启动失败: {str(e)}")
        messagebox.showerror("错误", f"程序启动失败:\n{str(e)}")

if __name__ == '__main__':
    main() 