import os
import logging
import tkinter as tk
from tkinter import Toplevel, Label
from face_database_manager import FaceDatabaseManager
from tkinter import messagebox

logging.basicConfig(level=logging.INFO)

def main():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.title("清空数据库")
    root.geometry("450x500")
    root.attributes("-topmost", True)
    db_manager = FaceDatabaseManager()

    def confirm_clear():
        confirm_text = confirm_var.get().strip()
        if confirm_text != "确认清空数据库":
            tk.messagebox.showerror("错误", "确认文字不正确！\n请输入：确认清空数据库")
            confirm_var.set("")
            confirm_entry.focus()
            return
        confirm_result = tk.messagebox.askyesno("最终确认", "确认文字正确！\n\n这是最后一次确认，确定要清空所有数据吗？\n\n此操作不可恢复！")
        if confirm_result:
            try:
                logging.info("用户确认清空数据库")
                success = db_manager.clear_database()
                if success:
                    logging.info("数据库清空成功")
                    tk.messagebox.showinfo("成功", "数据库已成功清空！\n\n所有数据已被删除。")
                else:
                    tk.messagebox.showerror("错误", "清空数据库失败！")
                root.destroy()
            except Exception as e:
                logging.error(f"清空数据库时出错: {str(e)}")
                tk.messagebox.showerror("错误", f"清空数据库时出错:\n{str(e)}")
                root.destroy()

    dialog = Toplevel(root)
    dialog.title("清空数据库")
    dialog.geometry("450x500")
    dialog.attributes("-topmost", True)
    dialog.grab_set()
    main_frame = tk.Frame(dialog)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    warning_label = Label(main_frame, text="⚠️ 危险操作警告", font=('Arial', 14, 'bold'), fg='red')
    warning_label.pack(pady=(0, 10))
    warning_text = """
此操作将清空所有数据：
• 所有人员信息
• 所有人脸特征
• 所有人脸图像
• 所有识别记录

此操作不可恢复！"""
    warning_info = Label(main_frame, text=warning_text, font=('Arial', 11), fg='red', justify='left')
    warning_info.pack(pady=(0, 20))
    confirm_frame = tk.Frame(main_frame)
    confirm_frame.pack(pady=(0, 20))
    confirm_label = Label(confirm_frame, text="请输入确认文字:", font=('Arial', 11))
    confirm_label.pack(side=tk.LEFT, padx=(0, 10))
    confirm_var = tk.StringVar()
    confirm_entry = tk.Entry(confirm_frame, textvariable=confirm_var, font=('Arial', 11), width=25)
    confirm_entry.pack(side=tk.LEFT)
    hint_label = Label(main_frame, text="请输入：确认清空数据库", font=('Arial', 10), fg='blue')
    hint_label.pack(pady=(0, 20))
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=(0, 10))
    confirm_btn = tk.Button(button_frame, text="确认清空", command=confirm_clear, font=('Arial', 11), width=12, bg='red', fg='white')
    confirm_btn.pack(side=tk.LEFT, padx=5)
    cancel_btn = tk.Button(button_frame, text="取消", command=root.destroy, font=('Arial', 11), width=12, bg='gray', fg='white')
    cancel_btn.pack(side=tk.LEFT, padx=5)
    confirm_entry.bind('<Return>', lambda e: confirm_clear())
    confirm_entry.focus()
    root.mainloop()

if __name__ == '__main__':
    main() 