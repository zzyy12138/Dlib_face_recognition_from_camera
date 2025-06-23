import os
import logging
import tkinter as tk
from tkinter import ttk, Toplevel, Label
from datetime import datetime
from face_database_manager import FaceDatabaseManager
from tkinter import messagebox

logging.basicConfig(level=logging.INFO)

def main():
    root = tk.Tk()
    root.title("重点关注人员管理")
    root.geometry("800x600")
    root.attributes("-topmost", True)
    db_manager = FaceDatabaseManager()

    def refresh_list():
        for item in tree.get_children():
            tree.delete(item)
        important_persons = db_manager.get_important_persons()
        for person in important_persons:
            display_name = person.get('real_name') or person['name']
            display_id = person.get('real_id_card') or person.get('id_card') or '无'
            person_type = "临时" if person['is_temp'] else "真实"
            created_time = person.get('created_time', '未知')
            if created_time and created_time != '未知':
                try:
                    dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    formatted_time = created_time
            else:
                formatted_time = "未知"
            tree.insert('', 'end', values=(person['id'], display_name, display_id, person_type, formatted_time))

    def remove_important():
        selection = tree.selection()
        if not selection:
            tk.messagebox.showwarning("警告", "请先选择要取消重点关注的人员", parent=root)
            return
        selected_item = tree.item(selection[0])
        person_id = selected_item['values'][0]
        person_name = selected_item['values'][1]
        result = tk.messagebox.askyesno("确认操作", f"确定要取消人员 '{person_name}' 的重点关注状态吗？", parent=root)
        if result:
            success = db_manager.set_important_status(person_id, False)
            if success:
                logging.info(f"已在数据库中取消 {person_name} (ID: {person_id}) 的重点关注状态。")
                refresh_list()
                tk.messagebox.showinfo("成功", f"已取消 '{person_name}' 的重点关注状态。", parent=root)
            else:
                tk.messagebox.showerror("数据库错误", "更新数据库失败，无法取消重点关注状态。", parent=root)

    columns = ('ID', '姓名', '身份证号', '类型', '创建时间')
    tree = ttk.Treeview(root, columns=columns, show='headings', height=15)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=120)
    tree.pack(fill=tk.BOTH, expand=True)
    refresh_list()

    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, pady=(20, 0))
    tk.Button(button_frame, text="刷新列表", command=refresh_list).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="取消重点关注", command=remove_important).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="关闭", command=root.destroy).pack(side=tk.RIGHT, padx=5)

    root.mainloop()

if __name__ == '__main__':
    main() 