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
        # 获取所有重点关注人员，按real_id_card去重，只保留最新一条
        important_persons = db_manager.get_important_persons()
        unique_persons = {}
        for person in important_persons:
            key = person.get('real_id_card') or person.get('id_card')
            if key and key not in unique_persons:
                unique_persons[key] = person
            elif not key:
                # 没有身份证号的也显示
                unique_persons[person['id']] = person
        for person in unique_persons.values():
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
        display_id = selected_item['values'][2]
        # 获取当前人员信息
        person_info = db_manager.get_person_by_id(person_id)
        if not person_info:
            tk.messagebox.showerror("错误", "获取人员信息失败", parent=root)
            return
        real_id_card = person_info.get('real_id_card') or person_info.get('id_card')
        if not real_id_card:
            tk.messagebox.showerror("错误", "该人员无身份证号，无法批量取消重点关注", parent=root)
            return
        result = tk.messagebox.askyesno("确认操作", f"确定要取消所有身份证号为 '{real_id_card}' 的人员的重点关注状态吗？", parent=root)
        if result:
            affected = db_manager.set_important_status_by_real_id_card(real_id_card, False)
            if affected > 0:
                logging.info(f"已取消身份证号为{real_id_card}的所有人员的重点关注状态，共{affected}人。")
                refresh_list()
                tk.messagebox.showinfo("成功", f"已取消身份证号为 '{real_id_card}' 的所有人员的重点关注状态。", parent=root)
            else:
                tk.messagebox.showerror("数据库错误", f"取消失败，无匹配人员。", parent=root)

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