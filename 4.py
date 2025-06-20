"""
基于Dlib的人脸识别监控程序
功能：
1. 实时捕获屏幕画面并进行人脸检测
2. 对检测到的人脸进行识别和跟踪
3. 支持透明窗口显示，不影响其他操作
4. 系统托盘图标控制
5. 自动保存未知人脸数据，弹出窗口输入名称
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
import pystray
from tkinter import Canvas, Toplevel, Label, simpledialog
import random
import string
from datetime import datetime
import threading

# 配置日志记录
logging.basicConfig(level=logging.INFO)

# 加载Dlib预训练模型
cnn_face_detector = dlib.cnn_face_detection_model_v1('data/data_dlib/mmod_human_face_detector.dat')
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class TransparentFaceRecognizer:
    def __init__(self):
        self.face_feature_known_list = []
        self.face_name_known_list = []
        self.face_image_path_list = []
        self.current_frame_face_feature_list = []
        self.current_frame_face_cnt = 0
        self.current_frame_face_name_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_known_list = []
        self.fps_show = 0
        self.frame_cnt = 0
        self.start_time = time.time()
        self.sct = mss.mss()
        self.screen_width = pyautogui.size().width
        self.screen_height = pyautogui.size().height
        self.monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)
        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.7)
        self.root.attributes("-transparentcolor", "white")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.canvas = Canvas(self.root, bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.set_window_clickthrough()
        self.get_face_database()
        self.setup_exit_controls()
        self.create_system_tray_icon()
        self.shown_faces = set()
        self.show_popup = False

    def create_system_tray_icon(self):
        image = Image.new('RGBA', (16, 16), (0, 120, 212, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((2, 2, 14, 14), fill=(255, 255, 255, 255))
        self.toggle_popup_item = pystray.MenuItem(
            lambda item: f"{'关闭' if self.show_popup else '开启'}弹窗显示",
            self.toggle_popup
        )
        menu = pystray.Menu(
            self.toggle_popup_item,
            pystray.MenuItem('退出', self.quit_program)
        )
        self.tray_icon = pystray.Icon("人脸识别监控", image, "人脸识别监控程序", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def toggle_popup(self, icon=None, item=None):
        self.show_popup = not self.show_popup
        if icon:
            icon.update_menu()
        logging.info(f"弹窗显示已 {'开启' if self.show_popup else '关闭'}")

    def setup_exit_controls(self):
        self.root.bind('<Escape>', lambda e: self.quit_program())
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="退出", command=self.quit_program)
        self.root.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))

    def quit_program(self):
        self.root.quit()
        self.root.destroy()
        self.sct.close()
        cv2.destroyAllWindows()
        logging.info("程序已退出")

    def set_window_clickthrough(self):
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        style = ctypes.windll.user32.GetWindowLongA(hwnd, -20)
        ctypes.windll.user32.SetWindowLongA(hwnd, -20, style | 0x00000020 | 0x80000)

    def get_face_database(self):
        path = "data/features_all.csv"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path, header=None)
            if not df.empty:
                for i in range(df.shape[0]):
                    name = df.iloc[i][0]
                    features = [float(x) if x != '' else 0.0 for x in df.iloc[i][1:129]]
                    self.face_name_known_list.append(name)
                    self.face_feature_known_list.append(features)
                    self.face_image_path_list.append(f"data/data_faces_from_camera/person_{name}/img_face_1.jpg")
                logging.info(f"已加载 {len(self.face_feature_known_list)} 张人脸")
            else:
                logging.info("特征文件为空，没有加载任何人脸")
        else:
            logging.info("特征文件不存在或为空，跳过加载")


    @staticmethod
    def return_euclidean_distance(f1, f2):
        return np.linalg.norm(np.array(f1) - np.array(f2))

    def update_fps(self):
        now = time.time()
        self.frame_cnt += 1
        if now - self.start_time >= 1:
            self.fps_show = self.frame_cnt / (now - self.start_time)
            self.start_time = now
            self.frame_cnt = 0

    def get_screen(self):
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img

    def show_face_info(self, name, idx):
        if not self.show_popup or name in self.shown_faces:
            return
        self.shown_faces.add(name)

        def show():
            popup = Toplevel(self.root)
            popup.title(f"识别到: {name}")
            popup.attributes("-topmost", True)
            img_path = self.face_image_path_list[idx] if idx < len(self.face_image_path_list) else ""
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).resize((200, 200))
                    img = ImageTk.PhotoImage(img)
                    label_img = Label(popup, image=img)
                    label_img.image = img
                    label_img.pack(pady=10)
                except:
                    Label(popup, text="图片加载失败").pack(pady=10)
            else:
                Label(popup, text="无图片").pack(pady=10)
            info = f"姓名: {name}\n识别时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            Label(popup, text=info, font=('Arial', 12)).pack(pady=10)
            def close():
                self.shown_faces.discard(name)
                popup.destroy()
            tk.Button(popup, text="关闭", command=close).pack(pady=10)
            popup.after(5000, close)

        threading.Thread(target=show).start()

    def create_new_face_data(self, img, face_rect, shape, feature):
        def ask_name_with_preview():
            popup = Toplevel(self.root)
            popup.title("发现新的人脸")
            popup.geometry("300x360")
            popup.attributes("-topmost", True)
            popup.grab_set()

            # 截取人脸图像（安全边界）
            top = max(0, face_rect.top())
            bottom = min(img.shape[0], face_rect.bottom())
            left = max(0, face_rect.left())
            right = min(img.shape[1], face_rect.right())
            face_img = img[top:bottom, left:right]

            # 将OpenCV图像转换为Tk可显示的格式
            try:
                pil_img = Image.fromarray(face_img).resize((200, 200))
                tk_img = ImageTk.PhotoImage(pil_img)
                img_label = Label(popup, image=tk_img)
                img_label.image = tk_img
                img_label.pack(pady=5)
            except Exception as e:
                Label(popup, text="图像显示失败").pack()

            Label(popup, text="请输入此人姓名：").pack(pady=5)
            name_entry = tk.Entry(popup)
            name_entry.pack(pady=5)

            def on_confirm():
                name = name_entry.get().strip() or "Unknown"
                folder = f"data/data_faces_from_camera/person_{name}"
                os.makedirs(folder, exist_ok=True)
                existing_imgs = [f for f in os.listdir(folder) if f.startswith("img_face") and f.endswith(".jpg")]
                img_index = len(existing_imgs) + 1
                img_filename = os.path.join(folder, f"img_face_{img_index}.jpg")
                cv2.imwrite(img_filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                self.face_name_known_list.append(name)
                self.face_feature_known_list.append(feature)
                self.face_image_path_list.append(img_filename)
                self.update_face_database_csv(name, feature)
                logging.info(f"新增未知人脸: {name}，图像保存为 {img_filename}")
                popup.destroy()

            tk.Button(popup, text="确认", command=on_confirm).pack(pady=10)
            popup.bind('<Return>', lambda e: on_confirm())

        self.root.after(0, ask_name_with_preview)


    def update_face_database_csv(self, name, feature):
        with open("data/features_all.csv", 'a', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow([name] + list(feature))

    def process_frame(self):
        self.canvas.delete("all")
        img = self.get_screen()
        scale = 0.5
        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        faces = cnn_face_detector(img_small, 0)
        self.current_frame_face_feature_list.clear()
        self.current_frame_face_position_list.clear()
        self.current_frame_face_name_list.clear()
        self.current_frame_face_known_list.clear()
        self.current_frame_face_cnt = len(faces)
        for face in faces:
            rect = face.rect
            rect = dlib.rectangle(int(rect.left() / scale), int(rect.top() / scale),
                                  int(rect.right() / scale), int(rect.bottom() / scale))
            shape = predictor(img, rect)
            feature = face_reco_model.compute_face_descriptor(img, shape)
            name = "Unknown"
            known = False
            if self.face_feature_known_list:
                distances = [self.return_euclidean_distance(feature, f) for f in self.face_feature_known_list]
                min_dist = min(distances)
                if min_dist < 0.6:
                    idx = distances.index(min_dist)
                    name = self.face_name_known_list[idx]
                    known = True
                    if name not in self.shown_faces:
                        self.root.after(100, lambda n=name, i=idx: self.show_face_info(n, i))
                else:
                    self.create_new_face_data(img, rect, shape, feature)
            self.current_frame_face_feature_list.append(feature)
            self.current_frame_face_position_list.append((rect.left(), rect.top(), rect.right(), rect.bottom()))
            self.current_frame_face_name_list.append(name)
            self.current_frame_face_known_list.append(known)
        self.draw_results()
        self.update_fps()
        self.root.after(50, self.process_frame)

    def draw_results(self):
        for i, (left, top, right, bottom) in enumerate(self.current_frame_face_position_list):
            color = 'red' if self.current_frame_face_known_list[i] else 'cyan'
            self.canvas.create_rectangle(left, top, right, bottom, outline=color, width=2)
            self.canvas.create_text(left, bottom + 20, text=self.current_frame_face_name_list[i],
                                    fill='yellow', font=('Arial', 12, 'bold'), anchor='nw')
        info = f"Faces: {self.current_frame_face_cnt} | FPS: {self.fps_show:.1f}"
        self.canvas.create_text(20, 20, text=info, fill='lime', font=('Arial', 12, 'bold'), anchor='nw')

    def run(self):
        self.process_frame()
        self.root.mainloop()

def main():
    recognizer = TransparentFaceRecognizer()
    recognizer.run()

if __name__ == '__main__':
    main()