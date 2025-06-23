"""
人脸识别监控系统启动脚本
自动启动API服务器和监控系统
"""

import subprocess
import sys
import time
import threading
import os
import signal
import requests

def check_api_server():
    """检查API服务器是否运行"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """启动API服务器"""
    print("正在启动API服务器...")
    try:
        # 启动API服务器
        api_process = subprocess.Popen([
            sys.executable, "face_recognition_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务器启动
        for i in range(30):  # 最多等待30秒
            if check_api_server():
                print("✓ API服务器启动成功")
                return api_process
            time.sleep(1)
            print(f"等待API服务器启动... ({i+1}/30)")
        
        print("✗ API服务器启动超时")
        api_process.terminate()
        return None
        
    except Exception as e:
        print(f"✗ 启动API服务器失败: {str(e)}")
        return None

def start_monitor_system():
    """启动监控系统"""
    print("正在启动人脸识别监控系统...")
    try:
        # 启动监控系统
        monitor_process = subprocess.Popen([
            sys.executable, "screen_face_monitor.py"
        ])
        
        print("✓ 人脸识别监控系统启动成功")
        return monitor_process
        
    except Exception as e:
        print(f"✗ 启动监控系统失败: {str(e)}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("人脸识别监控系统启动器")
    print("=" * 60)
    
    # 检查必要文件
    required_files = [
        "face_recognition_api.py",
        "screen_face_monitor.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("✗ 缺少必要文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保所有文件都在当前目录中")
        return
    
    print("✓ 所有必要文件检查通过")
    
    # 检查依赖
    try:
        import requests
        import flask
        import flask_cors
        print("✓ 依赖检查通过")
    except ImportError as e:
        print(f"✗ 缺少依赖: {str(e)}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 启动API服务器
    api_process = start_api_server()
    if not api_process:
        print("无法启动API服务器，程序退出")
        return
    
    # 等待一下确保API服务器完全启动
    time.sleep(2)
    
    # 启动监控系统
    monitor_process = start_monitor_system()
    if not monitor_process:
        print("无法启动监控系统，程序退出")
        api_process.terminate()
        return
    
    print("\n" + "=" * 60)
    print("系统启动完成!")
    print("=" * 60)
    print("API服务器: http://localhost:5000")
    print("监控系统: 已启动")
    print("\n按 Ctrl+C 停止所有服务")
    print("=" * 60)
    
    try:
        # 等待进程结束
        while True:
            # 检查进程是否还在运行
            if api_process.poll() is not None:
                print("API服务器已停止")
                break
            if monitor_process.poll() is not None:
                print("监控系统已停止")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止所有服务...")
        
        # 停止API服务器
        if api_process and api_process.poll() is None:
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
                print("✓ API服务器已停止")
            except subprocess.TimeoutExpired:
                api_process.kill()
                print("✓ API服务器已强制停止")
        
        # 停止监控系统
        if monitor_process and monitor_process.poll() is None:
            monitor_process.terminate()
            try:
                monitor_process.wait(timeout=5)
                print("✓ 监控系统已停止")
            except subprocess.TimeoutExpired:
                monitor_process.kill()
                print("✓ 监控系统已强制停止")
        
        print("所有服务已停止")

if __name__ == '__main__':
    main() 