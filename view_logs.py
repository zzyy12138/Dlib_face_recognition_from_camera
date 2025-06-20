"""
日志查看工具
"""
import os
import glob
from datetime import datetime
from datetime import timedelta

def list_log_files():
    """列出所有日志文件"""
    log_files = glob.glob('logs/face_monitor_*.log')
    log_files.sort(reverse=True)  # 按时间倒序排列
    
    print("可用的日志文件:")
    print("-" * 50)
    for i, log_file in enumerate(log_files, 1):
        file_size = os.path.getsize(log_file)
        file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        # 从文件名提取日期
        filename = os.path.basename(log_file)
        date_str = filename.replace('face_monitor_', '').replace('.log', '')
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            date_display = date_obj.strftime('%Y年%m月%d日')
        except:
            date_display = date_str
            
        print(f"{i}. {filename}")
        print(f"   日期: {date_display}")
        print(f"   大小: {file_size} 字节")
        print(f"   修改时间: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    return log_files

def view_log_file(log_file, lines=50):
    """查看指定日志文件的最后几行"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            print(f"日志文件: {os.path.basename(log_file)}")
            print(f"总行数: {len(all_lines)}")
            print(f"显示最后 {len(last_lines)} 行:")
            print("-" * 50)
            
            for line in last_lines:
                print(line.rstrip())
                
    except Exception as e:
        print(f"读取日志文件时出错: {e}")

def clean_old_logs(days_to_keep=30):
    """清理指定天数之前的日志文件"""
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    log_files = glob.glob('logs/face_monitor_*.log')
    
    deleted_count = 0
    for log_file in log_files:
        filename = os.path.basename(log_file)
        date_str = filename.replace('face_monitor_', '').replace('.log', '')
        
        try:
            file_date = datetime.strptime(date_str, '%Y%m%d')
            if file_date < cutoff_date:
                os.remove(log_file)
                deleted_count += 1
                print(f"已删除: {filename}")
        except:
            print(f"无法解析日期，跳过: {filename}")
    
    print(f"清理完成，共删除 {deleted_count} 个旧日志文件")

def main():
    print("人脸识别监控系统 - 日志查看工具")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 查看日志文件列表")
        print("2. 查看指定日志文件")
        print("3. 清理旧日志文件")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            log_files = list_log_files()
            if not log_files:
                print("没有找到日志文件")
        
        elif choice == '2':
            log_files = list_log_files()
            if not log_files:
                print("没有找到日志文件")
                continue
                
            try:
                file_choice = input(f"请选择要查看的日志文件 (1-{len(log_files)}) 或按回车查看最新的: ").strip()
                
                if not file_choice:
                    selected_file = log_files[0]  # 最新的日志文件
                else:
                    index = int(file_choice) - 1
                    if 0 <= index < len(log_files):
                        selected_file = log_files[index]
                    else:
                        print("无效的选择")
                        continue
                
                lines = input("显示最后几行? (默认50): ").strip()
                lines = int(lines) if lines.isdigit() else 50
                
                view_log_file(selected_file, lines)
                
            except KeyboardInterrupt:
                print("\n操作已取消")
            except Exception as e:
                print(f"操作出错: {e}")
        
        elif choice == '3':
            try:
                days = input("保留最近几天的日志? (默认30天): ").strip()
                days = int(days) if days.isdigit() else 30
                
                confirm = input(f"确定要删除 {days} 天之前的日志文件吗? (y/N): ").strip().lower()
                if confirm == 'y':
                    clean_old_logs(days)
                else:
                    print("操作已取消")
            except KeyboardInterrupt:
                print("\n操作已取消")
            except Exception as e:
                print(f"操作出错: {e}")
        
        elif choice == '4':
            print("退出日志查看工具")
            break
        
        else:
            print("无效的选择，请重新输入")

if __name__ == '__main__':
    main() 