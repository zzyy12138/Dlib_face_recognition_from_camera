#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理临时身份脚本
用于清理数据库中所有临时身份（unknown1、unknown2等）
"""

import sqlite3
import os
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cleanup_temp_identities():
    """清理所有临时身份"""
    db_path = "data/face_database.db"
    
    if not os.path.exists(db_path):
        print("错误: 找不到数据库文件")
        return False
    
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取清理前的统计信息
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_temp = 1")
        temp_persons_before = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_temp = 0")
        real_persons_before = cursor.fetchone()[0]
        
        print(f"清理前统计:")
        print(f"  临时人员: {temp_persons_before} 个")
        print(f"  真实人员: {real_persons_before} 个")
        print()
        
        if temp_persons_before == 0:
            print("没有需要清理的临时身份")
            return True
        
        # 确认清理
        confirm = input(f"确定要删除 {temp_persons_before} 个临时身份吗？(y/N): ")
        if confirm.lower() != 'y':
            print("取消清理操作")
            return False
        
        # 开始清理
        print("开始清理临时身份...")
        
        # 获取所有临时人员的ID
        cursor.execute("SELECT id, name FROM persons WHERE is_temp = 1")
        temp_persons = cursor.fetchall()
        
        deleted_count = 0
        
        for person_id, person_name in temp_persons:
            try:
                # 删除相关的人脸特征
                cursor.execute("DELETE FROM face_features WHERE person_id = ?", (person_id,))
                
                # 删除相关的人脸图像
                cursor.execute("DELETE FROM face_images WHERE person_id = ?", (person_id,))
                
                # 删除人员记录
                cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
                
                deleted_count += 1
                print(f"已删除临时身份: {person_name} (ID: {person_id})")
                
            except Exception as e:
                print(f"删除临时身份 {person_name} 时出错: {str(e)}")
                continue
        
        # 提交更改
        conn.commit()
        
        # 获取清理后的统计信息
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_temp = 1")
        temp_persons_after = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_temp = 0")
        real_persons_after = cursor.fetchone()[0]
        
        print()
        print(f"清理完成!")
        print(f"  成功删除: {deleted_count} 个临时身份")
        print(f"  剩余临时人员: {temp_persons_after} 个")
        print(f"  真实人员: {real_persons_after} 个")
        
        # 关闭数据库连接
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"清理过程中出错: {str(e)}")
        logging.error(f"清理临时身份时出错: {str(e)}")
        return False

def show_database_info():
    """显示数据库信息"""
    db_path = "data/face_database.db"
    
    if not os.path.exists(db_path):
        print("错误: 找不到数据库文件")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取统计信息
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_temp = 1")
        temp_persons = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_temp = 0")
        real_persons = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM face_features")
        total_features = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM face_images")
        total_images = cursor.fetchone()[0]
        
        print("数据库信息:")
        print(f"  临时人员: {temp_persons} 个")
        print(f"  真实人员: {real_persons} 个")
        print(f"  总人员: {temp_persons + real_persons} 个")
        print(f"  人脸特征: {total_features} 个")
        print(f"  人脸图像: {total_images} 个")
        
        if temp_persons > 0:
            print()
            print("临时人员列表:")
            cursor.execute("SELECT id, name, id_card, created_time FROM persons WHERE is_temp = 1 ORDER BY created_time DESC")
            temp_list = cursor.fetchall()
            
            for person_id, name, id_card, created_time in temp_list:
                print(f"  ID: {person_id}, 姓名: {name}, 身份证: {id_card}, 创建时间: {created_time}")
        
        conn.close()
        
    except Exception as e:
        print(f"获取数据库信息时出错: {str(e)}")

def main():
    """主函数"""
    print("=" * 50)
    print("临时身份清理工具")
    print("=" * 50)
    print()
    
    while True:
        print("请选择操作:")
        print("1. 查看数据库信息")
        print("2. 清理所有临时身份")
        print("3. 退出")
        print()
        
        choice = input("请输入选项 (1-3): ").strip()
        
        if choice == '1':
            print()
            show_database_info()
            print()
        elif choice == '2':
            print()
            cleanup_temp_identities()
            print()
        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效选项，请重新选择")
            print()

if __name__ == '__main__':
    main() 