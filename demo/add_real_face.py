#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速添加真实身份人脸脚本
用于手动添加已知人员的真实身份信息
"""

import sqlite3
import os
import logging
from face_database_manager import FaceDatabaseManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def add_real_person():
    """添加真实身份人员"""
    print("=" * 50)
    print("添加真实身份人员")
    print("=" * 50)
    
    try:
        # 获取用户输入
        name = input("请输入姓名: ").strip()
        if not name:
            print("姓名不能为空")
            return False
        
        id_card = input("请输入身份证号 (可选): ").strip()
        if not id_card:
            id_card = None
        
        # 创建数据库管理器
        db_manager = FaceDatabaseManager()
        
        # 检查是否已存在
        existing_person = db_manager.get_person_by_name_id(name, id_card)
        if existing_person:
            print(f"人员已存在: {existing_person['name']} - {existing_person['id_card']}")
            update = input("是否更新现有记录? (y/N): ").strip().lower()
            if update != 'y':
                return False
        
        # 添加人员
        person_id = db_manager.add_person(name, id_card, is_temp=False)
        print(f"成功添加人员: {name} (ID: {person_id})")
        
        # 提示用户添加人脸图像和特征
        print("\n注意: 您需要手动添加人脸图像和特征")
        print("可以使用以下方法:")
        print("1. 运行人脸采集工具: python face_collector_from_image.py")
        print("2. 使用系统托盘菜单中的'手动添加人脸'选项")
        print("3. 直接编辑数据库添加特征数据")
        
        return True
        
    except Exception as e:
        print(f"添加人员时出错: {str(e)}")
        logging.error(f"添加人员时出错: {str(e)}")
        return False

def list_real_persons():
    """列出所有真实身份人员"""
    print("=" * 50)
    print("真实身份人员列表")
    print("=" * 50)
    
    try:
        db_manager = FaceDatabaseManager()
        
        # 获取统计信息
        stats = db_manager.get_statistics()
        print(f"总人员: {stats['total_persons']} 个")
        print(f"真实身份: {stats['real_persons']} 个")
        print(f"临时身份: {stats['temp_persons']} 个")
        print()
        
        if stats['real_persons'] == 0:
            print("没有找到任何真实身份人员")
            return
        
        # 连接数据库获取详细信息
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, id_card, created_time, real_name, real_id_card
            FROM persons
            WHERE is_temp = 0
            ORDER BY created_time DESC
        ''')
        
        persons = cursor.fetchall()
        
        print("真实身份人员列表:")
        for i, (person_id, name, id_card, created_time, real_name, real_id_card) in enumerate(persons, 1):
            display_name = real_name if real_name else name
            display_id = real_id_card if real_id_card else id_card
            
            print(f"{i}. ID: {person_id}")
            print(f"   姓名: {display_name}")
            print(f"   身份证: {display_id or '无'}")
            print(f"   创建时间: {created_time}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"获取人员列表时出错: {str(e)}")
        logging.error(f"获取人员列表时出错: {str(e)}")

def delete_person():
    """删除人员"""
    print("=" * 50)
    print("删除人员")
    print("=" * 50)
    
    try:
        # 先列出所有人员
        list_real_persons()
        
        # 获取用户输入
        person_id = input("请输入要删除的人员ID: ").strip()
        if not person_id:
            print("人员ID不能为空")
            return False
        
        try:
            person_id = int(person_id)
        except ValueError:
            print("人员ID必须是数字")
            return False
        
        # 确认删除
        confirm = input("确定要删除这个人员吗? (y/N): ").strip().lower()
        if confirm != 'y':
            print("取消删除")
            return False
        
        # 删除人员
        db_manager = FaceDatabaseManager()
        if db_manager.delete_person(person_id):
            print(f"成功删除人员 ID: {person_id}")
            return True
        else:
            print("删除失败")
            return False
        
    except Exception as e:
        print(f"删除人员时出错: {str(e)}")
        logging.error(f"删除人员时出错: {str(e)}")
        return False

def show_help():
    """显示帮助信息"""
    print("=" * 50)
    print("帮助信息")
    print("=" * 50)
    
    print("1. 添加真实身份人员:")
    print("   - 选择选项 1")
    print("   - 输入姓名和身份证号")
    print("   - 然后使用人脸采集工具添加人脸图像和特征")
    
    print("\n2. 查看人员列表:")
    print("   - 选择选项 2")
    print("   - 查看所有已添加的真实身份人员")
    
    print("\n3. 删除人员:")
    print("   - 选择选项 3")
    print("   - 输入要删除的人员ID")
    
    print("\n4. 添加人脸图像和特征:")
    print("   - 运行: python face_collector_from_image.py")
    print("   - 或使用系统托盘菜单中的'手动添加人脸'选项")
    
    print("\n5. 清理临时身份:")
    print("   - 运行: python cleanup_temp_identities.py")
    print("   - 或使用系统托盘菜单中的'清理所有临时身份'选项")

def main():
    """主函数"""
    print("真实身份人员管理工具")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 添加真实身份人员")
        print("2. 查看人员列表")
        print("3. 删除人员")
        print("4. 显示帮助")
        print("5. 退出")
        print()
        
        choice = input("请输入选项 (1-5): ").strip()
        
        if choice == '1':
            print()
            add_real_person()
        elif choice == '2':
            print()
            list_real_persons()
        elif choice == '3':
            print()
            delete_person()
        elif choice == '4':
            print()
            show_help()
        elif choice == '5':
            print("退出程序")
            break
        else:
            print("无效选项，请重新选择")
            print()

if __name__ == '__main__':
    main() 