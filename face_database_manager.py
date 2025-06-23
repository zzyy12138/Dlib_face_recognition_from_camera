"""
人脸数据库管理器 - SQLite版本
用于管理人脸图像、特征向量和身份信息的SQLite数据库
"""

import sqlite3
import os
import cv2
import numpy as np
import logging
import json
import base64
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import threading

class FaceDatabaseManager:
    """人脸数据库管理器 - 使用SQLite存储"""
    
    def __init__(self, db_path: str = "data/face_database.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.lock = threading.Lock()  # 线程锁，确保数据库操作的线程安全
        
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        logging.info(f"人脸数据库管理器初始化完成，数据库路径: {db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # 创建人员信息表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS persons (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        id_card TEXT,
                        real_name TEXT,
                        real_id_card TEXT,
                        is_temp BOOLEAN DEFAULT 0,
                        is_important BOOLEAN DEFAULT 0,
                        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, id_card)
                    )
                ''')
                
                # 创建人脸图像表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER NOT NULL,
                        image_data BLOB NOT NULL,
                        image_format TEXT DEFAULT 'jpg',
                        image_size INTEGER,
                        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
                    )
                ''')
                
                # 创建人脸特征表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER NOT NULL,
                        feature_vector TEXT NOT NULL,  -- JSON格式存储128维特征向量
                        feature_hash TEXT UNIQUE,      -- 特征向量的哈希值，用于快速查找
                        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
                    )
                ''')
                
                # 创建识别记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER,
                        confidence REAL,
                        distance REAL,
                        frame_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE SET NULL
                    )
                ''')
                
                # 创建索引以提高查询性能
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_persons_id_card ON persons(id_card)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_persons_is_temp ON persons(is_temp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_features_hash ON face_features(feature_hash)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_recognition_logs_time ON recognition_logs(frame_time)')
                
                conn.commit()
                logging.info("数据库表结构初始化完成")
                
            except Exception as e:
                logging.error(f"初始化数据库时出错: {str(e)}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def add_person(self, name: str, id_card: str = None, is_temp: bool = False, 
                   real_name: str = None, real_id_card: str = None, is_important: bool = False) -> int:
        """
        添加新人员
        
        Args:
            name: 人员姓名
            id_card: 身份证号
            is_temp: 是否为临时身份
            real_name: 真实姓名（用于API识别后更新）
            real_id_card: 真实身份证号
            is_important: 是否为重点关注人员
            
        Returns:
            人员ID
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO persons (name, id_card, real_name, real_id_card, is_temp, is_important, updated_time)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (name, id_card, real_name, real_id_card, is_temp, is_important))
                
                person_id = cursor.lastrowid
                conn.commit()
                
                logging.info(f"添加人员成功: {name} (ID: {person_id})")
                return person_id
                
            except Exception as e:
                logging.error(f"添加人员失败: {str(e)}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def add_face_image(self, person_id: int, image_data: any,
                      image_format: str = 'jpg') -> int:
        """
        添加人脸图像，智能处理传入的数据类型。

        Args:
            person_id: 人员ID
            image_data: 图像数据，可以是文件路径(str)或二进制数据(bytes)
            image_format: 图像格式

        Returns:
            图像ID
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                img_bytes = None
                # 判断传入的是文件路径还是二进制数据
                if isinstance(image_data, str) and os.path.exists(image_data):
                    # 如果是有效的文件路径，读取文件内容
                    logging.info(f"从文件路径加载图像: {image_data}")
                    with open(image_data, 'rb') as f:
                        img_bytes = f.read()
                elif isinstance(image_data, bytes):
                    # 如果已经是二进制数据，直接使用
                    img_bytes = image_data
                else:
                    raise ValueError("无效的图像数据类型，必须是文件路径(str)或二进制数据(bytes)")

                if img_bytes is None:
                    raise IOError("无法获取有效的图像二进制数据")
                
                cursor.execute('''
                    INSERT INTO face_images (person_id, image_data, image_format, image_size, created_time)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (person_id, img_bytes, image_format, len(img_bytes)))

                image_id = cursor.lastrowid
                conn.commit()

                logging.debug(f"添加人脸图像成功: 人员ID {person_id}, 图像ID {image_id}")
                return image_id

            except Exception as e:
                logging.error(f"添加人脸图像失败: {str(e)}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def add_face_feature(self, person_id: int, feature_vector) -> int:
        """
        添加人脸特征向量
        
        Args:
            person_id: 人员ID
            feature_vector: 128维特征向量（dlib向量或Python列表）
            
        Returns:
            特征ID
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # 将dlib特征向量转换为Python列表
                if hasattr(feature_vector, '__iter__') and not isinstance(feature_vector, (list, tuple)):
                    # 如果是dlib向量类型，转换为Python列表
                    feature_list = list(feature_vector)
                else:
                    # 如果已经是列表或元组，直接使用
                    feature_list = list(feature_vector)
                
                # 将特征向量转换为JSON字符串
                feature_json = json.dumps(feature_list)
                
                # 生成特征哈希值（用于快速查找重复特征）
                feature_hash = self._hash_feature(feature_list)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO face_features (person_id, feature_vector, feature_hash, created_time)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (person_id, feature_json, feature_hash))
                
                feature_id = cursor.lastrowid
                conn.commit()
                
                logging.debug(f"添加人脸特征成功: 人员ID {person_id}, 特征ID {feature_id}")
                return feature_id
                
            except Exception as e:
                logging.error(f"添加人脸特征失败: {str(e)}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def _hash_feature(self, feature_vector) -> str:
        """生成特征向量的哈希值"""
        # 确保特征向量是列表格式
        if hasattr(feature_vector, '__iter__') and not isinstance(feature_vector, (list, tuple)):
            # 如果是dlib向量类型，转换为Python列表
            feature_list = list(feature_vector)
        else:
            # 如果已经是列表或元组，直接使用
            feature_list = list(feature_vector)
        
        # 将特征向量转换为字符串并生成哈希
        feature_str = ','.join(map(str, feature_list))
        return str(hash(feature_str))
    
    def get_person_by_name_id(self, name: str, id_card: str = None) -> Optional[Dict]:
        """根据姓名和身份证号获取人员信息"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                if id_card:
                    cursor.execute('''
                        SELECT id, name, id_card, real_name, real_id_card, is_temp, is_important, created_time, updated_time
                        FROM persons WHERE name = ? AND id_card = ?
                    ''', (name, id_card))
                else:
                    cursor.execute('''
                        SELECT id, name, id_card, real_name, real_id_card, is_temp, is_important, created_time, updated_time
                        FROM persons WHERE name = ?
                    ''', (name,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'id_card': row[2],
                        'real_name': row[3],
                        'real_id_card': row[4],
                        'is_temp': bool(row[5]),
                        'is_important': bool(row[6]),
                        'created_time': row[7],
                        'updated_time': row[8]
                    }
                return None
                
            finally:
                conn.close()
    
    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """根据姓名获取人员信息（返回第一个匹配的记录）"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT id, name, id_card, real_name, real_id_card, is_temp, is_important, created_time, updated_time
                    FROM persons WHERE name = ?
                    ORDER BY created_time DESC LIMIT 1
                ''', (name,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'id_card': row[2],
                        'real_name': row[3],
                        'real_id_card': row[4],
                        'is_temp': bool(row[5]),
                        'is_important': bool(row[6]),
                        'created_time': row[7],
                        'updated_time': row[8]
                    }
                return None
                
            finally:
                conn.close()
    
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """根据ID获取人员信息"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT id, name, id_card, real_name, real_id_card, is_temp, is_important, created_time, updated_time
                    FROM persons WHERE id = ?
                ''', (person_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'id_card': row[2],
                        'real_name': row[3],
                        'real_id_card': row[4],
                        'is_temp': bool(row[5]),
                        'is_important': bool(row[6]),
                        'created_time': row[7],
                        'updated_time': row[8]
                    }
                return None
                
            finally:
                conn.close()
    
    def get_all_persons(self, include_temp: bool = False) -> List[Dict]:
        """获取所有人员信息
        
        Args:
            include_temp: 是否包含临时身份
            
        Returns:
            人员信息列表
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                if include_temp:
                    cursor.execute('''
                        SELECT id, name, id_card, real_name, real_id_card, is_temp, is_important, created_time, updated_time
                        FROM persons ORDER BY name, id_card
                    ''')
                else:
                    cursor.execute('''
                        SELECT id, name, id_card, real_name, real_id_card, is_temp, is_important, created_time, updated_time
                        FROM persons WHERE is_temp = 0 ORDER BY name, id_card
                    ''')
                
                persons = []
                for row in cursor.fetchall():
                    persons.append({
                        'id': row[0],
                        'name': row[1],
                        'id_card': row[2],
                        'real_name': row[3],
                        'real_id_card': row[4],
                        'is_temp': bool(row[5]),
                        'is_important': bool(row[6]),
                        'created_time': row[7],
                        'updated_time': row[8]
                    })
                
                return persons
                
            finally:
                conn.close()
    
    def get_face_image(self, person_id: int, image_id: int = None) -> Optional[bytes]:
        """获取人脸图像数据"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                if image_id:
                    cursor.execute('''
                        SELECT image_data FROM face_images WHERE person_id = ? AND id = ?
                    ''', (person_id, image_id))
                else:
                    # 获取最新的图像
                    cursor.execute('''
                        SELECT image_data FROM face_images WHERE person_id = ? 
                        ORDER BY created_time DESC LIMIT 1
                    ''', (person_id,))
                
                row = cursor.fetchone()
                return row[0] if row else None
                
            finally:
                conn.close()
    
    def get_face_features(self, person_id: int = None) -> List[Tuple[int, List[float], str, str]]:
        """获取人脸特征数据
        返回格式: (person_id, feature_vector, person_name, real_name)
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                if person_id:
                    cursor.execute('''
                        SELECT ff.person_id, ff.feature_vector, p.name, p.real_name
                        FROM face_features ff
                        JOIN persons p ON ff.person_id = p.id
                        WHERE ff.person_id = ?
                    ''', (person_id,))
                else:
                    cursor.execute('''
                        SELECT ff.person_id, ff.feature_vector, p.name, p.real_name
                        FROM face_features ff
                        JOIN persons p ON ff.person_id = p.id
                        ORDER BY p.created_time DESC
                    ''')
                
                features = []
                for row in cursor.fetchall():
                    person_id, feature_str, person_name, real_name = row
                    
                    # 解析特征向量 - 使用JSON格式
                    try:
                        feature_vector = json.loads(feature_str)
                        features.append((person_id, feature_vector, person_name, real_name))
                    except Exception as e:
                        logging.warning(f"解析特征向量失败 (person_id: {person_id}): {str(e)}")
                        continue
                
                return features
                
            finally:
                conn.close()
    
    def find_similar_face(self, feature_vector, threshold: float = 0.5) -> Optional[Tuple[int, float, str, str, bool]]:
        """
        查找相似的人脸
        
        Args:
            feature_vector: 待匹配的特征向量（dlib向量或Python列表）
            threshold: 相似度阈值
            
        Returns:
            匹配结果 (person_id, distance, person_name, real_name, is_important) 或 None
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # 将dlib特征向量转换为Python列表
                if hasattr(feature_vector, '__iter__') and not isinstance(feature_vector, (list, tuple)):
                    # 如果是dlib向量类型，转换为Python列表
                    feature_list = list(feature_vector)
                else:
                    # 如果已经是列表或元组，直接使用
                    feature_list = list(feature_vector)
                
                # 获取所有非临时身份的特征，包括real_name和is_important
                cursor.execute('''
                    SELECT f.person_id, f.feature_vector, p.name, p.real_name, p.is_important
                    FROM face_features f
                    JOIN persons p ON f.person_id = p.id
                    WHERE p.is_temp = 0
                ''')
                
                min_distance = float('inf')
                best_match = None
                
                for row in cursor.fetchall():
                    stored_feature = json.loads(row[1])
                    distance = self._calculate_distance(feature_list, stored_feature)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = (row[0], distance, row[2], row[3], bool(row[4]))  # person_id, distance, name, real_name, is_important
                
                # 检查是否满足阈值
                if best_match and best_match[1] < threshold:
                    return best_match
                
                return None
                
            finally:
                conn.close()
    
    def _calculate_distance(self, feature1, feature2) -> float:
        """计算两个特征向量之间的欧氏距离"""
        # 确保两个特征向量都是列表格式
        if hasattr(feature1, '__iter__') and not isinstance(feature1, (list, tuple)):
            feature1_list = list(feature1)
        else:
            feature1_list = list(feature1)
            
        if hasattr(feature2, '__iter__') and not isinstance(feature2, (list, tuple)):
            feature2_list = list(feature2)
        else:
            feature2_list = list(feature2)
        
        return np.linalg.norm(np.array(feature1_list) - np.array(feature2_list))
    
    def update_person_real_info(self, person_id: int, real_name: str, real_id_card: str, is_temp: bool = None) -> bool:
        """
        更新人员的真实身份信息
        
        Args:
            person_id: 人员ID
            real_name: 真实姓名
            real_id_card: 真实身份证号
            is_temp: 是否为临时身份，None表示不更新此字段
            
        Returns:
            是否更新成功
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                if is_temp is not None:
                    # 同时更新身份类型
                    cursor.execute('''
                        UPDATE persons 
                        SET real_name = ?, real_id_card = ?, is_temp = ?, updated_time = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (real_name, real_id_card, is_temp, person_id))
                else:
                    # 只更新真实身份信息
                    cursor.execute('''
                        UPDATE persons 
                        SET real_name = ?, real_id_card = ?, updated_time = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (real_name, real_id_card, person_id))
                
                conn.commit()
                success = cursor.rowcount > 0
                
                if success:
                    status = "临时身份" if is_temp else "真实身份" if is_temp is not None else "身份信息"
                    logging.info(f"更新人员{status}成功: ID {person_id} -> {real_name} - {real_id_card}")
                else:
                    logging.warning(f"未找到人员ID {person_id}，更新失败")
                
                return success
                
            except Exception as e:
                logging.error(f"更新人员真实身份信息失败: {str(e)}")
                conn.rollback()
                return False
            finally:
                conn.close()
    
    def set_important_status(self, person_id: int, is_important: bool) -> bool:
        """
        设置人员的重点关注状态
        
        Args:
            person_id: 人员ID
            is_important: 是否为重点关注人员
            
        Returns:
            是否设置成功
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    UPDATE persons 
                    SET is_important = ?, updated_time = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (is_important, person_id))
                
                conn.commit()
                success = cursor.rowcount > 0
                
                if success:
                    status = "重点关注" if is_important else "普通人员"
                    logging.info(f"设置人员重点关注状态成功: ID {person_id} -> {status}")
                else:
                    logging.warning(f"未找到人员ID {person_id}，设置重点关注状态失败")
                
                return success
                
            except Exception as e:
                logging.error(f"设置人员重点关注状态失败: {str(e)}")
                conn.rollback()
                return False
            finally:
                conn.close()
    
    def get_important_persons(self) -> List[Dict]:
        """
        获取所有重点关注人员
        
        Returns:
            重点关注人员列表
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT id, name, id_card, real_name, real_id_card, is_temp, is_important, created_time, updated_time
                    FROM persons
                    WHERE is_important = 1
                    ORDER BY updated_time DESC
                ''')
                
                persons = []
                for row in cursor.fetchall():
                    person = {
                        'id': row[0],
                        'name': row[1],
                        'id_card': row[2],
                        'real_name': row[3],
                        'real_id_card': row[4],
                        'is_temp': bool(row[5]),
                        'is_important': bool(row[6]),
                        'created_time': row[7],
                        'updated_time': row[8]
                    }
                    persons.append(person)
                
                return persons
                
            except Exception as e:
                logging.error(f"获取重点关注人员失败: {str(e)}")
                return []
            finally:
                conn.close()
    
    def delete_temp_persons(self, max_age_hours: int = 24) -> int:
        """删除过期的临时人员数据"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # 删除超过指定时间的临时人员
                cursor.execute('''
                    DELETE FROM persons 
                    WHERE is_temp = 1 AND 
                          created_time < datetime('now', '-{} hours')
                '''.format(max_age_hours))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logging.info(f"已删除 {deleted_count} 个过期的临时人员")
                
                return deleted_count
                
            except Exception as e:
                logging.error(f"删除临时人员失败: {str(e)}")
                conn.rollback()
                return 0
            finally:
                conn.close()
    
    def add_recognition_log(self, person_id: int = None, confidence: float = None, 
                           distance: float = None) -> int:
        """添加识别记录"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO recognition_logs (person_id, confidence, distance, frame_time)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (person_id, confidence, distance))
                
                log_id = cursor.lastrowid
                conn.commit()
                return log_id
                
            except Exception as e:
                logging.error(f"添加识别记录失败: {str(e)}")
                conn.rollback()
                return 0
            finally:
                conn.close()
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                stats = {}
                
                # 总人员数
                cursor.execute('SELECT COUNT(*) FROM persons')
                stats['total_persons'] = cursor.fetchone()[0]
                
                # 临时人员数
                cursor.execute('SELECT COUNT(*) FROM persons WHERE is_temp = 1')
                stats['temp_persons'] = cursor.fetchone()[0]
                
                # 真实身份人员数
                cursor.execute('SELECT COUNT(*) FROM persons WHERE is_temp = 0')
                stats['real_persons'] = cursor.fetchone()[0]
                
                # 总图像数
                cursor.execute('SELECT COUNT(*) FROM face_images')
                stats['total_images'] = cursor.fetchone()[0]
                
                # 总特征数
                cursor.execute('SELECT COUNT(*) FROM face_features')
                stats['total_features'] = cursor.fetchone()[0]
                
                # 识别记录数
                cursor.execute('SELECT COUNT(*) FROM recognition_logs')
                stats['total_logs'] = cursor.fetchone()[0]
                
                return stats
                
            finally:
                conn.close()
    
    def export_to_csv(self, csv_path: str = "data/features_all.csv") -> bool:
        """导出特征数据到CSV文件（兼容旧格式）"""
        try:
            features = self.get_face_features()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                
                for person_id, feature_vector, person_name, real_name in features:
                    # 写入格式：姓名, 特征1, 特征2, ..., 特征128
                    row = [person_name] + feature_vector
                    writer.writerow(row)
            
            logging.info(f"特征数据已导出到: {csv_path}")
            return True
            
        except Exception as e:
            logging.error(f"导出CSV失败: {str(e)}")
            return False
    
    def import_from_csv(self, csv_path: str = "data/features_all.csv") -> bool:
        """从CSV文件导入特征数据"""
        try:
            if not os.path.exists(csv_path):
                logging.warning(f"CSV文件不存在: {csv_path}")
                return False
            
            import csv
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                for row in reader:
                    if len(row) >= 129:  # 姓名 + 128维特征
                        name = row[0]
                        feature_vector = [float(x) for x in row[1:129]]
                        
                        # 添加人员
                        person_id = self.add_person(name, is_temp=False)
                        
                        # 添加特征
                        self.add_face_feature(person_id, feature_vector)
            
            logging.info(f"从CSV文件导入特征数据成功: {csv_path}")
            return True
            
        except Exception as e:
            logging.error(f"从CSV导入失败: {str(e)}")
            return False
    
    def backup_database(self, backup_path: str = None) -> bool:
        """备份数据库"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"data/face_database_backup_{timestamp}.db"
            
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logging.info(f"数据库备份成功: {backup_path}")
            return True
            
        except Exception as e:
            logging.error(f"数据库备份失败: {str(e)}")
            return False
    
    def delete_person(self, person_id: int) -> bool:
        """删除指定的人员及其所有相关数据"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # 删除人员记录（由于外键约束，会自动删除相关的图像和特征）
                cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logging.info(f"已删除人员ID {person_id} 及其所有相关数据")
                    return True
                else:
                    logging.warning(f"未找到人员ID {person_id}")
                    return False
                
            except Exception as e:
                logging.error(f"删除人员失败: {str(e)}")
                conn.rollback()
                return False
            finally:
                conn.close()
    
    def clear_database(self) -> bool:
        """清空数据库中的所有数据"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # 开始事务
                cursor.execute('BEGIN TRANSACTION')
                
                # 清空所有表
                cursor.execute('DELETE FROM recognition_logs')
                cursor.execute('DELETE FROM face_features')
                cursor.execute('DELETE FROM face_images')
                cursor.execute('DELETE FROM persons')
                
                # 重置自增ID
                cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("persons", "face_images", "face_features", "recognition_logs")')
                
                # 提交事务
                conn.commit()
                
                logging.info("数据库已清空")
                return True
                
            except Exception as e:
                logging.error(f"清空数据库失败: {str(e)}")
                conn.rollback()
                return False
            finally:
                conn.close()
    
    def close(self):
        """关闭数据库连接"""
        # SQLite会自动管理连接，这里主要是清理资源
        logging.info("人脸数据库管理器已关闭")
