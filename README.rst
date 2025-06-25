Dlib 人脸识别监控系统
========================

项目简介
--------
本项目是一个基于 Dlib、OpenCV、Tkinter、SQLite 的屏幕人脸识别监控系统，支持 GPU 加速，具备弹窗提醒、系统托盘控制、API 联动、数据库管理等功能。适用于重点人员监控、身份核查、访客管理等场景。

主要功能
--------
- 屏幕实时人脸检测与识别：自动捕获屏幕内容，检测并识别所有出现的人脸。
- GPU/CPU 自动切换：自动检测 GPU 环境，优先使用 GPU 加速。
- 透明窗口覆盖：识别窗口可全屏透明显示，不影响用户正常操作。
- 系统托盘菜单：一键切换弹窗、自动发现、识别阈值、状态显示等功能。
- 重点关注人员弹窗提醒：识别到重点关注人员时自动弹窗并截图留存。
- 新面孔自动发现与身份升级：自动检测新面孔，调用 API 获取真实身份并升级数据库。
- SQLite 数据库存储：所有人脸、特征、图片、身份信息均持久化存储。
- 人脸库管理、手动采集、数据库清理等工具：便于维护和扩展。
- 详细日志记录：所有操作、异常、识别结果均有日志记录。

项目结构
--------
::

    Dlib_face_recognition_from_camera/
    │
    ├── screen_face_monitor.py         # 主程序（监控+识别+UI+托盘）
    ├── face_database_manager.py       # 数据库管理器（核心数据操作）
    ├── face_library_manager.py        # 人脸库管理工具（可视化管理）
    ├── face_collector_from_image.py   # 手动采集人脸工具
    ├── important_person_manager.py    # 重点关注人员管理工具
    ├── clear_database_tool.py         # 数据库清空工具
    ├── view_logs.py                   # 日志查看工具
    ├── face_recognition_api.py        # 人脸识别API服务
    ├── start_system.py                # 系统启动脚本
    ├── simsun.ttc                     # 中文字体文件
    ├── requirements.txt               # 依赖包列表
    ├── README.rst                     # 项目说明（本文件）
    ├── LICENSE                        # 开源许可证
    │
    ├── data/
    │   ├── data_dlib/                 # Dlib 模型文件
    │   │   ├── dlib_face_recognition_resnet_model_v1.dat
    │   │   ├── mmod_human_face_detector.dat
    │   │   └── shape_predictor_68_face_landmarks.dat
    │   ├── data_faces_from_camera/    # 采集的人脸图片/截图
    │   └── face_database.db           # 主数据库
    │
    ├── demo/                          # 演示和测试脚本
    │   ├── demo.py                    # 基础演示
    │   ├── check_gpu.py               # GPU检测工具
    │   ├── add_real_face.py           # 添加真实人脸
    │   ├── cleanup_temp_identities.py # 清理临时身份
    │   ├── features_extraction_to_csv.py # 特征提取到CSV
    │   ├── face_reco_from_camera.py   # 摄像头人脸识别
    │   ├── face_reco_from_camera_single_face.py # 单脸识别
    │   ├── face_reco_from_camera_ot.py # 实时跟踪识别
    │   ├── get_faces_from_camera.py   # 摄像头人脸采集
    │   ├── get_faces_from_camera_tkinter.py # GUI人脸采集
    │   ├── face_descriptor_from_camera.py # 人脸特征提取
    │   └── how_to_use_camera.py       # 摄像头使用说明
    │
    └── logs/                          # 日志文件
        ├── face_library_manager.log   # 人脸库管理日志
        └── face_monitor_YYYYMMDD.log  # 监控系统日志

核心组件说明
-----------
### 1. screen_face_monitor.py - 主程序
**功能**：屏幕人脸识别监控的核心程序
- **实现方法**：
  - 使用 mss 库实时捕获屏幕画面
  - 基于 Dlib CNN 人脸检测器检测人脸
  - 使用 ResNet 模型提取 128 维人脸特征向量
  - 透明 Tkinter 窗口显示识别结果
  - 系统托盘菜单控制各项功能
  - 支持 GPU/CPU 自动切换优化性能
  - 新面孔自动发现与 API 身份升级
  - 重点关注人员弹窗提醒与截图

### 2. face_database_manager.py - 数据库管理器
**功能**：SQLite 数据库的核心操作管理
- **实现方法**：
  - 线程安全的数据库操作（使用锁机制）
  - 四表结构：persons（人员信息）、face_images（人脸图片）、face_features（特征向量）、recognition_logs（识别记录）
  - 支持临时身份与真实身份管理
  - 人脸特征向量相似度计算与匹配
  - 重点关注人员标记与管理
  - 数据库备份、导入导出功能
  - 自动清理过期临时身份

### 3. face_library_manager.py - 人脸库管理工具
**功能**：可视化的人脸库管理界面
- **实现方法**：
  - Tkinter GUI 界面，左右分栏布局
  - 左侧显示人员列表（Treeview 组件）
  - 右侧显示详细信息和人脸图片预览
  - 支持按身份证号去重显示
  - 实时统计各类人员数量
  - 支持删除、设置重点关注、导出数据等操作
  - 图片数据从数据库二进制字段读取显示

### 4. face_collector_from_image.py - 人脸采集工具
**功能**：从图片中批量采集人脸并入库
- **实现方法**：
  - 支持选择单张或多张图片
  - 自动检测图片中的所有人脸
  - 可视化人脸选择界面
  - 支持批量保存不同人员的人脸
  - 智能处理文件路径和二进制数据
  - 实时预览已注册人员信息
  - 支持删除已注册人员

### 5. important_person_manager.py - 重点关注人员管理
**功能**：管理重点关注人员列表
- **实现方法**：
  - 简洁的 Tkinter 界面
  - 显示所有重点关注人员信息
  - 支持按身份证号批量取消重点关注
  - 按创建时间排序显示
  - 实时刷新数据

### 6. face_recognition_api.py - 人脸识别API服务
**功能**：模拟第三方人脸识别API服务
- **实现方法**：
  - Flask Web 服务，支持跨域请求
  - 接收 base64 编码的人脸图片
  - 模拟识别延迟和成功率
  - 随机生成中文姓名和身份证号
  - 返回标准化的 JSON 响应格式
  - 健康检查接口

### 7. 其他工具脚本
- **clear_database_tool.py**：一键清空数据库
- **view_logs.py**：日志查看工具
- **start_system.py**：系统启动脚本
- **demo/**：各种演示和测试脚本

依赖环境
--------
- Python 3.10+
- dlib==20.0.0 (CUDA 11.8支持)
- opencv-python==4.11.0.86
- numpy==2.2.6
- pandas==2.3.0
- pillow==11.2.1
- pyautogui==0.9.54
- mss==10.0.0
- pystray==0.19.5
- requests==2.32.4
- flask==3.1.1, flask-cors==6.0.1（如需API联动）
- pywin32==310（Windows系统）
- sqlite3（Python自带）
- tkinter（Python自带）

安装依赖::

    # 方法1：使用pip安装（基于requirements.txt）
    pip install -r requirements.txt
    
    # 方法2：使用conda安装dlib（推荐）
    conda install -c conda-forge dlib=20.0.0
    pip install -r requirements.txt
    
    # 方法3：完整conda环境安装
    conda create -n face_recognition python=3.10
    conda activate face_recognition
    conda install -c conda-forge dlib=20.0.0
    pip install opencv-python==4.11.0.86 numpy==2.2.6 pandas==2.3.0 pillow==11.2.1 pyautogui==0.9.54 mss==10.0.0 pystray==0.19.5 requests==2.32.4 flask==3.1.1 flask-cors==6.0.1 pywin32==310

**注意**：
- 推荐使用conda安装dlib，因为conda会自动处理dlib的依赖关系，避免编译问题
- 当前环境使用CUDA 11.8版本的dlib，支持GPU加速
- 如果使用pip安装dlib遇到编译错误，请尝试conda方式
- 完整的环境配置可参考 `requirements_conda.txt` 和 `requirements_pip.txt`

快速开始
--------
1. 准备 Dlib 模型文件
   确保 ``data/data_dlib/`` 下有如下 3 个模型文件（如缺失请自行下载）：
   - dlib_face_recognition_resnet_model_v1.dat
   - mmod_human_face_detector.dat
   - shape_predictor_68_face_landmarks.dat

2. 运行主程序::

    python screen_face_monitor.py

3. 系统托盘菜单操作
   - 右下角托盘图标可一键切换弹窗、自动发现、阈值、状态显示等
   - 支持手动添加人脸、管理人脸库、清理临时身份、清空数据库等

4. 数据库/人脸库管理
   - ``face_library_manager.py``：可视化管理人脸库
   - ``important_person_manager.py``：管理重点关注人员
   - ``clear_database_tool.py``：一键清空数据库

5. 演示和测试
   - ``demo/check_gpu.py``：检测GPU环境
   - ``demo/face_reco_from_camera.py``：摄像头人脸识别演示
   - ``demo/get_faces_from_camera.py``：摄像头人脸采集演示

数据存储说明
------------
- 数据库：所有身份、特征、图片、重点关注、临时/真实身份等均存储于 ``data/face_database.db``
- 内存缓存：加速识别与弹窗，定期与数据库同步
- 图片文件：采集图片、弹窗截图等保存在 ``data/data_faces_from_camera/``
- 日志：``logs/face_monitor_YYYYMMDD.log`` 记录所有操作与异常

常见问题
--------
1. 模型文件缺失/损坏
   请重新下载 Dlib 官方模型，放入 ``data/data_dlib/``。

2. 依赖缺失/安装失败
   检查 Python 版本，优先使用官方源安装，必要时使用清华/阿里镜像。

3. API 联动不可用
   检查 ``requests`` 是否安装，API 服务是否启动，地址端口是否正确。

4. 数据库异常/损坏
   可用 ``clear_database_tool.py`` 清空数据库，或手动恢复备份。

5. 界面乱码/字体问题
   确保 ``simsun.ttc`` 字体文件存在于项目根目录。

6. GPU不可用
   检查 dlib 是否为 CUDA 版本，显卡驱动与 CUDA 环境是否配置正确。

进阶与扩展
----------
- 支持多摄像头/多屏幕扩展
- API 联动可对接第三方身份库
- 可自定义弹窗样式、识别阈值、重点关注规则
- 支持更多数据库（如 MySQL、PostgreSQL）
- 日志与数据可远程同步/备份

联系方式
--------
如有问题或建议，请联系项目维护者，或在 issue 区留言。 