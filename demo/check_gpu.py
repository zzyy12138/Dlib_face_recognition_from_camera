"""
GPU检测工具
"""
import dlib
import sys

def check_gpu_status():
    """检查GPU状态"""
    print("=" * 50)
    print("人脸识别监控系统 - GPU检测工具")
    print("=" * 50)
    
    print("\n1. 检查dlib版本和CUDA支持:")
    print(f"   dlib版本: {dlib.__version__}")
    
    if hasattr(dlib, 'DLIB_USE_CUDA'):
        print(f"   CUDA支持: {'是' if dlib.DLIB_USE_CUDA else '否'}")
    else:
        print("   CUDA支持: 无法检测")
    
    print("\n2. 检查CUDA设备:")
    try:
        if hasattr(dlib, 'cuda') and hasattr(dlib.cuda, 'get_num_devices'):
            num_devices = dlib.cuda.get_num_devices()
            print(f"   检测到 {num_devices} 个CUDA设备")
            
            if num_devices > 0:
                print("   GPU设备详情:")
                for i in range(num_devices):
                    try:
                        device_name = dlib.cuda.get_device_name(i)
                        print(f"     设备 {i}: {device_name}")
                    except:
                        print(f"     设备 {i}: 无法获取名称")
            else:
                print("   未检测到可用的GPU设备")
        else:
            print("   无法获取CUDA设备信息")
    except Exception as e:
        print(f"   检测CUDA设备时出错: {e}")
    
    print("\n3. 测试人脸检测模型:")
    try:
        # 尝试加载模型
        print("   正在加载人脸检测模型...")
        cnn_face_detector = dlib.cnn_face_detection_model_v1('data/data_dlib/mmod_human_face_detector.dat')
        print("   ✓ 人脸检测模型加载成功")
        
        # 尝试加载特征点预测模型
        print("   正在加载特征点预测模型...")
        predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        print("   ✓ 特征点预测模型加载成功")
        
        # 尝试加载人脸识别模型
        print("   正在加载人脸识别模型...")
        face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        print("   ✓ 人脸识别模型加载成功")
        
        print("\n4. 运行模式建议:")
        if hasattr(dlib, 'DLIB_USE_CUDA') and dlib.DLIB_USE_CUDA:
            if hasattr(dlib, 'cuda') and hasattr(dlib.cuda, 'get_num_devices'):
                num_devices = dlib.cuda.get_num_devices()
                if num_devices > 0:
                    print("   ✓ 建议使用GPU加速模式")
                    print("   - 处理间隔: 50ms")
                    print("   - 图像缩放: 0.5")
                    print("   - 预期性能: 高")
                else:
                    print("   ⚠ 建议使用CPU优化模式")
                    print("   - 处理间隔: 100ms")
                    print("   - 图像缩放: 0.3")
                    print("   - 预期性能: 中等")
            else:
                print("   ⚠ 建议使用CPU优化模式")
        else:
            print("   ⚠ 建议使用CPU优化模式")
            print("   - 处理间隔: 100ms")
            print("   - 图像缩放: 0.3")
            print("   - 预期性能: 中等")
        
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        print("   请确保模型文件存在于 data/data_dlib/ 目录中")
    
    print("\n" + "=" * 50)

if __name__ == '__main__':
    check_gpu_status() 