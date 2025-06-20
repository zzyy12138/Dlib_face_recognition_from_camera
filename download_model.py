import urllib.request
import bz2
import os

def download_and_extract():
    # 创建目录
    os.makedirs('data/data_dlib', exist_ok=True)
    
    # 下载文件
    url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
    print("正在下载模型文件...")
    urllib.request.urlretrieve(url, "data/data_dlib/mmod_human_face_detector.dat.bz2")
    
    # 解压文件
    print("正在解压文件...")
    with bz2.open("data/data_dlib/mmod_human_face_detector.dat.bz2", 'rb') as source, \
         open("data/data_dlib/mmod_human_face_detector.dat", 'wb') as dest:
        dest.write(source.read())
    
    # 删除压缩文件
    os.remove("data/data_dlib/mmod_human_face_detector.dat.bz2")
    print("完成！")

if __name__ == "__main__":
    download_and_extract() 