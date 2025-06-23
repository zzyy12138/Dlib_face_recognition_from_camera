"""
人脸识别API模拟服务
功能：接收base64编码的人脸图片，返回随机的用户名和身份证号
"""

import base64
import json
import random
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 模拟的姓名库
CHINESE_NAMES = [
    "张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
    "郑一", "王二", "冯三", "陈四", "褚五", "卫六", "蒋七", "沈八",
    "韩九", "杨十", "朱一", "秦二", "尤三", "许四", "何五", "吕六",
    "施七", "张八", "孔九", "曹十", "严一", "华二", "金三", "魏四",
    "陶五", "姜六", "戚七", "谢八", "邹九", "喻十", "柏一", "水二",
    "窦三", "章四", "云五", "苏六", "潘七", "葛八", "奚九", "范十",
    "彭一", "郎二", "鲁三", "韦四", "昌五", "马六", "苗七", "凤八",
    "花九", "方十", "俞一", "任二", "袁三", "柳四", "酆五", "鲍六",
    "史七", "唐八", "费九", "廉十", "岑一", "薛二", "雷三", "贺四",
    "倪五", "汤六", "滕七", "殷八", "罗九", "毕十", "郝一", "邬二",
    "安三", "常四", "乐五", "于六", "时七", "傅八", "皮九", "卞十",
    "齐一", "康二", "伍三", "余四", "元五", "卜六", "顾七", "孟八",
    "平九", "黄十", "和一", "穆二", "萧三", "尹四", "姚五", "邵六",
    "湛七", "汪八", "祁九", "毛十", "禹一", "狄二", "米三", "贝四",
    "明五", "臧六", "计七", "伏八", "成九", "戴十", "谈一", "宋二",
    "茅三", "庞四", "熊五", "纪六", "舒七", "屈八", "项九", "祝十",
    "董一", "梁二", "杜三", "阮四", "蓝五", "闵六", "席七", "季八",
    "麻九", "强十", "贾一", "路二", "娄三", "危四", "江五", "童六",
    "颜七", "郭八", "梅九", "盛十", "林一", "刁二", "钟三", "徐四",
    "邱五", "骆六", "高七", "夏八", "蔡九", "田十", "樊一", "胡二",
    "凌三", "霍四", "虞五", "万六", "支七", "柯八", "昝九", "管十",
    "卢一", "莫二", "经三", "房四", "裘五", "缪六", "干七", "解八",
    "应九", "宗十", "丁一", "宣二", "贲三", "邓四", "郁五", "单六",
    "杭七", "洪八", "包九", "诸十", "左一", "石二", "崔三", "吉四",
    "钮五", "龚六", "程七", "嵇八", "邢九", "滑十", "裴一", "陆二",
    "荣三", "翁四", "荀五", "羊六", "於七", "惠八", "甄九", "曲十",
    "家一", "封二", "芮三", "羿四", "储五", "靳六", "汲七", "邴八",
    "糜九", "松十", "井一", "段二", "富三", "巫四", "乌五", "焦六",
    "巴七", "弓八", "牧九", "隗十", "山一", "谷二", "车三", "侯四",
    "宓五", "蓬六", "全七", "郗八", "班九", "仰十", "秋一", "仲二",
    "伊三", "宫四", "宁五", "仇六", "栾七", "暴八", "甘九", "钭十",
    "厉一", "戎二", "祖三", "武四", "符五", "刘六", "景七", "詹八",
    "束九", "龙十", "叶一", "幸二", "司三", "韶四", "郜五", "黎六",
    "蓟七", "薄八", "印九", "宿十", "白一", "怀二", "蒲三", "邰四",
    "从五", "鄂六", "索七", "咸八", "籍九", "赖十", "卓一", "蔺二",
    "屠三", "蒙四", "池五", "乔六", "阴七", "郁八", "胥九", "能十",
    "苍一", "双二", "闻三", "莘四", "党五", "翟六", "谭七", "贡八",
    "劳九", "逄十", "姬一", "申二", "扶三", "堵四", "冉五", "宰六",
    "郦七", "雍八", "舄九", "璩十", "桑一", "桂二", "濮三", "牛四",
    "寿五", "通六", "边七", "扈八", "燕九", "冀十", "郏一", "浦二",
    "尚三", "农四", "温五", "别六", "庄七", "晏八", "柴九", "瞿十",
    "阎一", "充二", "慕三", "连四", "茹五", "习六", "宦七", "艾八",
    "鱼九", "容十", "向一", "古二", "易三", "慎四", "戈五", "廖六",
    "庾七", "终八", "暨九", "居十", "衡一", "步二", "都三", "耿四",
    "满五", "弘六", "匡七", "国八", "文九", "寇十", "广一", "禄二",
    "阙三", "东四", "欧五", "殳六", "沃七", "利八", "蔚九", "越十",
    "夔一", "隆二", "师三", "巩四", "厍五", "聂六", "晁七", "勾八",
    "敖九", "融十", "冷一", "訾二", "辛三", "阚四", "那五", "简六",
    "饶七", "空八", "曾九", "毋十", "沙一", "乜二", "养三", "鞠四",
    "须五", "丰六", "巢七", "关八", "蒯九", "相十", "查一", "後二",
    "荆三", "红四", "游五", "竺六", "权七", "逯八", "盖九", "益十",
    "桓一", "公二", "万俟三", "司马四", "上官五", "欧阳六", "夏侯七", "诸葛八",
    "闻人九", "东方十", "赫连一", "皇甫二", "尉迟三", "公羊四", "澹台五", "公冶六",
    "宗政七", "濮阳八", "淳于九", "单于十", "太叔一", "申屠二", "公孙三", "仲孙四",
    "轩辕五", "令狐六", "钟离七", "宇文八", "长孙九", "慕容十", "司徒一", "司空二"
]

# 模拟的身份证号生成函数
def generate_random_id_card():
    """生成随机的18位身份证号"""
    # 省份代码 (前2位)
    province_codes = ['11', '12', '13', '14', '15', '21', '22', '23', '31', '32', '33', '34', '35', '36', '37', '41', '42', '43', '44', '45', '46', '50', '51', '52', '53', '54', '61', '62', '63', '64', '65']
    
    # 地区代码 (3-4位)
    area_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    
    # 年份 (5-8位)
    year = random.randint(1960, 2005)
    
    # 月份 (9-10位)
    month = random.randint(1, 12)
    
    # 日期 (11-12位)
    day = random.randint(1, 28)  # 使用28避免月份问题
    
    # 顺序码 (13-16位)
    sequence = random.randint(1, 999)
    
    # 校验码 (17位)
    check_codes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
    
    # 组装身份证号
    id_card = (
        random.choice(province_codes) +
        random.choice(area_codes) +
        str(year) +
        f"{month:02d}" +
        f"{day:02d}" +
        f"{sequence:03d}" +
        random.choice(check_codes)
    )
    
    return id_card

@app.route('/api/recognize_face', methods=['POST'])
def recognize_face():
    """
    人脸识别API接口
    接收base64编码的图片，返回识别结果
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': '缺少image_base64参数'
            }), 400
        
        # 获取base64图片数据
        image_base64 = data['image_base64']
        
        # 验证base64数据
        try:
            # 尝试解码base64数据
            image_data = base64.b64decode(image_base64)
            logging.info(f"接收到图片数据，大小: {len(image_data)} 字节")
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'无效的base64数据: {str(e)}'
            }), 400
        
        # 模拟处理延迟 (0.5-2秒)
        processing_time = random.uniform(0.5, 2.0)
        time.sleep(processing_time)
        
        # 随机生成识别结果
        name = random.choice(CHINESE_NAMES)
        id_card = generate_random_id_card()
        
        # 模拟识别成功率 (90%成功率)
        success_rate = random.random()
        if success_rate < 0.9:
            # 成功识别
            result = {
                'success': True,
                'data': {
                    'name': name,
                    'id_card': id_card,
                    'confidence': round(random.uniform(0.85, 0.99), 3),
                    'processing_time': round(processing_time, 2)
                }
            }
            logging.info(f"识别成功: {name} - {id_card}")
        else:
            # 识别失败
            result = {
                'success': False,
                'error': '无法识别该人脸',
                'data': {
                    'processing_time': round(processing_time, 2)
                }
            }
            logging.info("识别失败: 无法识别该人脸")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"API处理出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'服务器内部错误: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'success': True,
        'message': '人脸识别API服务正常运行',
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def index():
    """根路径，返回API使用说明"""
    return jsonify({
        'message': '人脸识别API服务',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/recognize_face': '人脸识别接口',
            'GET /api/health': '健康检查接口'
        },
        'usage': {
            'recognize_face': {
                'method': 'POST',
                'url': '/api/recognize_face',
                'content_type': 'application/json',
                'body': {
                    'image_base64': 'base64编码的图片数据'
                },
                'response': {
                    'success': 'boolean',
                    'data': {
                        'name': '识别出的姓名',
                        'id_card': '识别出的身份证号',
                        'confidence': '识别置信度',
                        'processing_time': '处理时间(秒)'
                    }
                }
            }
        }
    })

if __name__ == '__main__':
    # 启动API服务
    print("启动人脸识别API服务...")
    print("API地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/api/health")
    print("人脸识别: POST http://localhost:5000/api/recognize_face")
    print("按 Ctrl+C 停止服务")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 