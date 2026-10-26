# -*- coding: utf-8 -*-
"""
螺丝计数检测系统 - 优化版
支持: USB相机 / 海康相机 / 图片上传 / 模型选择

依赖:
    pip install flask opencv-python numpy pyyaml pillow ultralytics
    # 海康相机额外需要:
    pip install MVS SDK 并设置环境变量 HIK_MVS_ROOT
"""
import logging
import os
import sys
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response
from werkzeug.utils import secure_filename

# ==================== 统一配置 ====================
from config import (
    EdgeRuntimeConfig, SourceConfig, load_app_config, save_app_config,
    RUNTIME_ROOT, RUNTIME_DATA_DIR, create_default_config,
    get_detection_config, update_detection_config, DetectionConfig
)

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    template_folder=os.path.join(RUNTIME_ROOT, 'templates'),
    static_folder=os.path.join(RUNTIME_ROOT, 'static'),
    static_url_path='/static')

# 加载运行时配置
_config = create_default_config()

# 确保目录存在
_config.ensure_dirs()
app.config['UPLOAD_FOLDER'] = str(RUNTIME_ROOT / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['DATA_DIR'] = str(RUNTIME_ROOT / 'data')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(RUNTIME_ROOT / 'models', exist_ok=True)

# ==================== SDK 路径 ====================
def get_sdk_path() -> str | None:
    app_cfg = load_app_config()
    if app_cfg:
        path = app_cfg.get('sdk', {}).get('base_path')
        if path and os.path.exists(path):
            return path
    
    if sys.platform == 'win32':
        sdk_paths = [r'C:\MVS\Development', r'D:\MVS\Development']
    else:
        sdk_paths = ['/opt/MVS', '/opt/HIKVISION/MVS']
    
    for path in sdk_paths:
        if os.path.exists(path):
            return path
    return None

def set_sdk_path(path: str) -> None:
    app_cfg = load_app_config()
    if 'sdk' not in app_cfg:
        app_cfg['sdk'] = {}
    app_cfg['sdk']['base_path'] = path
    save_app_config(app_cfg)
    logger.info(f"SDK 路径已设置: {path}")

# ==================== 模型管理 ====================
def get_models_path() -> str:
    return str(RUNTIME_ROOT / 'models')

def scan_models() -> list:
    models_path = get_models_path()
    models = []
    if not os.path.exists(models_path):
        return models
    
    for filename in os.listdir(models_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.pt', '.pth', '.onnx', '.om']:
            models.append({'name': filename, 'path': os.path.join(models_path, filename), 'type': ext[1:]})
    return models

# ==================== 螺丝计数 - YOLO 模型 ====================
class ScrewCounter:
    """螺丝计数器 - 使用 YOLO 模型"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.loaded = False
        
    def load_model(self) -> bool:
        """加载 YOLO 模型"""
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.loaded = True
            logger.info(f"YOLO 模型已加载: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def count_screws(self, image: np.ndarray, return_annotated: bool = False) -> dict:
        """使用 YOLO 计数螺丝"""
        if not self.loaded:
            return {'count': 0, 'error': '模型未加载'}
        
        try:
            results = self.model.predict(image, verbose=False, conf=0.25)
            if not results:
                return {'count': 0, 'error': '无检测结果'}
            
            r = results[0]
            boxes = r.boxes
            count = len(boxes)
            
            # 可视化
            annotated = r.plot() if hasattr(r, 'plot') else image
            
            result = {
                'count': count,
                'boxes': boxes.xyxy.cpu().numpy().tolist() if count > 0 else [],
                'classes': boxes.cls.cpu().numpy().tolist() if count > 0 else [],
                'conf': boxes.conf.cpu().numpy().tolist() if count > 0 else []
            }
            if return_annotated:
                result['annotated_image'] = annotated
            return result
        except Exception as e:
            logger.error(f"计数失败: {e}")
            return {'count': 0, 'error': str(e)}

counter = ScrewCounter()


class AsyncDetector:
    """异步检测器：将推理从视频流线程剥离，复用最近标注结果。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_annotated = None
        self._job_frame = None
        self._job_conf = 0.25
        self._job_iou = 0.45
        self._job_seq = 0
        self._done_seq = 0
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, name="detector-worker", daemon=True)
        self._worker.start()

    def submit(self, frame: np.ndarray, conf: float, iou: float) -> None:
        with self._lock:
            self._job_seq += 1
            self._job_frame = frame.copy()
            self._job_conf = conf
            self._job_iou = iou

    def get_latest(self, fallback: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._latest_annotated is None:
                return fallback
            return self._latest_annotated

    def close(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=0.5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                has_job = self._job_frame is not None and self._job_seq > self._done_seq
                if has_job:
                    seq = self._job_seq
                    frame = self._job_frame
                    conf = self._job_conf
                    iou = self._job_iou
                else:
                    seq = 0

            if not has_job:
                self._stop_event.wait(0.005)
                continue

            annotated = frame
            try:
                if counter.loaded:
                    results = counter.model.predict(frame, verbose=False, conf=conf, iou=iou)
                    if results and len(results[0].boxes) > 0:
                        annotated = results[0].plot()
            except Exception as e:
                logger.warning(f"异步实时检测失败: {e}")

            with self._lock:
                if seq >= self._done_seq:
                    self._latest_annotated = annotated
                    self._done_seq = seq


detector = AsyncDetector()

# ==================== 摄像头模块 ====================
class Camera:
    def __init__(self):
        self.usb_camera = None
        self.frame_source = None
        self.type = None
        
    def open_usb(self, index: int = 0) -> bool:
        from cameras.usb_camera import USBCamera
        try:
            self.usb_camera = USBCamera(index)
            self.type = 'usb'
            logger.info(f"USB 相机已打开: {index}")
            return True
        except Exception as e:
            logger.error(f"USB 相机打开失败: {e}")
            return False
    
    def open_hikvision(self, camera_index: int = 0, capture_timeout_ms: int = 1000) -> bool:
        from cameras.hik_camera import HikCameraAdapter
        try:
            self.frame_source = HikCameraAdapter(camera_index, capture_timeout_ms)
            self.type = 'hikvision'
            logger.info(f"海康相机已打开: {camera_index}")
            return True
        except Exception as e:
            logger.error(f"海康相机打开失败: {e}")
            return False
    
    def read(self) -> tuple[bool, cv2.Mat | None]:
        if self.type == 'usb' and self.usb_camera:
            return self.usb_camera.get_frame()
        if self.type == 'hikvision' and self.frame_source:
            return self.frame_source.get_frame()
        return False, None
    
    def release(self) -> None:
        if self.usb_camera:
            self.usb_camera.release()
            self.usb_camera = None
        if self.frame_source:
            self.frame_source.close()
            self.frame_source = None
        self.type = None
        logger.info("相机已释放")

camera = Camera()

# ==================== 路由 ====================
from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    app_cfg = load_app_config()
    app_cfg['sdk']['base_path'] = get_sdk_path()
    return jsonify(app_cfg)

@app.route('/api/config', methods=['POST'])
def update_config():
    data = request.json
    if 'sdk_path' in data:
        set_sdk_path(data['sdk_path'])
    return jsonify({'success': True})

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(scan_models())

@app.route('/api/models', methods=['POST'])
def set_model():
    data = request.json
    model_name = data.get('model', '')
    if model_name:
        model_path = os.path.join(get_models_path(), model_name)
        if os.path.exists(model_path):
            counter.model_path = model_path
            counter.load_model()
            # 更新配置中的模型名称
            update_detection_config(model_name=model_name)
            return jsonify({'success': True, 'model': model_name})
    return jsonify({'success': False}), 400


@app.route('/api/config/detection', methods=['GET'])
def get_detection_settings():
    """获取检测参数"""
    cfg = get_detection_config()
    return jsonify(cfg.to_dict())


@app.route('/api/config/detection', methods=['POST'])
def update_detection_settings():
    """更新检测参数"""
    data = request.json
    if not data:
        return jsonify({'error': '请求体为空', 'code': 'EMPTY_REQUEST'}), 400
    
    try:
        cfg = update_detection_config(
            confidence=data.get('confidence'),
            iou_threshold=data.get('iou_threshold'),
            detect_interval=data.get('detect_interval'),
        )
        return jsonify({'success': True, 'config': cfg.to_dict()})
    except Exception as e:
        logger.error(f"参数更新失败: {e}")
        return jsonify({'error': f'参数更新失败: {str(e)}', 'code': 'UPDATE_ERROR'}), 400

@app.route('/api/cameras', methods=['GET'])
def scan_cameras():
    devices = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append({'id': i, 'name': f'USB Camera {i}', 'type': 'usb'})
            cap.release()
    return jsonify(devices)

@app.route('/api/camera/open', methods=['POST'])
def open_camera():
    data = request.json
    cam_type = data.get('type', 'usb')
    
    if cam_type == 'usb':
        success = camera.open_usb(data.get('index', 0))
    elif cam_type == 'hikvision':
        # 支持 ip 参数来指定相机IP
        camera_index = data.get('camera_index', 0)
        capture_timeout_ms = data.get('capture_timeout_ms', 1000)
        ip = data.get('ip')  # 海康相机IP地址
        
        if ip:
            # 如果提供了IP，使用IP连接
            # 注意：当前代码使用 camera_index，需要配置IP的逻辑
            # 暂时使用 camera_index
            pass
            
        success = camera.open_hikvision(camera_index, capture_timeout_ms)
    else:
        success = False
        
    return jsonify({'success': success, 'type': camera.type})

@app.route('/api/camera/close', methods=['POST'])
def close_camera():
    camera.release()
    return jsonify({'success': True})

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    return jsonify({'type': camera.type, 'is_open': camera.usb_camera is not None or camera.frame_source is not None})

@app.route('/api/sdk', methods=['GET'])
def get_sdk_info():
    sdk_path = get_sdk_path()
    return jsonify({'installed': sdk_path is not None, 'path': sdk_path, 'version': ''})

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': '请选择文件', 'code': 'NO_FILE'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名不能为空', 'code': 'EMPTY_FILENAME'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型，仅支持 jpg/jpeg/png/bmp', 'code': 'INVALID_TYPE'}), 400
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({'error': '图片读取失败，请检查文件是否损坏', 'code': 'READ_FAILED'}), 400
        return jsonify({'success': True, 'path': filepath, 'image': f'/uploads/{filename}'})
    except Exception as e:
        logger.error(f"上传图片失败: {e}")
        return jsonify({'error': f'上传失败: {str(e)}', 'code': 'UPLOAD_ERROR'}), 500


@app.route('/api/count', methods=['POST'])
def count_screws():
    data = request.json
    if not data:
        return jsonify({'error': '请求体为空', 'code': 'EMPTY_REQUEST'}), 400
    
    source = data.get('source', 'image')
    image = None
    error_msg = None
    
    if source == 'camera':
        ret, frame = camera.read()
        if not ret:
            return jsonify({'error': '相机未打开或读取失败', 'code': 'CAMERA_ERROR'}), 400
        image = frame
    elif source == 'image':
        image_path = data.get('image_path', '')
        if not image_path:
            return jsonify({'error': '图片路径不能为空', 'code': 'EMPTY_PATH'}), 400
        if not os.path.exists(image_path):
            return jsonify({'error': f'图片不存在: {image_path}', 'code': 'FILE_NOT_FOUND'}), 400
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': '图片读取失败', 'code': 'READ_FAILED'}), 400
    else:
        return jsonify({'error': f'未知数据源: {source}', 'code': 'UNKNOWN_SOURCE'}), 400
    
    if image is None:
        return jsonify({'error': '没有可用的图片', 'code': 'NO_IMAGE'}), 400
    
    try:
        result = counter.count_screws(image, return_annotated=True)
        
        # 保存带标注的结果图像
        if 'count' in result and result['count'] >= 0:
            try:
                annotated = result.get('annotated_image')
                if annotated is not None:
                    result_path = os.path.join(app.config['DATA_DIR'], 'result.jpg')
                    cv2.imwrite(result_path, annotated)
                    result['result_image'] = '/data/result.jpg'
            except Exception as e:
                logger.warning(f"保存标注图失败: {e}")
                result['result_image'] = None
        result.pop('annotated_image', None)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"计数失败: {e}")
        return jsonify({'error': f'计数失败: {str(e)}', 'code': 'COUNT_ERROR'}), 500

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """生成视频流帧（带实时检测框）"""
    frame_count = 0
    cfg = get_detection_config()
    
    while True:
        ret, frame = camera.read()
        if ret:
            frame_count += 1
            
            # 每隔几帧提交一次异步检测，流线程始终使用最近结果，避免被推理阻塞。
            interval = cfg.detect_interval if cfg.detect_interval and cfg.detect_interval > 0 else 1
            if counter.loaded and frame_count % interval == 0:
                detector.submit(frame, cfg.confidence, cfg.iou_threshold)
            annotated = detector.get_latest(frame)
            
            # 编码并发送
            _, buffer = cv2.imencode('.jpg', annotated)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # 相机未连接时，发送空白帧
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\r\n')

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    logger.info(f"项目目录: {RUNTIME_ROOT}")
    logger.info(f"模型目录: {get_models_path()}")
    logger.info(f"SDK 路径: {get_sdk_path()}")
    logger.info("访问: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
