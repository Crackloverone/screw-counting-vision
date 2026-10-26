# -*- coding: utf-8 -*-
"""
帧源模块 - 支持 USB 相机、海康相机、目录监控
基于 hls 项目优化
"""
from __future__ import annotations

import importlib
import threading
import time
import os
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


class RuntimeFrame:
    """运行时帧对象"""
    __slots__ = ('frame_id', 'image_path', 'timestamp')
    
    def __init__(self, frame_id: str, image_path: str, timestamp: float = None):
        self.frame_id = frame_id
        self.image_path = image_path
        self.timestamp = timestamp or time.time()


class SourceConfig:
    """帧源配置"""
    __slots__ = (
        'type', 'watch_dir', 'glob', 'stable_delay_ms', 'live_dir',
        'frame_interval_ms', 'camera_index', 'capture_timeout_ms',
        'serial_number', 'sdk_module'
    )
    
    def __init__(self, 
                 type: str,
                 watch_dir: str = '',
                 glob: str = '*.jpg',
                 stable_delay_ms: int = 300,
                 live_dir: str = '',
                 frame_interval_ms: int = 100,
                 camera_index: int = 0,
                 capture_timeout_ms: int = 1000,
                 serial_number: str = '',
                 sdk_module: str = 'hik_sdk_adapter'):
        
        self.type = type
        self.watch_dir = watch_dir
        self.glob = glob
        self.stable_delay_ms = stable_delay_ms
        self.live_dir = live_dir
        self.frame_interval_ms = frame_interval_ms
        self.camera_index = camera_index
        self.capture_timeout_ms = capture_timeout_ms
        self.serial_number = serial_number
        self.sdk_module = sdk_module
    
    def get(self, key: str, default=None):
        """获取配置值"""
        return getattr(self, key, default)


class USBFrameSource:
    """USB 相机帧源 (OpenCV)"""
    
    def __init__(self, config: SourceConfig) -> None:
        if cv2 is None:
            raise ImportError("opencv-python 需要用于 USB 相机")
        
        self.config = config
        self.camera_index = config.camera_index
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self._cap = None
        self._counter = 0

    def poll(self) -> list[RuntimeFrame]:
        """轮询获取帧"""
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_index)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        frames = []
        if self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                self._counter += 1
                frame_id = f"usb_{self.camera_index}_{self._counter}"
                
                import tempfile
                temp_path = Path(tempfile.gettempdir()) / f"{frame_id}.jpg"
                cv2.imwrite(str(temp_path), frame)
                
                frames.append(RuntimeFrame(image_path=str(temp_path), frame_id=frame_id))
        
        return frames
    
    def release(self):
        """释放资源"""
        if self._cap:
            self._cap.release()
            self._cap = None


class DirectoryFrameSource:
    """目录监控帧源 - 检测稳定的新文件"""
    
    def __init__(self, config: SourceConfig) -> None:
        source_type = config.type.strip().lower()
        if source_type not in {"directory", "directory_poll"}:
            raise ValueError(f"不支持的源类型: {config.type}")
        
        self.config = config
        self.watch_dir = Path(config.watch_dir).expanduser().resolve()
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._seen_files: set[str] = set()

    def poll(self) -> list[RuntimeFrame]:
        """轮询检测新文件"""
        now = time.time()
        frames: list[RuntimeFrame] = []
        
        for path in sorted(self.watch_dir.glob(self.config.glob)):
            if not path.is_file():
                continue
            
            resolved = str(path.resolve())
            if resolved in self._seen_files:
                continue
            
            # 检查文件是否稳定（不再变更）
            age_ms = (now - path.stat().st_mtime) * 1000.0
            if age_ms < float(self.config.stable_delay_ms):
                continue
            
            frame_id = path.stem
            self._seen_files.add(resolved)
            frames.append(RuntimeFrame(frame_id=frame_id, image_path=resolved))
        
        return frames


class HikCameraFrameSource:
    """
    海康工业相机帧源
    
    使用 hik_sdk_adapter 模块进行持续抓取
    """
    
    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        
        # 实时目录
        live_dir = config.live_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime_data', 'live')
        self.live_dir = Path(live_dir).expanduser().resolve()
        self.live_dir.mkdir(parents=True, exist_ok=True)
        
        self._last_capture_at = 0.0
        self._sequence = 0
        self._camera = None
        self.last_error = ""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_frames: list[RuntimeFrame] = []
        self._worker_started = False
        self._capture_started_at = 0.0
        self._last_frame_at = 0.0
        
        # 加载 SDK 适配器
        try:
            self._adapter = importlib.import_module(config.sdk_module)
        except ImportError as e:
            raise RuntimeError(
                f"海康 SDK 适配器未安装: {config.sdk_module}\n"
                f"请安装 Hikvision MVS SDK 或设置环境变量 HIK_MVS_ROOT\n"
                f"错误: {e}"
            ) from e
        
        # 检查必需的函数
        if not hasattr(self._adapter, "open_camera") and not hasattr(self._adapter, "capture_frame"):
            raise RuntimeError(
                f"SDK 适配器 '{config.sdk_module}' 必须定义 open_camera(config) 或 capture_frame(output_path, config)"
            )

    def poll(self) -> list[RuntimeFrame]:
        """轮询获取帧"""
        self._ensure_worker()
        
        with self._lock:
            frames = list(self._ready_frames)
            self._ready_frames.clear()
            capture_started_at = self._capture_started_at
            last_frame_at = self._last_frame_at
            has_error = bool(self.last_error)
        
        if frames:
            return frames
        
        # 检查超时
        now = time.time()
        timeout_s = max(0.1, float(self.config.capture_timeout_ms) / 1000.0)
        
        if capture_started_at > 0 and now - capture_started_at > timeout_s and not has_error:
            self._set_error(f"海康相机抓取超时 {int(timeout_s * 1000)}ms")
        elif last_frame_at <= 0 and now - self._last_capture_at > timeout_s and not has_error:
            self._set_error("海康相机尚未产生第一帧")
        
        return []

    def close(self) -> None:
        """关闭"""
        self._stop_event.set()
        self._close_camera()

    def _close_camera(self) -> None:
        """关闭相机连接"""
        if self._camera is not None and hasattr(self._camera, "close"):
            try:
                self._camera.close()
            except Exception as exc:
                print(f"[海康] 关闭错误: {exc}")
        self._camera = None

    def _ensure_worker(self) -> None:
        """确保工作线程已启动"""
        if self._worker_started:
            return
        
        self._worker_started = True
        worker = threading.Thread(target=self._capture_loop, name="hik-camera-capture", daemon=True)
        worker.start()

    def _capture_loop(self) -> None:
        """抓取循环"""
        interval_s = max(0.001, float(self.config.frame_interval_ms) / 1000.0)
        
        while not self._stop_event.is_set():
            started_at = time.time()
            
            with self._lock:
                self._last_capture_at = started_at
                self._capture_started_at = started_at
            
            self._sequence += 1
            frame_id = f"hik_{int(started_at * 1000)}_{self._sequence:06d}"
            image_path = self.live_dir / f"{frame_id}.jpg"
            
            try:
                ok = self._capture_frame(image_path)
            except Exception as exc:
                self._set_error(str(exc))
                print(f"[海康] 抓取错误: {exc}")
                self._close_camera()
                
                with self._lock:
                    self._capture_started_at = 0.0
                
                time.sleep(max(0.2, interval_s))
                continue
            
            with self._lock:
                self._capture_started_at = 0.0
            
            if not ok or not image_path.exists():
                self._set_error("海康相机未返回帧")
                time.sleep(max(0.2, interval_s))
                continue
            
            # 成功
            frame = RuntimeFrame(frame_id=frame_id, image_path=str(image_path.resolve()))
            
            with self._lock:
                self.last_error = ""
                self._last_frame_at = time.time()
                self._ready_frames = [frame]
            
            time.sleep(interval_s)

    def _set_error(self, message: str) -> None:
        """设置错误"""
        with self._lock:
            self.last_error = message

    def _capture_frame(self, image_path: Path) -> bool:
        """抓取一帧"""
        config_dict = {
            'camera_index': self.config.camera_index,
            'capture_timeout_ms': self.config.capture_timeout_ms,
        }
        
        if hasattr(self._adapter, "open_camera"):
            if self._camera is None:
                self._camera = self._adapter.open_camera(config_dict)
            return bool(self._camera.read(str(image_path)))
        
        return bool(self._adapter.capture_frame(str(image_path), config_dict))


def build_frame_source(config: SourceConfig):
    """构建帧源"""
    source_type = config.type.strip().lower()
    
    if source_type in {"directory", "directory_poll"}:
        return DirectoryFrameSource(config)
    
    if source_type in {"usb", "usb_camera"}:
        return USBFrameSource(config)
    
    if source_type in {"hik", "hik_camera", "hikvision"}:
        return HikCameraFrameSource(config)
    
    raise ValueError(f"不支持的帧源类型: {config.type}")


# 便捷函数
def create_source(type: str, **kwargs) -> tuple:
    """
    创建帧源便捷函数
    
    返回 (frame_source, config)
    """
    default_config = SourceConfig(type=type, **kwargs)
    return build_frame_source(default_config), default_config