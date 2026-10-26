# -*- coding: utf-8 -*-
"""
海康相机适配器模块
"""
import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class HikCameraAdapter:
    """海康相机适配器 - 解耦"""
    
    def __init__(self, camera_index: int = 0, capture_timeout_ms: int = 1000):
        self.camera_index = camera_index
        self.capture_timeout_ms = capture_timeout_ms
        self._adapter = None
        self._camera = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._worker_started = False
        self._init_adapter()
    
    def _init_adapter(self) -> None:
        """初始化适配器"""
        try:
            from hik_sdk_adapter import HikSdkCamera
            self._adapter = HikSdkCamera
            logger.info("海康 SDK 适配器已加载")
        except ImportError as e:
            raise RuntimeError(f"海康 SDK 适配器加载失败: {e}")
    
    def _ensure_worker(self) -> None:
        """确保工作线程已启动"""
        if self._worker_started:
            return
        self._worker_started = True
        
        worker = threading.Thread(target=self._capture_loop, name="hik-capture", daemon=True)
        worker.start()
        logger.info("海康相机抓取线程已启动")
    
    def _capture_loop(self) -> None:
        """抓取循环"""
        interval_s = 0.01  # 10ms，尽量由相机/SDK节奏决定
        
        while not self._stop_event.is_set():
            started = time.time()
            
            try:
                if self._camera is None:
                    self._camera = self._adapter({
                        'camera_index': self.camera_index,
                        'capture_timeout_ms': self.capture_timeout_ms
                    })
                    self._camera.open()

                frame = self._camera.read_frame()

                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                else:
                    logger.warning("海康相机抓取失败")
                    
            except Exception as e:
                logger.error(f"海康相机抓取错误: {e}")
                self._close_camera()
            
            elapsed = time.time() - started
            if elapsed < interval_s:
                time.sleep(interval_s - elapsed)

    def poll(self) -> tuple[bool, cv2.Mat | None]:
        """轮询获取最新帧"""
        self._ensure_worker()
        
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()
    
    def read(self) -> tuple[bool, cv2.Mat | None]:
        """读取帧（带重试和长等待）"""
        # 海康相机需要更长启动时间，最多等3秒
        max_wait = 30  # 3秒
        for _ in range(max_wait):
            ret, frame = self.poll()
            if ret and frame is not None:
                return True, frame
            time.sleep(0.1)  # 等待100ms
        return False, None
    
    def get_frame(self) -> tuple[bool, cv2.Mat | None]:
        """获取原始帧"""
        return self.read()
    
    def close(self) -> None:
        """关闭"""
        self._stop_event.set()
        self._close_camera()
        logger.info("海康相机已关闭")
    
    def _close_camera(self) -> None:
        """关闭相机连接"""
        if self._camera:
            try:
                self._camera.close()
            except Exception as e:
                logger.error(f"关闭海康相机错误: {e}")
        self._camera = None
