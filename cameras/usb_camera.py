# -*- coding: utf-8 -*-
"""
USB 相机模块
"""
import logging
from pathlib import Path
import tempfile

import cv2

logger = logging.getLogger(__name__)


class USBCamera:
    """USB 相机 - 解耦的相机类"""
    
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self._cap = cv2.VideoCapture(index)
        self._counter = 0
        
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开 USB 相机 {index}")
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info(f"USB 相机已初始化: index={index}, {width}x{height}")
    
    def read(self) -> tuple[bool, cv2.Mat | None]:
        """读取帧 - 返回原始图像"""
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        self._counter += 1
        return True, frame
    
    def read_path(self) -> tuple[bool, str | None]:
        """读取帧并保存到临时文件 - 返回路径"""
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        
        self._counter += 1
        frame_id = f"usb_{self.index}_{self._counter}"
        
        temp_path = Path(tempfile.gettempdir()) / f"{frame_id}.jpg"
        cv2.imwrite(str(temp_path), frame)
        
        return True, str(temp_path)
    
    def get_frame(self) -> tuple[bool, cv2.Mat]:
        """获取原始帧"""
        return self._cap.read()
    
    def release(self) -> None:
        """释放资源"""
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(f"USB 相机 {self.index} 已释放")