# -*- coding: utf-8 -*-
"""
海康威视工业相机 SDK 适配器
基于 hls 项目的实现优化
"""
from __future__ import annotations

import sys
import os
from ctypes import POINTER, byref, cast, c_ubyte, memset, sizeof
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Windows 平台的海康 SDK 常见安装路径
HIK_SDK_WINDOWS_PATHS = [
    r'C:\MVS\Development',
    r'D:\MVS\Development',
    r'C:\Program Files\MVS\Development',
    r'D:\Program Files\MVS\Development',
]

# Linux 平台的 SDK 路径 (支持多种架构)
HIK_SDK_LINUX_PATHS = [
    '/opt/MVS',
    '/opt/HIKVISION/MVS',
    '/usr/local/MVS',
]

# Linux 上不同架构的 SDK Python 目录
HIK_SDK_LINUX_ARCH_PATHS = [
    'Samples/aarch64/Python/MvImport',  # ARM64 (香橙派)
    'Samples/x86_64/Python/MvImport',  # x86_64 Linux
    'Samples/Python/MvImport',         # 通用
    'Samples/Python',                   # 另一种结构
]


def get_sdk_root() -> Path | None:
    """自动检测 SDK 根目录"""
    # 先检查环境变量
    env_path = os.environ.get('HIK_MVS_ROOT')
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    
    # 尝试常见路径
    if sys.platform == 'win32':
        paths = HIK_SDK_WINDOWS_PATHS
    else:
        # Linux/ARM: 先检查架构目录
        import platform
        machine = platform.machine()
        
        # 优先检查对应架构的目录
        if machine == 'aarch64':
            # 香橙派 ARM64
            paths = ['/opt/MVS', '/opt/HIKVISION/MVS', '/usr/local/MVS']
        elif machine == 'x86_64':
            # x86_64 Linux
            paths = ['/opt/MVS', '/opt/HIKVISION/MVS', '/usr/local/MVS']
        else:
            paths = HIK_SDK_LINUX_PATHS
    
    for path in paths:
        p = Path(path)
        if p.exists():
            return p
    return None


def _ensure_hik_import_path(sdk_root: Path) -> None:
    """确保 SDK Python 模块在 sys.path 中"""
    root_text = str(sdk_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    
    # 根据架构选择正确的 Python 模块路径
    # 香橙派 (ARM64): Samples/aarch64/Python/MvImport
    # x86_64 Linux: Samples/x86_64/Python/MvImport
    arch_paths = HIK_SDK_LINUX_ARCH_PATHS
    
    for subpath in arch_paths:
        python_path = sdk_root / subpath
        if python_path.exists():
            python_str = str(python_path)
            if python_str not in sys.path:
                sys.path.insert(0, python_str)
            return
    
    # 备选：尝试 Samples/Python
    python_path = sdk_root / 'Samples' / 'Python'
    if python_path.exists():
        python_str = str(python_path)
        if python_str not in sys.path:
            sys.path.insert(0, python_str)


def _load_hik_sdk():
    """加载海康 SDK 模块"""
    sdk_root = get_sdk_root()
    if sdk_root is None:
        raise RuntimeError(
            "海康 MVS SDK 未找到。请安装 Hikvision MVS 或设置环境变量 HIK_MVS_ROOT"
        )
    
    _ensure_hik_import_path(sdk_root)
    
    # 尝试不同的 SDK 目录结构
    for subdir in ['Samples/aarch64/Python/MvImport', 'Samples/Python/MvImport', 'MvImport', 'Python']:
        sdk_path = sdk_root / subdir
        if sdk_path.exists():
            _ensure_hik_import_path(sdk_path)
    
    try:
        from MvImport.CameraParams_const import MV_ACCESS_Exclusive, MV_GIGE_DEVICE, MV_USB_DEVICE
        from MvImport import CameraParams_header as camera_header
        from MvImport.CameraParams_header import MV_CC_DEVICE_INFO, MV_CC_DEVICE_INFO_LIST, MV_FRAME_OUT
        from MvImport.MvCameraControl_class import MvCamera
        from MvImport.MvErrorDefine_const import MV_OK
        from MvImport.PixelType_header import PixelType_Gvsp_BGR8_Packed, PixelType_Gvsp_RGB8_Packed
    except ImportError as e:
        # 尝试从 DLL 直接加载
        raise RuntimeError(
            f"无法导入海康 MVS SDK 模块。请确认 SDK 已正确安装。\n"
            f"SDK 路径: {sdk_root}\n错误: {e}"
        ) from e
    
    # 查找像素转换参数结构
    pixel_convert_param = getattr(camera_header, "MV_PIXEL_CONVERT_PARAM", None)
    if pixel_convert_param is None:
        pixel_convert_param = getattr(camera_header, "MV_CC_PIXEL_CONVERT_PARAM", None)
    if pixel_convert_param is None:
        raise RuntimeError(
            "无法找到海康像素转换结构。请确认 SDK 版本"
        )

    return {
        "MV_ACCESS_Exclusive": MV_ACCESS_Exclusive,
        "MV_GIGE_DEVICE": MV_GIGE_DEVICE,
        "MV_USB_DEVICE": MV_USB_DEVICE,
        "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
        "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
        "MV_FRAME_OUT": MV_FRAME_OUT,
        "MV_PIXEL_CONVERT_PARAM": pixel_convert_param,
        "MvCamera": MvCamera,
        "MV_OK": MV_OK,
        "PixelType_Gvsp_BGR8_Packed": PixelType_Gvsp_BGR8_Packed,
        "PixelType_Gvsp_RGB8_Packed": PixelType_Gvsp_RGB8_Packed,
    }


class HikSdkCamera:
    """
    海康工业相机连接类
    
    支持 GigE 和 USB 相机，持久化连接用于持续抓取
    """
    
    def __init__(self, config: dict = None) -> None:
        self.config = config or {}
        self.sdk = None
        self.camera = None
        self.opened = False
        self.grabbing = False
        self._consecutive_failures = 0
        
    @property
    def capture_timeout_ms(self) -> int:
        return self.config.get('capture_timeout_ms', 1000)
    
    @property
    def camera_index(self) -> int:
        return self.config.get('camera_index', 0)
    
    def open(self) -> None:
        """打开相机连接"""
        if self.grabbing:
            return
            
        self.sdk = _load_hik_sdk()
        
        MvCamera = self.sdk["MvCamera"]
        MV_OK = self.sdk["MV_OK"]
        device_list = self.sdk["MV_CC_DEVICE_INFO_LIST"]()
        transport_layer = self.sdk["MV_GIGE_DEVICE"] | self.sdk["MV_USB_DEVICE"]
        
        ret = MvCamera.MV_CC_EnumDevices(transport_layer, device_list)
        _check_ret(ret, MV_OK, "MV_CC_EnumDevices")
        if device_list.nDeviceNum <= 0:
            raise RuntimeError("未找到海康相机设备")
        
        # 选择相机索引
        index = self.camera_index
        if index < 0 or index >= int(device_list.nDeviceNum):
            raise RuntimeError(f"相机索引 {index} 超出范围，设备数量: {device_list.nDeviceNum}")
        
        device_info = cast(device_list.pDeviceInfo[index], POINTER(self.sdk["MV_CC_DEVICE_INFO"])).contents
        
        self.camera = MvCamera()
        ret = self.camera.MV_CC_CreateHandle(device_info)
        _check_ret(ret, MV_OK, "MV_CC_CreateHandle")
        
        # 独占访问模式打开设备
        ret = self.camera.MV_CC_OpenDevice(self.sdk["MV_ACCESS_Exclusive"], 0)
        _check_ret(ret, MV_OK, "MV_CC_OpenDevice")
        self.opened = True
        
        # 关闭触发模式
        _set_trigger_off(self.camera, MV_OK)
        
        ret = self.camera.MV_CC_StartGrabbing()
        _check_ret(ret, MV_OK, "MV_CC_StartGrabbing")
        self.grabbing = True

    def read(self, output_path: str) -> bool:
        """读取一帧图像并保存"""
        try:
            self.open()
            self._read_once(output_path)
            self._consecutive_failures = 0
            return True
        except Exception as exc:
            self._consecutive_failures += 1
            print(f"[海康] 抓取失败: {exc}")
            self._reconnect()
            return False

    def read_frame(self) -> np.ndarray | None:
        """读取一帧并直接返回 BGR ndarray（避免磁盘 JPEG 往返）"""
        try:
            self.open()
            frame = self._read_once_frame()
            self._consecutive_failures = 0
            return frame
        except Exception as exc:
            self._consecutive_failures += 1
            print(f"[海康] 抓取失败: {exc}")
            self._reconnect()
            return None

    def close(self) -> None:
        """关闭相机连接"""
        camera = self.camera
        if camera is None:
            return
        try:
            if self.grabbing:
                camera.MV_CC_StopGrabbing()
        finally:
            self.grabbing = False
            try:
                if self.opened:
                    camera.MV_CC_CloseDevice()
            finally:
                self.opened = False
                camera.MV_CC_DestroyHandle()
                self.camera = None

    def _read_once(self, output: str) -> None:
        """内部抓取方法"""
        if self.camera is None:
            raise RuntimeError("海康相机未打开")
        
        frame_out = self.sdk["MV_FRAME_OUT"]()
        memset(byref(frame_out), 0, sizeof(frame_out))
        
        ret = self.camera.MV_CC_GetImageBuffer(frame_out, self.capture_timeout_ms)
        _check_ret(ret, self.sdk["MV_OK"], "MV_CC_GetImageBuffer")
        
        try:
            image = _frame_to_image(self.camera, frame_out, self.sdk)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            image.save(output, format="JPEG", quality=90)
        finally:
            self.camera.MV_CC_FreeImageBuffer(frame_out)

    def _read_once_frame(self) -> np.ndarray:
        """内部抓取方法，返回 BGR 帧"""
        if self.camera is None:
            raise RuntimeError("海康相机未打开")

        frame_out = self.sdk["MV_FRAME_OUT"]()
        memset(byref(frame_out), 0, sizeof(frame_out))

        ret = self.camera.MV_CC_GetImageBuffer(frame_out, self.capture_timeout_ms)
        _check_ret(ret, self.sdk["MV_OK"], "MV_CC_GetImageBuffer")

        try:
            image = _frame_to_image(self.camera, frame_out, self.sdk)
            # PIL RGB -> OpenCV BGR
            rgb = np.array(image)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        finally:
            self.camera.MV_CC_FreeImageBuffer(frame_out)

    def _reconnect(self) -> None:
        """重连"""
        try:
            self.close()
        except Exception as exc:
            print(f"[海康] 重连清理失败: {exc}")


def open_camera(config: dict = None) -> HikSdkCamera:
    """创建并打开海康相机连接"""
    camera = HikSdkCamera(config)
    camera.open()
    return camera


def capture_frame(output_path: str, config: dict = None) -> bool:
    """一次性抓取（兼容旧接口）"""
    camera = HikSdkCamera(config)
    try:
        return camera.read(output_path)
    finally:
        camera.close()


def _select_device_index(device_list, camera_index: int) -> int:
    """选择设备索引"""
    index = int(camera_index)
    if index < 0 or index >= int(device_list.nDeviceNum):
        raise RuntimeError(f"索引 {index} 超出范围，设备数: {device_list.nDeviceNum}")
    return index


def _set_trigger_off(camera, ok_code: int) -> None:
    """关闭触发模式"""
    if hasattr(camera, "MV_CC_SetEnumValue"):
        ret = camera.MV_CC_SetEnumValue("TriggerMode", 0)
        if ret != ok_code:
            print(f"[海康] 警告: 设置 TriggerMode 失败, ret=0x{ret:x}")


def _frame_to_image(camera, frame_out, sdk) -> Image.Image:
    """将帧数据转换为 PIL Image"""
    frame_info = frame_out.stFrameInfo
    width = int(frame_info.nWidth)
    height = int(frame_info.nHeight)
    pixel_type = int(frame_info.enPixelType)
    frame_len = int(frame_info.nFrameLen)
    
    # RGB8 直接转换
    if pixel_type == int(sdk["PixelType_Gvsp_RGB8_Packed"]):
        data = bytes(cast(frame_out.pBufAddr, POINTER(c_ubyte * frame_len)).contents)
        return Image.frombytes("RGB", (width, height), data)
    
    # BGR8 转换
    if pixel_type == int(sdk["PixelType_Gvsp_BGR8_Packed"]):
        data = bytes(cast(frame_out.pBufAddr, POINTER(c_ubyte * frame_len)).contents)
        return Image.frombytes("RGB", (width, height), data, "raw", "BGR")
    
    # 其他格式需要转换
    converted = _convert_to_bgr8(camera, frame_out, sdk)
    return Image.frombytes("RGB", (width, height), converted, "raw", "BGR")


def _convert_to_bgr8(camera, frame_out, sdk) -> bytes:
    """转换为 BGR8 格式"""
    frame_info = frame_out.stFrameInfo
    width = int(frame_info.nWidth)
    height = int(frame_info.nHeight)
    dst_size = width * height * 3
    dst_buffer = (c_ubyte * dst_size)()
    
    convert_param = sdk["MV_PIXEL_CONVERT_PARAM"]()
    memset(byref(convert_param), 0, sizeof(convert_param))
    convert_param.nWidth = width
    convert_param.nHeight = height
    convert_param.pSrcData = frame_out.pBufAddr
    convert_param.nSrcDataLen = int(frame_info.nFrameLen)
    convert_param.enSrcPixelType = frame_info.enPixelType
    convert_param.enDstPixelType = sdk["PixelType_Gvsp_BGR8_Packed"]
    convert_param.pDstBuffer = dst_buffer
    convert_param.nDstBufferSize = dst_size
    
    ret = camera.MV_CC_ConvertPixelType(convert_param)
    _check_ret(ret, sdk["MV_OK"], "MV_CC_ConvertPixelType")
    return bytes(dst_buffer[: int(convert_param.nDstLen)])


def _check_ret(ret: int, ok_code: int, api_name: str) -> None:
    """检查返回码"""
    if ret != ok_code:
        raise RuntimeError(f"{api_name} 失败, ret=0x{ret & 0xFFFFFFFF:x}")
