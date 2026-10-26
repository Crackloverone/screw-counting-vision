# -*- coding: utf-8 -*-
"""
配置模块 - 统一配置管理
基于 hls 项目的 config.py 优化
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


# 项目根目录
RUNTIME_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DATA_DIR = RUNTIME_ROOT / "runtime_data"


@dataclass(slots=True)
class SourceConfig:
    """帧源配置"""
    type: str = "directory"
    watch_dir: str = str((RUNTIME_ROOT / "data").resolve())
    glob: str = "*.jpg"
    stable_delay_ms: int = 300
    live_dir: str = str((RUNTIME_DATA_DIR / "live").resolve())
    frame_interval_ms: int = 100
    camera_index: int = 0
    capture_timeout_ms: int = 1000
    serial_number: str = ""
    sdk_module: str = "hik_sdk_adapter"


@dataclass(slots=True)
class RoiRuntimeConfig:
    """ROI 区域配置"""
    enabled: bool = False
    output_dir: str = str((RUNTIME_DATA_DIR / "roi").resolve())
    debug_dir: str = str((RUNTIME_DATA_DIR / "roi_debug").resolve())


@dataclass(slots=True)
class InferenceConfig:
    """推理配置"""
    backend: str = "opencv"  # opencv / onnx / torch / ascend
    model_path: str = ""
    model_version: str = ""
    device: str = "auto"
    confidence: float = 0.5
    iou: float = 0.45
    input_size: int = 416
    timeout_ms: int = 5000
    labels_path: str = ""
    command: list[str] | str | None = None


@dataclass(slots=True)
class OutputConfig:
    """输出配置"""
    result_dir: str = str((RUNTIME_DATA_DIR / "results").resolve())
    annotated_dir: str = str((RUNTIME_DATA_DIR / "annotated").resolve())
    latest_result_path: str = str((RUNTIME_DATA_DIR / "latest_result.json").resolve())
    latest_frame_path: str = str((RUNTIME_DATA_DIR / "latest_frame.jpg").resolve())
    save_result_json: bool = True
    save_annotated: bool = True
    save_ok_images: bool = False
    save_ng_images: bool = True
    json_indent: int = 2


@dataclass(slots=True)
class SdkConfig:
    """SDK 配置"""
    base_path: str = ""  # 海康 MVS SDK 路径
    sdk_module: str = "hik_sdk_adapter"


@dataclass(slots=True)
class EdgeRuntimeConfig:
    """完整运行时配置"""
    runtime_name: str = "newls-screw-detection"
    poll_interval_ms: int = 200
    source: SourceConfig = None
    roi: RoiRuntimeConfig = None
    inference: InferenceConfig = None
    output: OutputConfig = None
    
    def __post_init__(self):
        if self.source is None:
            self.source = SourceConfig()
        if self.roi is None:
            self.roi = RoiRuntimeConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.output is None:
            self.output = OutputConfig()

    @classmethod
    def load(cls, path: str | Path) -> "EdgeRuntimeConfig":
        """从 JSON 文件加载配置"""
        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        base_dir = config_path.parent
        
        # 解析 SourceConfig
        source = SourceConfig(**payload.get("source", {}))
        source.watch_dir = str(_resolve_path(base_dir, source.watch_dir))
        source.live_dir = str(_resolve_path(base_dir, source.live_dir))
        
        # 解析 ROI
        roi = RoiRuntimeConfig(**payload.get("roi", {}))
        roi.output_dir = str(_resolve_path(base_dir, roi.output_dir))
        roi.debug_dir = str(_resolve_path(base_dir, roi.debug_dir))
        
        # 解析 Inference
        inference = InferenceConfig(**payload.get("inference", {}))
        if inference.model_path:
            inference.model_path = str(_resolve_path(base_dir, inference.model_path))
        if inference.labels_path:
            inference.labels_path = str(_resolve_path(base_dir, inference.labels_path))
        if inference.command:
            if isinstance(inference.command, str):
                inference.command = _resolve_command_string(base_dir, inference.command)
            else:
                inference.command = [_resolve_command_part(base_dir, part) for part in inference.command]
        
        # 解析 Output
        output = OutputConfig(**payload.get("output", {}))
        output.result_dir = str(_resolve_path(base_dir, output.result_dir))
        output.annotated_dir = str(_resolve_path(base_dir, output.annotated_dir))
        output.latest_result_path = str(_resolve_path(base_dir, output.latest_result_path))
        output.latest_frame_path = str(_resolve_path(base_dir, output.latest_frame_path))
        
        return cls(
            runtime_name=str(payload.get("runtime_name", "newls-screw-detection")),
            poll_interval_ms=max(50, int(payload.get("poll_interval_ms", 200))),
            source=source,
            roi=roi,
            inference=inference,
            output=output,
        )

    def ensure_dirs(self) -> None:
        """确保所有必要目录存在"""
        for path_text in (
            self.source.watch_dir,
            self.source.live_dir,
            self.roi.output_dir,
            self.roi.debug_dir,
            self.output.result_dir,
            self.output.annotated_dir,
            str(Path(self.output.latest_result_path).expanduser().resolve().parent),
            str(Path(self.output.latest_frame_path).expanduser().resolve().parent),
        ):
            Path(path_text).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "runtime_name": self.runtime_name,
            "poll_interval_ms": self.poll_interval_ms,
            "source": _config_to_dict(self.source),
            "roi": _config_to_dict(self.roi),
            "inference": _config_to_dict(self.inference),
            "output": _config_to_dict(self.output),
        }

    def save(self, path: str | Path) -> None:
        """保存配置到文件"""
        config_path = Path(path).expanduser().resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


def _resolve_path(base_dir: Path, value: str) -> Path:
    """解析路径"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_command_part(base_dir: Path, value: str) -> str:
    """解析命令部分"""
    if "{" in value or "}" in value:
        return value
    candidate = (base_dir / value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _resolve_command_string(base_dir: Path, value: str) -> str:
    """解析命令字符串"""
    parts = shlex.split(value)
    resolved_parts = [_resolve_command_part(base_dir, part) for part in parts]
    return shlex.join(resolved_parts)


def _config_to_dict(config_obj) -> dict:
    """配置对象转字典"""
    result = {}
    for f in fields(config_obj):
        value = getattr(config_obj, f.name)
        if value is not None:
            result[f.name] = value
    return result


# 便捷函数
def create_default_config() -> EdgeRuntimeConfig:
    """创建默认配置"""
    return EdgeRuntimeConfig()


def load_app_config() -> dict:
    """加载应用配置 (YAML 格式)"""
    config_path = RUNTIME_ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_app_config(config: dict) -> None:
    """保存应用配置 (YAML 格式)"""
    config_path = RUNTIME_ROOT / "config.yaml"
    import yaml
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


# ==================== 检测参数配置 ====================
class DetectionConfig:
    """检测参数配置"""
    
    def __init__(
        self,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        detect_interval: int = 3,  # 每几帧检测一次
        model_name: str = "",
        confidence_min: float = 0.1,
        confidence_max: float = 0.9,
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.detect_interval = detect_interval
        self.model_name = model_name
        self.confidence_min = confidence_min
        self.confidence_max = confidence_max
    
    def to_dict(self) -> dict:
        return {
            'confidence': self.confidence,
            'iou_threshold': self.iou_threshold,
            'detect_interval': self.detect_interval,
            'model_name': self.model_name,
            'confidence_min': self.confidence_min,
            'confidence_max': self.confidence_max,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DetectionConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__init__.__code__.co_varnames})


# 全局检测配置
_detection_config = DetectionConfig()


def get_detection_config() -> DetectionConfig:
    return _detection_config


def update_detection_config(**kwargs) -> DetectionConfig:
    """更新检测配置"""
    for key, value in kwargs.items():
        if hasattr(_detection_config, key):
            setattr(_detection_config, key, value)
    return _detection_config
    """配置日志级别
    
    Args:
        level: 日志级别，可选 DEBUG, INFO, WARNING, ERROR
    """
    import logging
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)