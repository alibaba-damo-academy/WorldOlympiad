"""
Image Utilities
图像处理工具函数
"""

import os
import base64
from typing import Tuple, Optional
from pathlib import Path


def encode_image_base64(image_path: str) -> str:
    """
    将图片编码为 base64
    
    Args:
        image_path: 图片路径
        
    Returns:
        base64 编码的字符串
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_data_url(image_path: str) -> str:
    """
    获取图片的 data URL
    
    Args:
        image_path: 图片路径
        
    Returns:
        data URL 字符串
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    mime_type = mime_types.get(ext, "image/png")
    base64_image = encode_image_base64(image_path)
    return f"data:{mime_type};base64,{base64_image}"


def load_image(image_path: str):
    """
    加载图片
    
    Args:
        image_path: 图片路径
        
    Returns:
        PIL Image 对象
    """
    try:
        from PIL import Image
        return Image.open(image_path)
    except ImportError:
        raise ImportError("Please install Pillow: pip install Pillow")


def resize_image(
    image_path: str,
    output_path: str,
    size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> str:
    """
    调整图片大小
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        size: 目标尺寸 (width, height)
        keep_aspect_ratio: 是否保持宽高比
        
    Returns:
        输出路径
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Please install Pillow: pip install Pillow")
    
    img = Image.open(image_path)
    
    if keep_aspect_ratio:
        img.thumbnail(size, Image.LANCZOS)
    else:
        img = img.resize(size, Image.LANCZOS)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    img.save(output_path)
    return output_path


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    获取图片尺寸
    
    Args:
        image_path: 图片路径
        
    Returns:
        (width, height)
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        return img.size
    except ImportError:
        raise ImportError("Please install Pillow: pip install Pillow")


def convert_image_format(
    image_path: str,
    output_path: str,
    format: str = "PNG"
) -> str:
    """
    转换图片格式
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        format: 目标格式
        
    Returns:
        输出路径
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Please install Pillow: pip install Pillow")
    
    img = Image.open(image_path)
    
    # 如果图片有透明通道且目标格式不支持，转换为 RGB
    if img.mode == "RGBA" and format.upper() == "JPEG":
        img = img.convert("RGB")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path, format=format)
    
    return output_path


def create_thumbnail(
    image_path: str,
    output_path: str,
    max_size: int = 256
) -> str:
    """
    创建缩略图
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        max_size: 最大边长
        
    Returns:
        输出路径
    """
    return resize_image(
        image_path=image_path,
        output_path=output_path,
        size=(max_size, max_size),
        keep_aspect_ratio=True
    )
