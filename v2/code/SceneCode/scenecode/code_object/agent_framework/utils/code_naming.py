"""
Code artifact naming helpers.
统一管理代码产物命名，避免生成非法 Python 标识符。
"""

import re


def sanitize_code_token(raw_name: str, fallback: str = "object") -> str:
    """
    将任意名称转为仅包含 [a-z0-9_] 的 token。
    注意：该 token 允许数字开头，调用方应在需要时添加前缀。
    """
    token = (raw_name or "").strip().lower()
    token = token.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or fallback


def get_parts_package_name(object_name: str) -> str:
    """
    生成部件包名。
    示例: 0a5a-... -> parts_0a5a_...
    """
    return f"parts_{sanitize_code_token(object_name)}"


def get_object_builder_func_name(object_name: str) -> str:
    """
    生成主脚本中的构建函数名（保证合法标识符）。
    """
    return f"create_object_{sanitize_code_token(object_name)}"
