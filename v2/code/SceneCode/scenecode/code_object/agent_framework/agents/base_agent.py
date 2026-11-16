"""
Base Agent
所有 Agent 的基类
"""

import os
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

import yaml


class BaseAgent(ABC):
    """Agent 基类"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_client: Optional[Any] = None,
        blender_client: Optional[Any] = None
    ):
        """
        初始化 Agent
        
        Args:
            config_path: 配置文件路径
            llm_client: LLM 客户端
            blender_client: Blender MCP 客户端
        """
        self.config = self._load_config(config_path)
        self.llm_client = llm_client
        self.blender_client = blender_client
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 默认配置路径
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            self._get_default_config()
            
        return {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "model": {
                "llm": {
                    "provider": "openai",
                    "model_name": "gpt-4-vision-preview",
                    "temperature": 0.7,
                    "max_tokens": 40960
                }
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(self.__class__.__name__)
        # Prevent duplicate output through the root logger.
        logger.propagate = False

        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _validate_single_input(
        self,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
        *,
        require_exactly_one: bool = True
    ) -> str:
        """
        校验输入模式。

        Returns:
            "image" | "text" | "none"
        """
        has_image = bool(image_path and str(image_path).strip())
        has_text = bool(text_input and str(text_input).strip())

        if require_exactly_one and has_image == has_text:
            raise ValueError("Exactly one of image_path or text_input must be provided")

        if has_image:
            return "image"
        if has_text:
            return "text"
        return "none"

    def _sanitize_object_name(self, raw_name: str, fallback: str = "object") -> str:
        """将名称转为安全的 snake_case 标识符。"""
        raw_name = (raw_name or "").strip().lower()
        raw_name = raw_name.encode("ascii", "ignore").decode("ascii")
        raw_name = re.sub(r"[^a-z0-9]+", "_", raw_name)
        raw_name = re.sub(r"_+", "_", raw_name).strip("_")
        if not raw_name:
            return fallback
        if raw_name[0].isdigit():
            raw_name = f"obj_{raw_name}"
        return raw_name

    def _extract_object_name_from_text(self, text_input: Optional[str]) -> str:
        """
        从文本中提取简名（不含时间戳）。
        若无法提取，回退到通用名称。
        """
        text = (text_input or "").strip()
        lower = text.lower()

        keyword_map = [
            ("armchair", "armchair"),
            ("dining chair", "chair"),
            ("chair", "chair"),
            ("椅", "chair"),
            ("stool", "stool"),
            ("凳", "stool"),
            ("table", "table"),
            ("desk", "desk"),
            ("桌", "table"),
            ("cabinet", "cabinet"),
            ("drawer", "cabinet"),
            ("柜", "cabinet"),
            ("shelf", "shelf"),
            ("bookcase", "shelf"),
            ("架", "shelf"),
            ("bench", "bench"),
            ("长凳", "bench"),
            ("sofa", "sofa"),
            ("沙发", "sofa"),
            ("bed", "bed"),
            ("床", "bed"),
        ]
        for key, mapped in keyword_map:
            if key in lower:
                return mapped

        tokens = re.findall(r"[a-zA-Z0-9]+", lower)
        stopwords = {
            "a", "an", "the", "and", "or", "with", "for", "to", "of",
            "generate", "create", "build", "furniture", "object",
            "please", "simple", "modern", "style", "design",
        }
        core = [t for t in tokens if t not in stopwords and len(t) > 1]
        if core:
            return self._sanitize_object_name("_".join(core[:3]), fallback="object")
        return "object"
    
    def _load_prompt(self, prompt_name: str) -> str:
        """加载提示词模板"""
        prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
        
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    async def _call_llm(
        self,
        messages: list,
        images: Optional[list] = None,
        **kwargs
    ) -> str:
        """
        调用 LLM
        
        Args:
            messages: 消息列表
            images: 图片列表（base64 或 URL）
            **kwargs: 其他参数
            
        Returns:
            LLM 响应文本
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")
        
        # 如果有图片，添加到消息中
        if images:
            for i, img in enumerate(images):
                if isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
                    # 添加图片到用户消息
                    content = messages[-1].get("content", "")
                    if isinstance(content, str):
                        messages[-1]["content"] = [
                            {"type": "text", "text": content},
                            {"type": "image_url", "image_url": {"url": img}}
                        ]
                    elif isinstance(content, list):
                        content.append({"type": "image_url", "image_url": {"url": img}})
        
        # 调用 LLM
        response = await self.llm_client.chat(messages, **kwargs)
        return response
    
    async def _call_blender(self, action: str, params: Dict[str, Any]) -> Any:
        """
        调用 Blender MCP
        
        Args:
            action: 动作名称
            params: 参数字典
            
        Returns:
            Blender 执行结果
        """
        if self.blender_client is None:
            raise RuntimeError("Blender client not initialized")
        
        return await self.blender_client.execute(action, params)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的 JSON
        
        Args:
            response: LLM 响应文本
            
        Returns:
            解析后的字典
        """
        # 尝试提取 JSON 块
        import re
        
        # 匹配 ```json ... ``` 或 ``` ... ```
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON response: {response}...")
            raise ValueError("Invalid JSON response from LLM")
    
    def _save_result(self, result: Any, filepath: str) -> None:
        """保存结果到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if isinstance(result, dict):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif isinstance(result, str):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(result)
        else:
            raise TypeError(f"Unsupported result type: {type(result)}")
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """
        执行 Agent 主逻辑
        
        子类必须实现此方法
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
