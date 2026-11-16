"""
LLM Client
LLM 客户端封装

Prompt Caching 说明：
- OpenAI: 自动缓存（对 >1024 token 的 prompt 前缀），无需额外参数
- Anthropic: 需要显式添加 cache_control 到 system message，本模块自动处理
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TokenUsageTracker:
    """
    统计每次 API 调用的输入/输出 token 数量，并写入日志与本地文件。
    文件保存在 log_dir/token_usage/ 下，按日分文件：usage_YYYYMMDD.jsonl
    """

    def __init__(self, log_dir: str, logger_name: str = "LLMClient"):
        self.log_dir = Path(log_dir)
        self.usage_dir = self.log_dir / "token_usage"
        self.usage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(logger_name)
        self._total_input = 0
        self._total_output = 0
        self._call_count = 0

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录单次调用的 token 用量并写入日志与文件。"""
        self._call_count += 1
        self._total_input += input_tokens
        self._total_output += output_tokens
        total = input_tokens + output_tokens

        # 提取缓存相关信息用于日志
        cache_info = ""
        if extra:
            # OpenAI: cached_tokens
            if "cached_tokens" in extra:
                cache_info = f" cached={extra['cached_tokens']}"
            # Anthropic: cache_creation_tokens, cache_read_tokens
            if "cache_creation_tokens" in extra:
                cache_info += f" cache_write={extra['cache_creation_tokens']}"
            if "cache_read_tokens" in extra:
                cache_info += f" cache_read={extra['cache_read_tokens']}"

        self.logger.info(
            "API usage: model=%s input_tokens=%d output_tokens=%d total=%d%s (call #%d)",
            model,
            input_tokens,
            output_tokens,
            total,
            cache_info,
            self._call_count,
        )

        record = {
            "ts": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "call_index": self._call_count,
        }
        if extra:
            record["extra"] = extra

        file_path = self.usage_dir / f"usage_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            self.logger.warning("Failed to write token usage to %s: %s", file_path, e)

    def get_summary(self) -> Dict[str, Any]:
        """返回当前会话的累计用量。"""
        return {
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_tokens": self._total_input + self._total_output,
            "call_count": self._call_count,
        }


class LLMClient(ABC):
    """LLM 客户端基类"""
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        发送聊天消息
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            响应文本
        """
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI 客户端
    
    Prompt Caching:
    - OpenAI 自动缓存 >1024 token 的 prompt 前缀
    - 静态内容（system prompt）应放在消息最前面以最大化缓存命中
    - 缓存命中时输入 token 成本降低 50%，延迟降低最多 80%
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-vision-preview",
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 40960,
        usage_tracker: Optional[TokenUsageTracker] = None,
    ):
        """
        初始化 OpenAI 客户端

        Args:
            api_key: API Key，默认从环境变量读取
            model: 模型名称
            api_base: API 基础 URL
            temperature: 温度参数
            max_tokens: 最大 token 数
            usage_tracker: 可选，用于统计并保存 token 用量
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage_tracker = usage_tracker

        self._client = None
    
    def _get_client(self):
        """获取或创建 OpenAI 客户端（单例模式复用连接）"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                
                kwargs = {"api_key": self.api_key}
                if self.api_base:
                    kwargs["base_url"] = self.api_base
                
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        
        return self._client
    
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ):
        """
        发送聊天消息

        OpenAI 自动对 prompt 前缀进行缓存，无需额外配置。
        确保 system prompt 在 messages 最前面以获得最佳缓存效果。

        当传入 tools/tool_choice 且 API 返回 tool_calls 时，返回
        ``{"content": str, "tool_calls": [{"name": ..., "arguments": ...}, ...]}``；
        否则返回纯文本 ``str``。
        """
        client = self._get_client()
        model = kwargs.get("model", self.model)
        model_lc = str(model).lower()
        need_thought_signature = "gemini-3" in model_lc

        # 一些 OpenAI 兼容接口（包括代理）要求 text content block 不能为空，
        # 如果 content 是空字符串或仅包含空白，会直接 400。
        # 这里在发送前做一次轻量级清洗，对所有消息的文本内容做占位填充。
        safe_messages: List[Dict[str, Any]] = []
        for msg in messages:
            msg_copy = deepcopy(msg)
            content = msg_copy.get("content")

            # Gemini-3: 某些 OpenAI 兼容接口要求第一个 functionCall part 带 thought_signature，
            # 后续并行 call 不附加（这是 Gemini-3 的正确行为）。
            if need_thought_signature and isinstance(msg_copy.get("tool_calls"), list):
                for tc_idx, tc in enumerate(msg_copy["tool_calls"]):
                    if not isinstance(tc, dict):
                        continue
                    if tc_idx == 0:
                        # 仅第一个 functionCall part 附加 thought_signature
                        extra = tc.get("extra_content")
                        if not isinstance(extra, dict):
                            extra = {}
                            tc["extra_content"] = extra
                        google = extra.get("google")
                        if not isinstance(google, dict):
                            google = {}
                            extra["google"] = google
                        google.setdefault("thought_signature", "skip_thought_signature_validator")
                    else:
                        # 后续并行 call：清除 thought_signature 以避免 400
                        extra = tc.get("extra_content")
                        if isinstance(extra, dict):
                            google = extra.get("google")
                            if isinstance(google, dict):
                                google.pop("thought_signature", None)
                                if not google:
                                    extra.pop("google", None)
                            if not extra:
                                tc.pop("extra_content", None)
                        tc.pop("thought_signature", None)
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            fn.pop("thought_signature", None)

            # 简单字符串形式
            if isinstance(content, str):
                if not content.strip():
                    # 占位文本，避免空 content 触发服务端校验错误
                    msg_copy["content"] = " "

            # content blocks 形式（包含 text/image 等）
            elif isinstance(content, list):
                new_blocks: List[Dict[str, Any]] = []
                for block in content:
                    block_copy = deepcopy(block)
                    if block_copy.get("type") == "text":
                        text = block_copy.get("text", "")
                        if not isinstance(text, str) or not text.strip():
                            block_copy["text"] = " "
                    # Gemini-3: content blocks 中的 function/tool_call 类型 block
                    # 仅在需要时（即整个 content 只有一个这样的 block 且是首个）才注入
                    # 但一般 content blocks 不包含 functionCall，这里做保守兜底
                    if need_thought_signature and isinstance(block_copy, dict):
                        btype = str(block_copy.get("type", "")).lower()
                        if btype in ("function", "function_call", "tool_call", "tool_use"):
                            # 对 content block 中的 function-like block 不再统一注入
                            # thought_signature，仅保留已有的
                            pass
                    new_blocks.append(block_copy)
                msg_copy["content"] = new_blocks

            safe_messages.append(msg_copy)

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": safe_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # 转发 tools / tool_choice（原生 function calling）
        if kwargs.get("tools"):
            create_kwargs["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            create_kwargs["tool_choice"] = kwargs["tool_choice"]

        response = await client.chat.completions.create(**create_kwargs)

        if self.usage_tracker and getattr(response, "usage", None):
            u = response.usage
            # 记录缓存命中情况（如果 API 返回）
            cached_tokens = 0
            if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
                cached_tokens = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
            
            self.usage_tracker.record(
                model=model,
                input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
                extra={"cached_tokens": cached_tokens} if cached_tokens > 0 else None,
            )

        msg = response.choices[0].message

        # 如果 API 返回了 tool_calls，封装成 dict 返回
        # content 和 tool_calls 字段是并列的
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,  # JSON string
                })
            return {
                "content": msg.content or "",
                "tool_calls": tool_calls,
            }

        return msg.content


class AnthropicClient(LLMClient):
    """
    Anthropic (Claude) 客户端
    
    Prompt Caching:
    - 通过 cache_control: {"type": "ephemeral"} 显式启用缓存
    - 本客户端自动为 system message 添加缓存控制
    - 缓存命中时输入 token 成本降低最多 90%，延迟降低最多 85%
    - 缓存 TTL 默认 5 分钟，每次命中刷新
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        api_base: Optional[str] = None,
        max_tokens: int = 40960,
        usage_tracker: Optional[TokenUsageTracker] = None,
        enable_cache: bool = True,
    ):
        """
        初始化 Anthropic 客户端

        Args:
            api_key: API Key
            model: 模型名称
            api_base: API 基础 URL（用于兼容 OpenAI 格式的代理）
            max_tokens: 最大 token 数
            usage_tracker: 可选，用于统计并保存 token 用量
            enable_cache: 是否启用 prompt caching（默认 True）
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.usage_tracker = usage_tracker
        self.enable_cache = enable_cache

        self._client = None
    
    def _get_client(self):
        """获取或创建 Anthropic 客户端（单例模式复用连接）"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                
                kwargs = {"api_key": self.api_key}
                if self.api_base:
                    kwargs["base_url"] = self.api_base
                
                self._client = AsyncAnthropic(**kwargs)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        
        return self._client
    
    def _add_cache_control_to_system(
        self, 
        system_content: Any
    ) -> List[Dict[str, Any]]:
        """
        为 system message 内容添加 cache_control
        
        Args:
            system_content: system message 的内容（字符串或 content blocks 列表）
            
        Returns:
            带 cache_control 的 content blocks 列表
        """
        if not self.enable_cache:
            # 不启用缓存时，返回原格式
            if isinstance(system_content, str):
                return [{"type": "text", "text": system_content}]
            return system_content
        
        # 将 system content 转为 blocks 格式并添加 cache_control
        if isinstance(system_content, str):
            return [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        elif isinstance(system_content, list):
            # 对最后一个 block 添加 cache_control（Anthropic 建议在缓存边界处添加）
            blocks = deepcopy(system_content)
            if blocks:
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
            return blocks
        else:
            return [
                {
                    "type": "text",
                    "text": str(system_content),
                    "cache_control": {"type": "ephemeral"}
                }
            ]
    
    def _add_cache_control_to_user_images(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        为 user message 中的静态图片添加 cache_control
        
        对于包含 image 的 user message，在图片内容块上添加缓存控制，
        以便相同图片的多次请求可以复用缓存。
        """
        if not self.enable_cache:
            return messages
        
        result = []
        for msg in messages:
            if msg.get("role") != "user":
                result.append(msg)
                continue
            
            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg)
                continue
            
            # 检查是否包含图片
            new_content = []
            for block in content:
                block_copy = deepcopy(block)
                # 对图片块添加 cache_control
                if block_copy.get("type") in ("image", "image_url"):
                    block_copy["cache_control"] = {"type": "ephemeral"}
                new_content.append(block_copy)
            
            result.append({**msg, "content": new_content})
        
        return result
    
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        发送聊天消息
        
        自动为 system message 和图片内容添加 cache_control 以启用 prompt caching。
        """
        client = self._get_client()
        
        # 提取并处理 system message
        system_content = None
        converted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                converted_messages.append(msg)
        
        # 为 user messages 中的图片添加缓存控制
        converted_messages = self._add_cache_control_to_user_images(converted_messages)
        
        model = kwargs.get("model", self.model)
        
        # 构建 API 调用参数
        api_kwargs = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # 添加 system message（带 cache_control）
        if system_content:
            api_kwargs["system"] = self._add_cache_control_to_system(system_content)
        
        response = await client.messages.create(**api_kwargs)

        if self.usage_tracker and getattr(response, "usage", None):
            u = response.usage
            # 记录缓存命中情况
            cache_creation_tokens = getattr(u, "cache_creation_input_tokens", 0) or 0
            cache_read_tokens = getattr(u, "cache_read_input_tokens", 0) or 0
            
            extra = {}
            if cache_creation_tokens > 0:
                extra["cache_creation_tokens"] = cache_creation_tokens
            if cache_read_tokens > 0:
                extra["cache_read_tokens"] = cache_read_tokens
            
            self.usage_tracker.record(
                model=model,
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                extra=extra if extra else None,
            )

        return response.content[0].text


def create_llm_client(
    provider: str = "openai",
    usage_tracker: Optional[TokenUsageTracker] = None,
    enable_cache: bool = True,
    **kwargs
) -> LLMClient:
    """
    创建 LLM 客户端
    
    Prompt Caching:
    - OpenAI: 自动对 >1024 token 的 prompt 前缀进行缓存，无需配置
    - Anthropic: 自动为 system message 和图片添加 cache_control

    Args:
        provider: 提供商名称 (openai, anthropic)
        usage_tracker: 可选，用于统计并保存 token 用量
        enable_cache: 是否启用 prompt caching（仅 Anthropic 需要，默认 True）
        **kwargs: 客户端参数

    Returns:
        LLMClient 实例
    """
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

    if usage_tracker is not None:
        kwargs["usage_tracker"] = usage_tracker
    
    # Anthropic 支持 enable_cache 参数
    if provider == "anthropic":
        kwargs["enable_cache"] = enable_cache
    
    return providers[provider](**kwargs)
