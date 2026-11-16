"""Collect Code_Object Chat Completions SSE without exposing partial code."""
import logging
import os
import time
from types import SimpleNamespace

logger = logging.getLogger("LLMClient")


async def collect_chat_completion(client, create_kwargs):
    """Return the existing response shape after a complete streamed choice."""
    if os.environ.get("SCENECODE_CODE_OBJECT_API", "chat") == "responses":
        return await collect_responses_completion(client, create_kwargs)
    request = dict(create_kwargs)
    reasoning_effort = os.environ.get("SCENECODE_CODE_OBJECT_REASONING_EFFORT")
    if reasoning_effort:
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("Code_Object reasoning effort must be low, medium or high")
        request["reasoning_effort"] = reasoning_effort
    started = time.monotonic()
    stream = await client.chat.completions.create(
        **request, stream=True, stream_options={"include_usage": True}
    )
    content = []
    calls = {}
    usage = None
    finish_reason = None
    first_chunk = True
    chunk_count = 0
    try:
        async for chunk in stream:
            chunk_count += 1
            if first_chunk:
                logger.info("Code_Object SSE first event: model=%s elapsed=%.2fs",
                            request["model"], time.monotonic() - started)
                first_chunk = False
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                if choice.index != 0:
                    continue
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    content.append(delta.content)
                for tool in delta.tool_calls or []:
                    call = calls.setdefault(tool.index, {"name": "", "arguments": ""})
                    if tool.function is not None:
                        call["name"] += tool.function.name or ""
                        call["arguments"] += tool.function.arguments or ""
        if finish_reason not in ("stop", "tool_calls"):
            raise RuntimeError(
                f"Code_Object SSE incomplete response: finish_reason={finish_reason!r}; "
                "partial output discarded"
            )
        if finish_reason == "tool_calls" and not calls:
            raise RuntimeError("Code_Object SSE ended with missing tool calls")
        if any(not call["name"] for call in calls.values()):
            raise RuntimeError("Code_Object SSE ended with an unnamed tool call")
        logger.info("Code_Object SSE complete: model=%s effort=%s elapsed=%.2fs chunks=%d finish=%s",
                    request["model"], reasoning_effort or "provider_default",
                    time.monotonic() - started, chunk_count, finish_reason)
        message = SimpleNamespace(
            content="".join(content),
            tool_calls=[SimpleNamespace(function=SimpleNamespace(**calls[i]))
                        for i in sorted(calls)],
        )
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)])
    finally:
        # Includes cancellation and interrupted streams: never leave a socket open.
        await stream.close()


def responses_request(create_kwargs):
    """Keep image inputs and tool history when adapting the API envelope."""
    inputs = []
    for original in create_kwargs["messages"]:
        message = dict(original)
        if message["role"] == "tool":
            inputs.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                           "output": message["content"]})
            continue
        calls = message.pop("tool_calls", [])
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block["type"] == "text":
                    blocks.append({"type": "input_text", "text": block["text"]})
                elif block["type"] == "image_url":
                    image = block["image_url"]
                    blocks.append({"type": "input_image", "image_url": image["url"],
                                   "detail": image.get("detail", "auto")})
                else:
                    raise ValueError(f"unsupported SceneCode image input block: {block['type']}")
            message["content"] = blocks
        if content:
            inputs.append(message)
        for call in calls:
            inputs.append({"type": "function_call", "call_id": call["id"],
                           "name": call["function"]["name"],
                           "arguments": call["function"]["arguments"]})
    request = {"model": create_kwargs["model"], "input": inputs,
               "max_output_tokens": create_kwargs.get("max_completion_tokens", create_kwargs.get("max_tokens", 40960))}
    if create_kwargs.get("tools"):
        request["tools"] = [{"type": "function", **item["function"]} for item in create_kwargs["tools"]]
    if "tool_choice" in create_kwargs:
        choice = create_kwargs["tool_choice"]
        request["tool_choice"] = ({"type": "function", "name": choice["function"]["name"]}
                                  if isinstance(choice, dict) else choice)
    effort = os.environ.get("SCENECODE_CODE_OBJECT_REASONING_EFFORT")
    if effort:
        request["reasoning"] = {"effort": effort}
    return request


async def collect_responses_completion(client, create_kwargs):
    request = responses_request(create_kwargs)
    stream_mode=os.environ.get("SCENECODE_CODE_OBJECT_RESPONSES_STREAM","1")
    if stream_mode not in {"0","1"}:
        raise ValueError("Code_Object Responses streaming mode must be 0 or 1")
    transport_options = {}
    read_timeout = os.environ.get("SCENECODE_CODE_OBJECT_READ_TIMEOUT", "")
    if read_timeout:
        if read_timeout != "none":
            raise ValueError("Code_Object read timeout override must be 'none'")
        import httpx
        transport_options["timeout"] = httpx.Timeout(connect=30, read=None, write=30, pool=30)
    started = time.monotonic()
    event_counts = {}
    terminal_type = None
    http_status = None
    try:
        if stream_mode=="0":
            response = await client.responses.create(**request, **transport_options)
        else:
            logger.info("Code_Object Responses stream start: model=%s",request['model'])
            stream=await client.responses.create(**request,stream=True, **transport_options)
            http_status = getattr(getattr(stream, "response", None), "status_code", None)
            response=None
            try:
                async for event in stream:
                    event_counts[event.type] = event_counts.get(event.type, 0) + 1
                    if event.type in {"response.completed","response.incomplete","response.failed"}:
                        terminal_type = event.type
                        response=event.response
                        break
                    if event.type=="error":
                        terminal_type = event.type
                        raise RuntimeError("Code_Object Responses stream error; partial output discarded")
            finally:
                await stream.close()
            if response is None:
                raise RuntimeError("Code_Object Responses stream ended without a completed response; partial output discarded")
        if response.status != "completed":
            raise RuntimeError(f"Code_Object Responses incomplete: status={response.status!r}; partial output discarded")
    except Exception as error:
        http_status = getattr(error, "status_code", http_status)
        raise
    finally:
        logger.info(
            "Code_Object Responses transport: model=%s streaming=%s elapsed=%.2fs http_status=%s events=%s terminal=%s",
            request["model"], stream_mode, time.monotonic() - started,
            http_status, dict(sorted(event_counts.items())), terminal_type,
        )
    calls = []
    for output in response.output:
        if output.type == "function_call":
            calls.append(SimpleNamespace(function=SimpleNamespace(name=output.name, arguments=output.arguments)))
    if not response.output_text and not calls:
        raise RuntimeError("Code_Object Responses completed without text or tool calls")
    usage = response.usage
    normalized_usage = (SimpleNamespace(prompt_tokens=usage.input_tokens, completion_tokens=usage.output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=getattr(usage.input_tokens_details, "cached_tokens", 0)))
        if usage else None)
    logger.info("Code_Object Responses complete: model=%s streaming=%s image_inputs=%d", request["model"],stream_mode,
        sum(block.get("type") == "input_image" for msg in request["input"]
            for block in msg.get("content", []) if isinstance(block, dict)))
    return SimpleNamespace(usage=normalized_usage, choices=[SimpleNamespace(message=SimpleNamespace(
        content=response.output_text, tool_calls=calls))])
