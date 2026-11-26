from model import qwenvl, qwenvl_server


def test_qwenvl_generation_limit_can_be_overridden_without_changing_default(monkeypatch):
    monkeypatch.setenv("QWENVL_MAX_NEW_TOKENS", "1024")
    assert qwenvl._generation_kwargs()["max_new_tokens"] == 1024
    assert qwenvl._generation_kwargs(max_new_tokens=16)["max_new_tokens"] == 16


def test_server_warmup_uses_bounded_generation_and_preserves_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(qwenvl_server, "chat_qwenvl_call", lambda **kwargs: calls.append(kwargs))

    qwenvl_server.warmup_qwenvl("local/model", timeout=123)

    assert len(calls) == 1
    assert calls[0]["model_name"] == "local/model"
    assert calls[0]["timeout"] == 123
    assert calls[0]["max_retries"] == 1
    assert calls[0]["max_new_tokens"] == qwenvl_server.WARMUP_MAX_NEW_TOKENS
    assert calls[0]["max_new_tokens"] < 1024
