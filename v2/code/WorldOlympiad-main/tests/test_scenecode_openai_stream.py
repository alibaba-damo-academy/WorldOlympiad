import asyncio
from types import SimpleNamespace

import pytest

from autoresearch.scenecode_openai_stream import collect_chat_completion


class _Stream:
    def __init__(self):
        self.closed = False

    def __aiter__(self):
        async def chunks():
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                index=0, finish_reason="stop", delta=SimpleNamespace(
                    content="OK", tool_calls=None))])
        return chunks()

    async def close(self):
        self.closed = True


class _Client:
    def __init__(self):
        self.requests = []
        self.stream = _Stream()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.stream


def test_code_object_effort_is_explicit_only_for_a_frozen_trial(monkeypatch):
    client = _Client()
    kwargs = {"model": "registered-model", "messages": [{"role": "user", "content": "OK"}]}
    result = asyncio.run(collect_chat_completion(client, kwargs))
    assert result.choices[0].message.content == "OK"
    assert "reasoning_effort" not in client.requests[0]
    assert client.stream.closed
    monkeypatch.setenv("SCENECODE_CODE_OBJECT_REASONING_EFFORT", "medium")
    client = _Client()
    asyncio.run(collect_chat_completion(client, kwargs))
    assert client.requests[0]["reasoning_effort"] == "medium"
    assert kwargs == {"model": "registered-model", "messages": [{"role": "user", "content": "OK"}]}
    monkeypatch.setenv("SCENECODE_CODE_OBJECT_REASONING_EFFORT", "invalid")
    with pytest.raises(ValueError, match="reasoning effort"):
        asyncio.run(collect_chat_completion(_Client(), kwargs))


def test_responses_adapter_preserves_image_and_function_history():
    from autoresearch.scenecode_openai_stream import responses_request
    original={'model':'gpt-5.6-sol','max_tokens':4096,'messages':[
        {'role':'user','content':[{'type':'text','text':'Build this object'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,AA','detail':'high'}}]},
        {'role':'assistant','content':None,'tool_calls':[{'id':'call_1','function':{'name':'inspect','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'call_1','content':'ok'}],
        'tools':[{'type':'function','function':{'name':'inspect','parameters':{'type':'object'}}}]}
    request=responses_request(original)
    assert request['input'][0]['content'][1]['image_url']=='data:image/png;base64,AA'
    assert request['input'][1]['type']=='function_call'
    assert request['input'][2]=={'type':'function_call_output','call_id':'call_1','output':'ok'}
    assert request['max_output_tokens']==4096
    assert request['tools'][0]['name']=='inspect'


def test_responses_partial_output_is_never_executed(monkeypatch):
    monkeypatch.setenv('SCENECODE_CODE_OBJECT_RESPONSES_STREAM','0')
    from autoresearch.scenecode_openai_stream import collect_responses_completion
    class Responses:
        async def create(self, **kwargs):
            return SimpleNamespace(status='incomplete',output_text='partial code',output=[])
    client=SimpleNamespace(responses=Responses())
    with pytest.raises(RuntimeError,match='partial output discarded'):
        asyncio.run(collect_responses_completion(client,{'model':'test','messages':[{'role':'user','content':'hello'}]}))


@pytest.mark.parametrize('terminal',['response.completed','response.incomplete','response.failed','error',None])
def test_responses_stream_requires_terminal_completion_and_closes_connection(terminal, caplog):
    import logging
    caplog.set_level(logging.INFO, logger='LLMClient')
    from autoresearch.scenecode_openai_stream import collect_responses_completion
    response=SimpleNamespace(status='completed' if terminal=='response.completed' else 'incomplete',
        output_text='complete code',output=[],usage=None)
    class Stream:
        closed=False
        response=SimpleNamespace(status_code=200)
        def __aiter__(self):
            async def events():
                yield SimpleNamespace(type='response.output_text.delta',delta='partial code')
                if terminal:yield SimpleNamespace(type=terminal,response=response)
            return events()
        async def close(self):self.closed=True
    stream=Stream();requests=[]
    class Responses:
        async def create(self,**kwargs):requests.append(kwargs);return stream
    client=SimpleNamespace(responses=Responses());request={'model':'test','messages':[{'role':'user','content':'hello'}]}
    if terminal=='response.completed':
        result=asyncio.run(collect_responses_completion(client,request))
        assert result.choices[0].message.content=='complete code'
    else:
        with pytest.raises(RuntimeError,match='partial output discarded'):
            asyncio.run(collect_responses_completion(client,request))
    assert stream.closed and requests[0]['stream'] is True
    diagnostics = [record.message for record in caplog.records if 'Responses transport:' in record.message]
    assert len(diagnostics) == 1
    assert 'http_status=200' in diagnostics[0] and 'elapsed=' in diagnostics[0]
    assert "'response.output_text.delta': 1" in diagnostics[0]
    assert ('terminal='+str(terminal)) in diagnostics[0]
    assert 'partial code' not in diagnostics[0] and 'complete code' not in diagnostics[0]


def test_responses_empty_stream_reports_zero_events_without_accepting_output(caplog):
    import logging
    from autoresearch.scenecode_openai_stream import collect_responses_completion
    caplog.set_level(logging.INFO, logger='LLMClient')
    class Stream:
        response=SimpleNamespace(status_code=200)
        closed=False
        def __aiter__(self):
            async def events():
                if False: yield None
            return events()
        async def close(self): self.closed=True
    stream=Stream()
    class Responses:
        async def create(self, **kwargs): return stream
    with pytest.raises(RuntimeError, match='ended without a completed response'):
        asyncio.run(collect_responses_completion(SimpleNamespace(responses=Responses()), {'model':'test','messages':[]}))
    assert stream.closed
    assert 'events={} terminal=None' in caplog.text


def test_responses_http_failure_preserves_exception_and_reports_status(caplog):
    import logging
    from autoresearch.scenecode_openai_stream import collect_responses_completion
    caplog.set_level(logging.INFO, logger='LLMClient')
    class Unavailable(RuntimeError): status_code=503
    error=Unavailable('provider error')
    class Responses:
        async def create(self, **kwargs): raise error
    with pytest.raises(Unavailable) as caught:
        asyncio.run(collect_responses_completion(SimpleNamespace(responses=Responses()), {'model':'test','messages':[]}))
    assert caught.value is error
    assert 'http_status=503 events={} terminal=None' in caplog.text
    assert 'provider error' not in caplog.text
