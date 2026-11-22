"""All production providers obey the public call contract at the SDK boundary."""
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize('provider', ['anthropic', 'deepseek', 'openrouter', 'openai_compat'])
@pytest.mark.parametrize('broken', [False, True])
def test_public_backend_contract(provider, broken):
    from vector_os_nano.interfaces import LLMBackend, LLMResponse, LLMToolCall, TokenUsage
    from vector_os_nano.vcli.backends import create_backend

    sdk = 'anthropic.Anthropic' if provider == 'anthropic' else 'openai.OpenAI'
    with patch(sdk):
        backend = create_backend(provider, 'test-key', 'test-model', 'http://localhost:9/v1')
    assert isinstance(backend, LLMBackend)
    failure = RuntimeError('transport unavailable')
    if provider == 'anthropic':
        stream = MagicMock()
        stream.__iter__.return_value = iter([NS(type='content_block_delta', delta=NS(text='ready'))])
        stream.get_final_message.return_value = NS(
            content=[NS(type='tool_use', id='call-1', name='look', input={'query': 'cup'})],
            stop_reason='tool_use', usage=NS(input_tokens=11, output_tokens=7))
        transport = backend._client.messages.stream
        transport.return_value.__enter__.return_value = stream
    else:
        transport = backend._client.chat.completions.create
        transport.return_value = iter([
            NS(choices=[NS(finish_reason='tool_calls', delta=NS(content='ready',
                tool_calls=[NS(index=0, id='call-1', function=NS(name='look', arguments='{"query":"cup"}'))]))]),
            NS(choices=[], usage=NS(prompt_tokens=11, completion_tokens=7)),
        ])
    if broken:
        transport.side_effect = failure
    chunks=[]
    args=dict(messages=[{'role':'user','content':'look at the cup'}], tools=[], system=[],
              max_tokens=32, on_text=chunks.append, on_reasoning=lambda _: None)
    if broken:
        with pytest.raises(RuntimeError) as exc:
            backend.call(**args)
        assert exc.value is failure
        assert not chunks
    else:
        response=backend.call(**args)
        assert response == LLMResponse('ready', [LLMToolCall('call-1','look',{'query':'cup'})],
                                       'tool_use', TokenUsage(input_tokens=11, output_tokens=7))
        assert chunks == ['ready']
