"""REPL sessions stop before Python shuts down rendering executors."""
from types import SimpleNamespace
import pytest
from vector_os_nano.vcli import cli
from vector_os_nano.sim import session


@pytest.mark.parametrize('failure', [None, RuntimeError('repl failed'), SystemExit(143)])
def test_repl_stops_live_sessions_before_returning(monkeypatch, failure):
    events = []
    monkeypatch.setattr(cli, 'parse_args', lambda _: SimpleNamespace(verbose=False, print_prompt=None))
    monkeypatch.setattr(cli, '_maybe_reexec_under_mjpython', lambda _: None)
    monkeypatch.setattr(cli, '_ensure_sigint_under_mjpython', lambda: None)
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(cli, '_build', lambda *a, **k: object())
    monkeypatch.setattr(session, 'stop_all', lambda: events.append('stopped'))
    def repl(_):
        events.append('repl')
        if failure:
            raise failure
    monkeypatch.setattr(cli, 'run_repl', repl)
    if failure:
        with pytest.raises(type(failure)):
            cli.main([])
    else:
        cli.main([])
    assert events == ['repl', 'stopped']
