"""An in-flight CLI run leaves readable evidence before completion or timeout."""
from tests.harness import pty_cli


def test_pty_capture_streams_exact_bytes_and_does_not_reuse_an_old_run(tmp_path, monkeypatch):
    path = tmp_path/'run'/'transcript.raw'
    chunks = iter([b'\x1b[31mworking\r', b'failed\n', None])
    monkeypatch.setattr(pty_cli, '_read_chunk', lambda *a: next(chunks))
    tty = pty_cli._Pty(-1, path)
    tty.pump(0.)
    assert path.read_bytes() == b'\x1b[31mworking\r'
    tty.pump(0.)
    tty.pump(0.)
    assert tty.eof and path.read_bytes() == bytes(tty.buf)
    pty_cli._Pty(-1, path)
    assert path.read_bytes() == b''
