"""The release gate preserves each attempt and needs three green e2e runs."""
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('failed', ['', 'e2e-2'])
def test_all_attempts_are_isolated_and_every_failure_counts(tmp_path, failed):
    source = (Path(__file__).resolve().parents[2]/'scripts/gate.sh').read_text()
    script = tmp_path/'scripts/gate.sh'; script.parent.mkdir()
    # Exercise the real orchestration with cheap child tiers, without starting simulators.
    script.write_text(source.partition('# The per-test timeout;')[0]+'''\n
mkdir -p "$GATE_OUT"
printf '%s\\n' "$TIER" "$@" > "$GATE_OUT/call.txt"
[ "${GATE_OUT##*/}" != "$FAIL_ATTEMPT" ]
''')
    script.chmod(0o755)
    out = tmp_path/'evidence'
    result = subprocess.run([str(script), 'all', '-k', 'one expression'], capture_output=True,
        env={**os.environ, 'GATE_OUT': str(out), 'FAIL_ATTEMPT': failed})
    expected = ['hermetic', 'sim', 'e2e-1', 'e2e-2', 'e2e-3']
    assert sorted(p.name for p in out.iterdir()) == sorted(expected)
    for attempt in expected:
        assert (out/attempt/'call.txt').read_text().splitlines() == [attempt.split('-')[0], '-k', 'one expression']
    assert result.returncode == bool(failed)
    again = subprocess.run([str(script), 'all'], capture_output=True,
        env={**os.environ, 'GATE_OUT': str(out), 'FAIL_ATTEMPT': ''})
    assert again.returncode != 0
    assert (out/'e2e-1/call.txt').read_text().splitlines()[-1] == 'one expression'
