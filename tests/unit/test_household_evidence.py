"""A repeated development episode must not overwrite its earlier evidence."""
import pytest
from tests.household.test_grasp import snapshot


def test_household_snapshot_refuses_an_existing_episode(tmp_path):
    out = tmp_path/'episode'; out.mkdir()
    evidence = out/'controller.patch'; evidence.write_bytes(b'original checkpoint')
    with pytest.raises(FileExistsError):
        snapshot(out, tmp_path/'absent-assets', 'world')
    assert evidence.read_bytes() == b'original checkpoint'
