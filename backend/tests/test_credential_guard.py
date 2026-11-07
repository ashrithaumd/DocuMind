"""scripts/check_no_keys.sh catches a key by name and by content, and passes clean."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("git")), reason="needs bash and git"
)


def _guard(tmp_path, files):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/check_no_keys.sh", tmp_path / "scripts")
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return subprocess.run(
        ["bash", "scripts/check_no_keys.sh"], cwd=tmp_path, capture_output=True, text=True
    )


def test_clean_repo_passes(tmp_path):
    assert _guard(tmp_path, {"a.json": '{"type": "other"}'}).returncode == 0


def test_key_by_name_fails(tmp_path):
    r = _guard(tmp_path, {"my-firebase-adminsdk-x.json": "{}"})
    assert r.returncode == 1 and "adminsdk" in r.stdout


# Joined at runtime so this file never holds the literal patterns itself.
@pytest.mark.parametrize("body", [
    '{"type": "service' + '_account"}',
    '{"private' + '_key": "x"}',
])
def test_renamed_key_fails_by_content(tmp_path, body):
    r = _guard(tmp_path, {"innocent.txt": body})
    assert r.returncode == 1 and "innocent.txt" in r.stdout


def test_guard_has_no_exemptions_and_passes_on_this_repo():
    # Only the script itself may be skipped; no test or doc is exempt.
    script = (ROOT / "scripts/check_no_keys.sh").read_text()
    assert script.count("grep -zvxF") == 1 and "test_" not in script
    r = subprocess.run(["bash", "scripts/check_no_keys.sh"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
