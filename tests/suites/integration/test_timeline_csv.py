from pathlib import Path

from ser.config import Config
from ser.utils.timeline_utils import save_timeline_to_csv


def test_save_timeline_to_csv_writes_headers_and_rows(tmp_path, monkeypatch):
    target_dir = tmp_path / "transcripts"
    target_dir.mkdir()
    monkeypatch.setitem(Config.TIMELINE_CONFIG, "folder", target_dir.as_posix())

    timeline = [(0.1234, "happy", "Hello world")]

    csv_path = save_timeline_to_csv(timeline, "sample.wav")

    written = Path(csv_path)
    assert written.exists()
    rows = written.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "Time (s),Emotion,Speech"
    assert rows[1] == "0.12,happy,Hello world"
