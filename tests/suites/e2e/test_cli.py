import logging
import ser.__main__ as cli


def _stub_processing(monkeypatch, subtitles=None):
    monkeypatch.setattr(cli, "predict_emotions", lambda *_: [("happy", 0.0, 1.0)])
    monkeypatch.setattr(cli, "extract_transcript", lambda *args: [("hello", 0.0, 0.5)])
    monkeypatch.setattr(cli, "build_timeline", lambda *args: [(0.0, "happy", "hello")])
    monkeypatch.setattr(cli, "print_timeline", lambda *args: None)

    captured = {"generated": None, "formatter": None}

    def fake_timeline_to_subtitles(timeline):
        return subtitles or [(0.0, 1.0, "hello", "happy")]

    class DummySubtitleGenerator:
        def __init__(self, formatter):
            captured["formatter"] = formatter

        def generate_file(self, subs, output_path):
            captured["generated"] = (subs, output_path)

    monkeypatch.setattr(cli, "timeline_to_subtitles", fake_timeline_to_subtitles)
    monkeypatch.setattr(cli, "SubtitleGenerator", DummySubtitleGenerator)
    return captured


def test_cli_without_file_exits_with_error(run_cli, caplog):
    caplog.set_level(logging.ERROR)
    exit_code, _ = run_cli([])
    assert exit_code == 1
    assert any("No audio file provided" in message for message in caplog.messages)


def test_cli_requires_subtitle_output_when_flags_present(run_cli, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    _stub_processing(monkeypatch)
    exit_code, _ = run_cli(["--file", "input.wav", "--subtitle-format", "srt"])
    assert exit_code == 1
    assert any("--subtitle-output is required" in message for message in caplog.messages)


def test_cli_inferrs_format_from_output_extension(run_cli, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    captured = _stub_processing(monkeypatch)
    output_path = tmp_path / "export.srt"

    exit_code, _ = run_cli(
        ["--file", "input.wav", "--subtitle-output", output_path.as_posix()],
        expect_exit=False,
    )

    assert exit_code == 0
    assert captured["generated"] == ([(0.0, 1.0, "hello", "happy")], output_path.as_posix())
    assert captured["formatter"].__class__.__name__.lower().startswith("srt")


def test_cli_format_override_logs_precedence(run_cli, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    captured = _stub_processing(monkeypatch)
    output_path = tmp_path / "export.vtt"

    exit_code, _ = run_cli(
        [
            "--file",
            "input.wav",
            "--subtitle-output",
            output_path.as_posix(),
            "--subtitle-format",
            "ass",
        ],
        expect_exit=False,
    )

    assert exit_code == 0
    assert captured["formatter"].__class__.__name__.lower().startswith("ass")
    assert any(
        "overriding inferred format" in message for message in caplog.messages
    )
