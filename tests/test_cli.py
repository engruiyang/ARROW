from arrow_score.cli import main


def test_cli_info_returns_success(capsys) -> None:
    assert main(["info"]) == 0
    captured = capsys.readouterr()
    assert "TASK 1C" in captured.out


def test_cli_sequence_help_includes_debug_options(capsys) -> None:
    try:
        main(["sequence", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "--save-debug" in captured.out
    assert "--diff-threshold" in captured.out
