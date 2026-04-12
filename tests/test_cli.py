"""Tests for CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from blasbench.cli import app

runner = CliRunner()


class TestNormalize:
    def test_normalize_irish(self) -> None:
        result = runner.invoke(app, ["normalize", "Céad míle fáilte!"])
        assert result.exit_code == 0
        assert "céad míle fáilte" in result.stdout

    def test_normalize_preserves_fada(self) -> None:
        result = runner.invoke(app, ["normalize", "Á É Í Ó Ú"])
        assert result.exit_code == 0
        assert "á é í ó ú" in result.stdout


class TestListDatasets:
    def test_list_datasets(self) -> None:
        result = runner.invoke(app, ["list-datasets"])
        assert result.exit_code == 0
        assert "common-voice-ga" in result.stdout


class TestListModels:
    def test_list_models(self) -> None:
        result = runner.invoke(app, ["list-models"])
        assert result.exit_code == 0
        assert "whisper-large-v3" in result.stdout
