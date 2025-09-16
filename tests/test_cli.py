"""Tests for CLI functionality."""

from pathlib import Path

from click.testing import CliRunner
from timeline_extraction.cli import cli


class TestCLI:
    """Test CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Timeline Extraction" in result.output

    def test_data_group_help(self):
        """Test data command group help."""
        result = self.runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "Data management" in result.output

    def test_model_group_help(self):
        """Test model command group help."""
        result = self.runner.invoke(cli, ["model", "--help"])
        assert result.exit_code == 0
        assert "Model evaluation" in result.output

    def test_results_group_help(self):
        """Test results command group help."""
        result = self.runner.invoke(cli, ["results", "--help"])
        assert result.exit_code == 0
        assert "Results analysis" in result.output

    def test_cycles_group_help(self):
        """Test cycles command group help."""
        result = self.runner.invoke(cli, ["cycles", "--help"])
        assert result.exit_code == 0
        assert "Temporal cycle" in result.output

    def test_config_group_help(self):
        """Test config command group help."""
        result = self.runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "Configuration management" in result.output

    def test_config_generate(self):
        """Test config generate command."""
        result = self.runner.invoke(
            cli, ["config", "generate", "--output", "test_config.yaml"]
        )
        assert result.exit_code == 0
        assert "Configuration file generated" in result.output

        # Check if file was created
        config_file = Path("test_config.yaml")
        assert config_file.exists()

        # Clean up
        config_file.unlink()

    def test_verbose_flag(self):
        """Test verbose flag."""
        result = self.runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_log_level(self):
        """Test log level option."""
        result = self.runner.invoke(cli, ["--log-level", "DEBUG", "--help"])
        assert result.exit_code == 0
