"""Installed diagnostic entrypoints expose the unchanged concrete options."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("diagnostic", ("ingress", "raster"))
def test_waymo_doctor_installed_help(diagnostic, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "novel_view.cli.main", "doctor-waymo", diagnostic, "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--waymo-root" in result.stdout and "--split-config" in result.stdout
    assert ("--max-rss-mib" in result.stdout) == (diagnostic == "ingress")
