"""Односторонние imports, установленный wheel и граница host/Python."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src/novel_view"
ROOT = SOURCE.parents[3]
LAUNCHERS = [ROOT / "distil3d", *sorted((ROOT / "containers/launcher").glob("*.sh"))]
MARKERS = ("tests/", "tests.", "research/", "research.")


@pytest.fixture(scope="module")
def python_facts():
    facts = {}
    for source in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(source.read_text())
        docstrings = {
            node.body[0].value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(node, clean=False) is not None
        }
        imports, strings = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((alias.name, 0) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append((node.module or "", node.level))
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node not in docstrings
            ):
                strings.append(node.value)
        facts[source.relative_to(SOURCE)] = (imports, strings)
    return facts


def test_production_imports_and_dispatch_do_not_use_tests_or_research(python_facts):
    for path, (imports, strings) in python_facts.items():
        assert not any(name.split(".")[0] in {"tests", "research"} for name, _ in imports), path
        assert not any(marker in value for value in strings for marker in MARKERS), path
    scripts = [*LAUNCHERS, *(SOURCE.parents[1] / "scripts").rglob("*.sh")]
    for script in scripts:
        # Явные host-проверки: test и единственный TE node в accept.
        allowed = ()
        if script in LAUNCHERS:
            allowed = (
                ("tests/", "tests.")
                if script.name == "tests.sh"
                else ("containers/launcher/tests.sh",)
            )
        for number, line in enumerate(script.read_text().splitlines(), 1):
            if script == ROOT / "containers/launcher/accept.sh" and line.strip() == (
                "/tests/integration/test_model_api.py::"
                "test_te_recompute_preserves_result_and_gradients"
            ):
                continue
            if line.lstrip().startswith("#") or any(value in line for value in allowed):
                continue
            assert not any(marker in line for marker in MARKERS), (script, number)


def test_geometry_vggt_and_records_keep_one_way_dependencies(python_facts):
    for path, (imports, _) in python_facts.items():
        if path.parent == Path("geometry"):
            for name, level in imports:
                assert level <= 1, (path, name)
                assert (
                    level == 1
                    or name.startswith("novel_view.geometry")
                    or name.split(".")[0] in sys.stdlib_module_names | {"numpy"}
                ), (path, name)
        if path.parent == Path("models/vggt"):
            assert not any(
                name.startswith(
                    (
                        "novel_view.config.legacy",
                        "novel_view.inputs.euvs",
                        "novel_view.generation.euvs",
                        "novel_view.source_geometry",
                    )
                )
                for name, _ in imports
            ), path
        if path.parent == Path("runs"):
            assert not any(
                name.startswith("novel_view.") and not name.startswith("novel_view.runs")
                for name, _ in imports
            ), path


def test_host_owns_docker_and_attempt_writer_stays_in_python(python_facts):
    for path, (imports, strings) in python_facts.items():
        names = {name for name, _ in imports}
        docker = any(value.partition(" ")[0] == "docker" for value in strings)
        assert not ({"subprocess", "os"} & names and docker), path
        if path.parts[0] in {
            "workflows",
            "models",
            "generation",
            "training",
            "evaluation",
            "preparation",
        }:
            assert not any(
                name.startswith(("novel_view.runtime.candidate", "novel_view.cli.candidate"))
                for name in names
            ), path
            assert not set(strings) & {"candidate.json", "profile.yaml", "DISTIL3D_DATA_ROOT"}, path
    host_text = "\n".join(path.read_text() for path in LAUNCHERS)
    assert "attempt.json" not in host_text and "_worker" not in host_text
    assert not any(
        term in host_text.lower()
        for term in ("staging", "no-overwrite", "atomic", "checkpoint_compat")
    )
    assert not list(ROOT.glob("*compose.y*ml")) and not list(SOURCE.rglob("paths.py"))
    ignored = set((ROOT / ".dockerignore").read_text().splitlines())
    required = set(
        ".git logs plans data models weights runs prepared cache profiles/local "
        "jobs/local selections/local **/.env **/candidate.json **/images.tar".split()
    )
    assert required <= ignored


def test_installed_wheel_and_executable_plan_import_are_cold():
    script = """
import sys
from pathlib import Path
from importlib.metadata import version
import novel_view
import novel_view.cli.main
import novel_view.models
import novel_view.geometry
package = Path(novel_view.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve())
assert 'site-packages' in package.parts
assert version('novel-view-pipeline') == novel_view.__version__
blocked = {'novel_view.cli.legacy', 'cv2', 'pyarrow', 'torch', 'transformers', 'megatron', 'apex'} & sys.modules.keys()
assert not blocked, sorted(blocked)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
