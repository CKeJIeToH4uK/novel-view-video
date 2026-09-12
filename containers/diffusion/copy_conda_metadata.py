"""Preserve installed Conda packages' legal texts and build recipes in an image.

Called only by Docker builders with their package cache mounted read-only.
The helper and package cache are not installed in the runtime image.
"""

import json
from pathlib import Path
import shutil
import sys


def copy_conda_metadata(prefix: Path, cache: Path, output: Path) -> None:
    for record_path in sorted((prefix / "conda-meta").glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        package = f"{record['name']}-{record['version']}-{record['build']}"
        source = cache / package / "info"
        destination = output / package
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "index.json", destination / "index.json")
        for entry in source.iterdir():
            if entry.name not in {"about.json", "licenses", "recipe"}:
                continue
            if entry.is_dir():
                shutil.copytree(entry, destination / entry.name, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, destination / entry.name)


if __name__ == "__main__":
    copy_conda_metadata(*(Path(argument) for argument in sys.argv[1:]))
