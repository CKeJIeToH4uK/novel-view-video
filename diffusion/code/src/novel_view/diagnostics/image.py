"""Явная проверка установленных prefix образа без GPU, моделей и данных."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
from importlib.metadata import version
import importlib.util
import json
import os
import subprocess
import sys
from types import ModuleType


def _core_imports() -> list[ModuleType]:
    modules = [
        importlib.import_module(name)
        for name in (
            "novel_view.cli.main",
            "novel_view.workflows.runner",
            "novel_view.workflows.euvs_source_views",
            "novel_view.workflows.euvs_generation",
            "novel_view.workflows.euvs_evaluation",
            "novel_view.workflows.euvs_comparison",
            "novel_view.workflows.gaussian_generation",
            "novel_view.workflows.ddw_preparation",
            "novel_view.workflows.gen3c_training",
            "novel_view.workflows.gen3c_checkpoint_selection",
            "novel_view.workflows.ddw_evaluation",
        )
    ]
    if "torch" in sys.modules:
        raise RuntimeError("Core CLI/workflows imported torch")
    return modules


def _gen3c_imports(variant: str) -> list[ModuleType]:
    import apex.optimizers
    from cosmos_predict1.diffusion.inference import cache_3d
    from cosmos_predict1.diffusion.inference import forward_warp_utils_pytorch
    import sam2.build_sam
    import transformer_engine.pytorch

    sys.path.insert(0, "/opt/upstream/dinov2")
    try:
        from dinov2.hub import backbones
    finally:
        sys.path.pop(0)
    modules = [
        cache_3d,
        forward_warp_utils_pytorch,
        backbones,
        apex.optimizers,
        transformer_engine.pytorch,
        sam2.build_sam,
    ]
    if variant == "moge":
        from moge.model.v1 import MoGeModel

        modules.append(sys.modules[MoGeModel.__module__])
    elif importlib.util.find_spec("moge") is not None:
        raise RuntimeError("MoGe is present in the core image")
    return modules


def _vggt_imports() -> list[ModuleType]:
    from novel_view.models.vggt import _worker
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    return [
        _worker,
        sys.modules[VGGTOmega.__module__],
        sys.modules[encoding_to_camera.__module__],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", choices=("core", "gen3c", "vggt"), required=True)
    parser.add_argument("--variant", choices=("core", "moge"), required=True)
    arguments = parser.parse_args()
    prefix_root = f"/opt/envs/{arguments.prefix}"
    if sys.prefix != prefix_root:
        raise RuntimeError(
            f"Expected interpreter prefix {prefix_root}, got {sys.prefix}"
        )
    if os.geteuid() == 0:
        raise RuntimeError("The image diagnostic requires its non-root default user")
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        stdout=sys.stderr,
    )
    with redirect_stdout(sys.stderr):
        import novel_view

        if arguments.prefix == "core":
            modules = _core_imports()
        elif arguments.prefix == "gen3c":
            modules = _gen3c_imports(arguments.variant)
        else:
            modules = _vggt_imports()
    origins = {module.__name__: module.__file__ for module in [novel_view, *modules]}
    for name, origin in origins.items():
        root = prefix_root
        if name.startswith("cosmos_predict1."):
            root = "/opt/upstream/gen3c"
        elif name.startswith("dinov2."):
            root = "/opt/upstream/dinov2"
        if not origin or not origin.startswith(root + "/"):
            raise RuntimeError(f"Import {name} came from outside {root}: {origin}")
    torch = sys.modules.get("torch")
    cuda_initialized = torch.cuda.is_initialized() if torch is not None else False
    if cuda_initialized:
        raise RuntimeError("Image imports initialized CUDA")
    print(
        json.dumps(
            {
                "prefix": arguments.prefix,
                "variant": arguments.variant,
                "project_version": version("novel-view-pipeline"),
                "import_origins": origins,
                "non_root": True,
                "cuda_initialized": cuda_initialized,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
