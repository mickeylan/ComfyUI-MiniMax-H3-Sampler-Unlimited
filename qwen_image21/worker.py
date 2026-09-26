"""Dedicated process entrypoint for Qwen Image 2.1 prompt rewriting.

This loads the isolated runtime_qwen36_38 module so the HR Endless/MiniMax-H3
qwen36_38.py runtime remains untouched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


_ROOT = Path(__file__).resolve().parent
_PACKAGE = "qwen_image21_worker"
_PLUGIN_ROOT = _ROOT.parent


def _load_package_module(name, *, root=_ROOT):
    package = sys.modules.get(_PACKAGE)
    if package is None:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_ROOT)]
        sys.modules[_PACKAGE] = package
    qualified = f"{_PACKAGE}.{name}"
    existing = sys.modules.get(qualified)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(qualified, root / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name}.py for Qwen Image 2.1 worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def main():
    _load_package_module("director_errors", root=_PLUGIN_ROOT)
    _load_package_module("story_format", root=_PLUGIN_ROOT)
    runtime = _load_package_module("runtime_qwen36_38")
    return runtime._worker_main()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("worker.py is an internal Qwen Image prompt-rewrite worker")
    raise SystemExit(main())
