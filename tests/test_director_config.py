import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hr_endless_director_config_test"

package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE] = package

comfy_api = sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
latest = types.ModuleType("comfy_api.latest")
latest.io = types.SimpleNamespace(
    Custom=lambda _name: types.SimpleNamespace(Output=lambda **_kwargs: None),
    ComfyNode=object,
    NodeOutput=lambda value: (value,),
)
sys.modules["comfy_api.latest"] = latest
comfy_api.latest = latest

backend = types.ModuleType(f"{PACKAGE}.director_backend")
backend.DIRECTOR_BACKENDS = ("gemma4", "qwen3.5", "qwen3.6", "qwen3.8")
backend.QWEN_DIRECTOR_BACKENDS = ("qwen3.5", "qwen3.6", "qwen3.8")
backend.director_model_options = lambda projector=False: ["auto"]
backend.resolve_director_selection = lambda *args: None
sys.modules[backend.__name__] = backend

spec = importlib.util.spec_from_file_location(f"{PACKAGE}.director_config", PLUGIN_ROOT / "director_config.py")
director_config = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = director_config
spec.loader.exec_module(director_config)


class DirectorConfigTests(unittest.TestCase):
    def test_rejects_gemma4_because_this_node_is_qwen_only(self):
        with self.assertRaisesRegex(ValueError, "does not support gemma4"):
            director_config.normalize_qwen38_config({
                "version": director_config.CONFIG_VERSION,
                "backend": "gemma4",
            })

    def test_qwen_backend_requires_a_complete_local_pair(self):
        selection = types.SimpleNamespace(model_path=None, mmproj_path=None)
        with patch.object(director_config, "resolve_director_selection", return_value=selection):
            with self.assertRaisesRegex(ValueError, "requires a local qwen3.8"):
                director_config.HRQwen38DirectorConfig.execute(backend="qwen3.8")


if __name__ == "__main__":
    unittest.main()
