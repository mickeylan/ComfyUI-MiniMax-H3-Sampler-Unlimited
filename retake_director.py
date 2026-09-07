"""Read-only browser view of the last HR Endless Sampler replay cache."""

import json
from pathlib import Path

from aiohttp import web
from comfy_api.latest import io

from .nodes import REPLAY_CACHE_FORMAT, _replay_cache_root

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


def replay_cache_snapshot():
    root = _replay_cache_root()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"available": False, "reason": "No last-run replay cache exists", "chunks": []}
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "reason": f"Could not read replay manifest: {error}", "chunks": []}
    chunks = []
    for entry in manifest.get("chunks", []):
        try:
            number = int(entry["chunk"])
            relative = Path(str(entry["metadata_path"]))
            path = (root / relative).resolve()
            if root.resolve() not in path.parents or relative.is_absolute():
                raise ValueError("metadata path leaves replay cache")
            metadata = json.loads(path.read_text(encoding="utf-8"))
            tensor_relative = Path(str(metadata.get("tensor_path", "")))
            tensor_path = (root / tensor_relative).resolve()
            tensor_exists = not tensor_relative.is_absolute() and root.resolve() in tensor_path.parents and tensor_path.is_file()
            images = []
            for value in metadata.get("observation_images", ()):
                image_relative = Path(str(value))
                image_path = (root / image_relative).resolve()
                if not image_relative.is_absolute() and root.resolve() in image_path.parents and image_path.is_file():
                    images.append(str(value))
            chunks.append({
                "chunk": number,
                "frame_start": metadata.get("frame_start"),
                "frame_end": metadata.get("frame_end"),
                "output_trim_frames": metadata.get("output_trim_frames", 0),
                "source_prompt": metadata.get("source_prompt", ""),
                "effective_h3_prompt": metadata.get("effective_h3_prompt", ""),
                "director_description": metadata.get("director_description", ""),
                "observation_images": images,
                "active_revision": metadata.get("active_revision", 0),
                "complete": bool(tensor_exists and metadata.get("effective_h3_prompt")),
            })
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            chunks.append({"chunk": entry.get("chunk"), "complete": False, "error": str(error), "observation_images": []})
    return {
        "available": True,
        "format": manifest.get("format"),
        "supported_format": REPLAY_CACHE_FORMAT,
        "status": manifest.get("status", "unknown"),
        "created": manifest.get("created"),
        "completed_chunks": manifest.get("completed_chunks", 0),
        "fps": manifest.get("fingerprint", {}).get("fps", 24.0),
        "compatible": manifest.get("format") == REPLAY_CACHE_FORMAT,
        "chunks": sorted(chunks, key=lambda item: int(item.get("chunk") or 0)),
    }


def _asset_path(relative_value):
    root = _replay_cache_root().resolve()
    relative = Path(str(relative_value))
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents or not path.is_file():
        raise web.HTTPNotFound()
    return path


_PROMPT_SERVER = None if PromptServer is None else getattr(PromptServer, "instance", None)
if _PROMPT_SERVER is not None:
    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_retake/cache")
    async def hr_endless_sampler_retake_cache(_request):
        return web.json_response(replay_cache_snapshot(), headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_retake/asset")
    async def hr_endless_sampler_retake_asset(request):
        return web.FileResponse(_asset_path(request.rel_url.query.get("path", "")),
                                headers={"Cache-Control": "no-store"})


class HREndlessSegmentRetakeDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessSegmentRetakeDirector",
            display_name="HR Endless Segment Retake Director",
            category="model/sampling/custom",
            description="Inspect the last run's chunk images and prompts. Retake execution will be added in a later phase.",
            outputs=[io.String.Output(display_name="cache summary")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(json.dumps(replay_cache_snapshot(), ensure_ascii=False))
