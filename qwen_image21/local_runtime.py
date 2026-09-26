"""Cached local Qwen3.5-VL runtime for Qwen Image 2.1 prompt rewriting.

This runtime is private to the qwen_image21 subpackage. It reuses model/mmproj
selection from HRDirectorConfig without changing the HR Endless runtimes.
"""

from __future__ import annotations

import atexit
import base64
import io
import threading
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image


class QwenImage21RuntimeError(RuntimeError):
    pass


def _image_data_uri(frame: torch.Tensor, max_pixels: int) -> str:
    image = frame.detach().to(device="cpu", dtype=torch.float32)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise QwenImage21RuntimeError(f"Expected one NHWC IMAGE batch, got {tuple(image.shape)}")
    pixels = image[0, ..., :3].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    pil = Image.fromarray(pixels, mode="RGB")
    width, height = pil.size
    if max_pixels > 0 and width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        pil = pil.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    pil.save(output, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class LocalQwen35Cache:
    def __init__(self):
        self._lock = threading.RLock()
        self._key: tuple[Any, ...] | None = None
        self._llm = None
        self._handler = None

    @staticmethod
    def _cache_key(model_path: Path, mmproj_path: Path, context_length: int) -> tuple[Any, ...]:
        model_stat = model_path.stat()
        mmproj_stat = mmproj_path.stat()
        return (
            str(model_path.resolve()), model_stat.st_size, model_stat.st_mtime_ns,
            str(mmproj_path.resolve()), mmproj_stat.st_size, mmproj_stat.st_mtime_ns,
            int(context_length),
        )

    def _load(self, model_path: Path, mmproj_path: Path, context_length: int):
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen35ChatHandler
        except ImportError as error:
            raise QwenImage21RuntimeError(
                "Qwen Image 2.1 local mode requires llama-cpp-python with Qwen35ChatHandler"
            ) from error
        handler = Qwen35ChatHandler(
            clip_model_path=str(mmproj_path),
            enable_thinking=False,
            preserve_thinking=False,
            image_min_tokens=256,
            image_max_tokens=1344,
            verbose=False,
            use_gpu=True,
        )
        try:
            llm = Llama(
                model_path=str(model_path), chat_handler=handler, n_gpu_layers=-1,
                n_ctx=int(context_length), n_batch=256, n_ubatch=256,
                flash_attn=True, type_k=8, type_v=8, swa_full=False, verbose=False,
            )
        except Exception:
            close_handler = getattr(handler, "close", None)
            if callable(close_handler):
                close_handler()
            raise
        self._handler = handler
        self._llm = llm

    def ensure_loaded(self, model_path: Path, mmproj_path: Path, context_length: int) -> bool:
        key = self._cache_key(model_path, mmproj_path, context_length)
        if self._llm is not None and self._key == key:
            return True
        self.unload()
        self._load(model_path, mmproj_path, context_length)
        self._key = key
        return False

    def generate(self, *, model_path: Path, mmproj_path: Path, context_length: int,
                 system_prompt: str, user_prompt: str, frames: Sequence[torch.Tensor],
                 max_pixels: int, temperature: float, top_p: float, top_k: int,
                 max_tokens: int, seed: int, thinking: bool,
                 json_response: bool = True) -> tuple[str, bool]:
        with self._lock:
            reused = self.ensure_loaded(model_path, mmproj_path, context_length)
            self._handler.enable_thinking = bool(thinking)
            self._handler.extra_template_arguments["enable_thinking"] = bool(thinking)
            content = [
                {"type": "image_url", "image_url": {"url": _image_data_uri(frame, max_pixels)}}
                for frame in frames
            ]
            content.append({"type": "text", "text": user_prompt})
            try:
                response = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    response_format={"type": "json_object"} if json_response and not thinking else None,
                    temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                    min_p=0.0, max_tokens=int(max_tokens), seed=int(seed),
                    present_penalty=0.0,
                )
            except Exception:
                self.unload()
                raise
            message = response["choices"][0]["message"]
            content_text = str(message.get("content") or "")
            reasoning_text = str(message.get("reasoning_content") or "")
            if reasoning_text and content_text:
                return f"<think>{reasoning_text}</think>\n{content_text}", reused
            return content_text or reasoning_text, reused

    def unload(self):
        with self._lock:
            llm, handler = self._llm, self._handler
            self._llm = None
            self._handler = None
            self._key = None
            close_llm = getattr(llm, "close", None)
            if callable(close_llm):
                close_llm()
            close_handler = getattr(handler, "close", None)
            if callable(close_handler):
                close_handler()


LOCAL_QWEN35_CACHE = LocalQwen35Cache()
atexit.register(LOCAL_QWEN35_CACHE.unload)
