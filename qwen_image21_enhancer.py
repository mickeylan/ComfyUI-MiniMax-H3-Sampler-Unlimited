"""
Qwen Image 2.1 提示词增强节点

直接使用 HRDirectorConfig 进行提示词增强和翻译
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import torch

from comfy_api.latest import io
from typing_extensions import override

from .director_config import HRDirectorConfig, normalize_qwen38_config
from .director_backend import resolve_director_selection
from .qwen35 import Qwen35ContinuityDirector


# ============== 自定义类型定义 ==============
EnhancedPromptResult = io.Custom("ENHANCED_PROMPT_RESULT")
_PROMPT_DIR = Path(__file__).with_name("qwen_image21_prompts")
T2I_SYSTEM_PROMPT = (_PROMPT_DIR / "system_prompt_t2i.txt").read_text(encoding="utf-8").strip()
I2I_SYSTEM_PROMPT = (_PROMPT_DIR / "system_prompt_edit.txt").read_text(encoding="utf-8").strip()


# ============== 主增强节点 ==============
class QwenImage21PromptEnhancer(io.ComfyNode):
    """Qwen Image 2.1 提示词增强节点"""
    
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="QwenImage21PromptEnhancer",
            display_name="Qwen Image 2.1 Prompt Enhancer",
            category="model/sampling/custom",
            description="Qwen Image 2.1 提示词增强，支持 T2I/I2I 模式",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input("mode", options=["t2i", "i2i"], default="t2i"),
                io.Combo.Input("language", options=["en", "zh"], default="en"),
                io.Autogrow.Input("images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_", min=0, max=16)),
                HRDirectorConfig.Input("director_config"),
            ],
            outputs=[
                io.String.Output(display_name="enhanced_prompt"),
                io.String.Output(display_name="thinking"),
                io.String.Output(display_name="wh_ratio"),
                io.String.Output(display_name="ratio_follow"),
                EnhancedPromptResult.Output(display_name="full_result"),
            ],
        )
    
    @classmethod
    @override
    def execute(
        cls,
        prompt: str,
        mode: str,
        language: str,
        images: Optional[dict] = None,
        director_config: Optional[dict] = None,
        **dynamic_inputs,
    ) -> io.NodeOutput:
        if not prompt.strip():
            return io.NodeOutput("", "", "", "", "{}")
        
        # 验证 director_config
        if director_config is None:
            return io.NodeOutput("[No director_config]", "", "", "", "{}")
        
        try:
            config = normalize_qwen38_config(director_config)
        except Exception as e:
            return io.NodeOutput(f"[Config error: {e}]", "", "", "", "{}")
        
        selection = resolve_director_selection(
            config["backend"],
            config["model"],
            config["mmproj"]
        )
        
        if selection.model_path is None or selection.mmproj_path is None:
            return io.NodeOutput("[No Qwen model found]", "", "", "", "{}")
        
        # 构建系统提示词
        if mode == "t2i":
            system_prompt = T2I_SYSTEM_PROMPT
        else:
            system_prompt = I2I_SYSTEM_PROMPT
        
        output_language = "Chinese" if language == "zh" else "English"
        user_prompt = f"""User Request: {prompt}

Write rewritten_prompt in {output_language}. Preserve explicitly requested visible text verbatim in its original script.
Return only the requested JSON object."""

        frames = cls._autogrow_images(images, dynamic_inputs)
        if mode == "i2i" and not frames:
            raise ValueError("Qwen Image 2.1 i2i enhancement requires at least one image")
        
        # 创建 Director 并执行
        start_time = time.time()
        try:
            director = Qwen35ContinuityDirector(
                model_path=selection.model_path,
                mmproj_path=selection.mmproj_path,
                backend=config["backend"],
                mtp_enabled=config["mtp"],
                mtp_draft_tokens=config["mtp_draft_tokens"],
                reasoning_effort=config["reasoning_effort"],
                debug=config["debug"],
                cpu_moe=config["cpu_moe"],
                n_cpu_moe=config["n_cpu_moe"],
            )

            raw_result = director.plan_jzl_storyboard(
                frames, system_prompt=system_prompt, user_prompt=user_prompt,
            )
            enhanced_prompt, wh_ratio, ratio_follow, thinking = cls._parse_result(
                raw_result, allow_ratio_follow=(mode == "i2i")
            )
            
        except Exception as e:
            logging.error(f"Prompt enhancement failed: {e}")
            return io.NodeOutput(f"[Error: {e}]", "", "", "", "{}")
        
        # 构建完整结果
        full_result = {
            "enhanced_prompt": enhanced_prompt,
            "thinking": thinking,
            "wh_ratio": wh_ratio,
            "ratio_follow": ratio_follow,
            "mode": mode,
            "image_count": len(frames),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        
        return io.NodeOutput(
            enhanced_prompt,
            thinking,
            wh_ratio,
            ratio_follow,
            json.dumps(full_result, ensure_ascii=False),
        )
    
    @staticmethod
    def _autogrow_images(images, dynamic_inputs) -> tuple[torch.Tensor, ...]:
        values = dict(images or {}) if isinstance(images, dict) else {}
        if images is not None and not isinstance(images, dict):
            raise ValueError("images must come from the Autogrow input")
        unknown = []
        for name, value in dynamic_inputs.items():
            if name.startswith("image_"):
                if value is not None:
                    values[name] = value
            else:
                unknown.append(name)
        if unknown:
            raise ValueError("Unknown Qwen Image 2.1 enhancer inputs: " + ", ".join(sorted(unknown)))

        def slot_index(item):
            name = str(item[0])
            suffix = name.removeprefix("image_")
            return int(suffix) if suffix.isdigit() else 10**9

        frames = []
        for name, value in sorted(values.items(), key=slot_index):
            if not str(name).startswith("image_") or value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[-1] not in (1, 3, 4):
                raise ValueError(f"{name} must be an NHWC IMAGE batch")
            frames.extend(value[index:index + 1] for index in range(value.shape[0]))
        if len(frames) > 16:
            raise ValueError(f"Qwen Image 2.1 supports at most 16 images, received {len(frames)}")
        return tuple(frames)

    @staticmethod
    def _parse_result(text: str, *, allow_ratio_follow: bool) -> tuple[str, str, str, str]:
        raw = str(text or "").strip()
        thinking = ""
        if "</think>" in raw:
            thinking, _, raw = raw.partition("</think>")
            thinking = thinking.removeprefix("<think>").strip()
            raw = raw.strip()
        decoder = json.JSONDecoder()
        for start, char in enumerate(raw):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(raw[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
            if isinstance(rewritten, str) and rewritten.strip():
                return (
                    rewritten.strip(),
                    str(obj.get("wh_ratio") or "").strip(),
                    str(obj.get("ratio_follow") or "").strip() if allow_ratio_follow else "",
                    thinking,
                )
        return raw, "", "", thinking


# ============== 翻译节点 ==============
class QwenImage21Translator(io.ComfyNode):
    """中英互译节点"""
    
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="QwenImage21Translator",
            display_name="Qwen Image 2.1 Translator",
            category="model/sampling/custom",
            description="中英互译节点",
            inputs=[
                io.String.Input("text", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input("direction", options=["auto", "zh2en", "en2zh"], default="auto"),
                HRDirectorConfig.Input("director_config"),
            ],
            outputs=[
                io.String.Output(display_name="translated_text"),
                io.String.Output(display_name="detected_lang"),
            ],
        )
    
    @classmethod
    @override
    def execute(
        cls,
        text: str,
        direction: str,
        director_config: Optional[dict] = None,
    ) -> io.NodeOutput:
        if not text.strip():
            return io.NodeOutput("", "")
        
        if direction == "auto":
            detected = cls._detect_language(text)
            direction = "zh2en" if detected == "zh" else "en2zh"
        else:
            detected = "zh" if direction == "zh2en" else "en"
        
        if direction == "zh2en":
            translate_prompt = f"""Translate the following Chinese text to English.
Only output the translated text, nothing else.

Chinese: {text}"""
        else:
            translate_prompt = f"""Translate the following English text to Chinese.
Only output the translated text, nothing else.

English: {text}"""
        
        if director_config is None:
            return io.NodeOutput(text, detected)
        
        try:
            config = normalize_qwen38_config(director_config)
        except Exception as e:
            return io.NodeOutput(f"[Config error: {e}]", detected)
        
        selection = resolve_director_selection(
            config["backend"],
            config["model"],
            config["mmproj"]
        )
        
        if selection.model_path is None or selection.mmproj_path is None:
            return io.NodeOutput("[No Qwen model found]", detected)
        
        try:
            director = Qwen35ContinuityDirector(
                model_path=selection.model_path,
                mmproj_path=selection.mmproj_path,
                backend=config["backend"],
                mtp_enabled=config["mtp"],
                mtp_draft_tokens=config["mtp_draft_tokens"],
                reasoning_effort=config["reasoning_effort"],
                debug=config["debug"],
                cpu_moe=config["cpu_moe"],
                n_cpu_moe=config["n_cpu_moe"],
            )

            translated = director.plan_jzl_storyboard(
                (),
                system_prompt=(
                    "You are a professional translator. Preserve meaning, tone, proper nouns, quoted text, "
                    "and formatting. Output only the translation with no explanation or JSON."
                ),
                user_prompt=translate_prompt,
            )
            
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return io.NodeOutput(f"[Error: {e}]", detected)
        
        return io.NodeOutput(translated.strip(), detected)
    
    @staticmethod
    def _detect_language(text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'[\w\u4e00-\u9fff]', text))
        if total_chars == 0:
            return "en"
        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"

