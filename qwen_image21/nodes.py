"""
Qwen Image 2.1 提示词增强节点

直接使用 HRDirectorConfig 进行提示词增强和翻译
"""

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import torch

from comfy_api.latest import io
from typing_extensions import override

from ..director_config import HRDirectorConfig, normalize_qwen38_config
from ..director_backend import resolve_director_selection
from .runtime_qwen35 import Qwen35ContinuityDirector


# ============== 自定义类型定义 ==============
EnhancedPromptResult = io.Custom("ENHANCED_PROMPT_RESULT")
_PROMPT_DIR = Path(__file__).with_name("prompts")
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
                io.Combo.Input("mode", options=["auto", "t2i", "i2i"], default="auto",
                               tooltip="Connected images always select I2I; without images auto/t2i select T2I"),
                io.Combo.Input("language", options=["en", "zh"], default="zh",
                               tooltip="Controls only the language of rewritten_prompt; enhancement always uses the full official rules"),
                io.Float.Input("temperature", default=1.0, min=0.01, max=2.0, step=0.01,
                               tooltip="Official Qwen Image 2.1 PE default: 1.0"),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=20, min=0, max=1000, step=1),
                io.Int.Input("max_new_tokens", default=16384, min=512, max=24000, step=256,
                             tooltip="Includes thinking plus final rewrite"),
                io.Int.Input("thinking_budget", default=4096, min=0, max=8192, step=256,
                             tooltip="Official PE requires thinking; 0 disables it"),
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
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        thinking_budget: int,
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
        
        frames = cls._autogrow_images(images, dynamic_inputs)
        effective_mode = "i2i" if frames else "t2i"
        if mode == "i2i" and not frames:
            raise ValueError("I2I enhancement requires at least one image")

        official_system_prompt = T2I_SYSTEM_PROMPT if effective_mode == "t2i" else I2I_SYSTEM_PROMPT
        output_language = "Chinese" if language == "zh" else "English"
        system_prompt = (
            f"{official_system_prompt}\n\nOutput-language requirement: write rewritten_prompt descriptive prose in {output_language}. "
            "This changes only the final prompt language; follow every official enhancement rule above. "
            "Visible text requested inside the generated image remains verbatim in its requested script."
        )
        user_prompt = prompt
        
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

            generation = {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "max_tokens": int(max_new_tokens),
                "reasoning_budget": int(thinking_budget),
            }
            raw_result = director.plan_jzl_storyboard(
                frames, system_prompt=system_prompt, user_prompt=user_prompt,
                generation=generation, json_response=False, enable_thinking=(thinking_budget > 0),
            )
            enhanced_prompt, wh_ratio, ratio_follow, thinking, parse_ok = cls._parse_result(
                raw_result, allow_ratio_follow=(effective_mode == "i2i")
            )
            validation_issues = cls._validation_issues(
                original_prompt=prompt,
                rewritten_prompt=enhanced_prompt,
                language=language,
                image_count=len(frames),
                parse_ok=parse_ok,
            )
            retried = False
            if validation_issues:
                retried = True
                correction_prompt = f"""Your previous response failed validation:
- {chr(10).join(validation_issues)}

Previous response:
<previous_response>
{raw_result}
</previous_response>

Follow the official Qwen Image 2.1 system rules exactly. Re-read all {len(frames)} supplied images in their original order and correct the validation failures. When this is multi-image compositing or a new scene, apply the official "construct actively" branch rather than merely paraphrasing the placement request. Return exactly one valid JSON object and nothing else.

<user_request>
{prompt}
</user_request>"""
                raw_result = director.plan_jzl_storyboard(
                    frames, system_prompt=system_prompt, user_prompt=correction_prompt,
                    generation=generation, json_response=False, enable_thinking=(thinking_budget > 0),
                )
                enhanced_prompt, wh_ratio, ratio_follow, thinking, parse_ok = cls._parse_result(
                    raw_result, allow_ratio_follow=(effective_mode == "i2i")
                )
                validation_issues = cls._validation_issues(
                    original_prompt=prompt,
                    rewritten_prompt=enhanced_prompt,
                    language=language,
                    image_count=len(frames),
                    parse_ok=parse_ok,
                )
            if validation_issues:
                raise ValueError("Qwen Image 2.1 rewrite failed validation after correction: " + "; ".join(validation_issues))
            
        except Exception as e:
            logging.error(f"Prompt enhancement failed: {e}")
            return io.NodeOutput(f"[Error: {e}]", "", "", "", "{}")
        
        # 构建完整结果
        full_result = {
            "enhanced_prompt": enhanced_prompt,
            "thinking": thinking,
            "wh_ratio": wh_ratio,
            "ratio_follow": ratio_follow,
            "requested_mode": mode,
            "effective_mode": effective_mode,
            "language": language,
            "image_count": len(frames),
            "parse_ok": parse_ok,
            "retried": retried,
            "validation_issues": validation_issues,
            "generation": generation,
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
    def _parse_result(text: str, *, allow_ratio_follow: bool) -> tuple[str, str, str, str, bool]:
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
                    True,
                )
        return raw, "", "", thinking, False

    @staticmethod
    def _validation_issues(*, original_prompt: str, rewritten_prompt: str, language: str,
                           image_count: int, parse_ok: bool) -> list[str]:
        issues = []
        rewritten = str(rewritten_prompt or "").strip()
        original = str(original_prompt or "").strip()
        if not parse_ok:
            issues.append("response is not the required JSON object")
        if not rewritten:
            issues.append("rewritten_prompt is empty")
            return issues

        normalized_original = re.sub(r"\s+", "", original).casefold()
        normalized_rewritten = re.sub(r"\s+", "", rewritten).casefold()
        similarity = SequenceMatcher(None, normalized_original, normalized_rewritten).ratio()
        if normalized_rewritten == normalized_original or similarity >= 0.92:
            issues.append("rewritten_prompt merely repeats the user request")
        elif len(normalized_original) >= 12 and len(normalized_rewritten) < int(len(normalized_original) * 1.2):
            issues.append("rewritten_prompt is not materially expanded or clarified")

        if image_count >= 2:
            missing = [f"<image{index}>" for index in range(1, image_count + 1)
                       if f"<image{index}>" not in rewritten]
            if missing:
                issues.append("missing required image references: " + ", ".join(missing))
            active_composition = image_count >= 3 or bool(re.search(
                r"(?:合成|组合|合影|合照|置于|放到|移入|躺在|一起|场景|composit|place|put|together|scene)",
                original, re.IGNORECASE,
            ))
            if active_composition:
                if language == "zh":
                    detail_count = len(re.findall(r"[\u4e00-\u9fff]", rewritten))
                    if detail_count < 120:
                        issues.append(f"multi-image composition did not use the official construct-actively branch ({detail_count} < 120 Chinese characters)")
                else:
                    detail_count = len(re.findall(r"\b[A-Za-z]+\b", rewritten))
                    if detail_count < 80:
                        issues.append(f"multi-image composition did not use the official construct-actively branch ({detail_count} < 80 English words)")

        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", rewritten))
        letter_count = len(re.findall(r"[A-Za-z]", rewritten))
        if language == "zh" and chinese_count == 0:
            issues.append("rewritten_prompt is not in Chinese")
        elif language == "en" and letter_count == 0:
            issues.append("rewritten_prompt is not in English")
        return issues


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

