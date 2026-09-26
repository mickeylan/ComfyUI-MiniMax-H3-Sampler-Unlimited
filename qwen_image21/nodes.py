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
from .local_runtime import LOCAL_QWEN35_CACHE


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
                io.Int.Input("max_new_tokens", default=4096, min=512, max=24000, step=256,
                             tooltip="Upper generation limit; 4096 is normally enough for one rewrite"),
                io.Int.Input("context_length", default=32768, min=8192, max=131072, step=1024,
                             tooltip="Exact llama.cpp context length. Increase for more images or larger output budgets; uses more memory"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("image_max_pixels", default=1048576, min=262144, max=4194304, step=262144,
                             tooltip="Each reference image is resized below this pixel count before vision encoding"),
                io.Boolean.Input("enable_thinking", default=False,
                                 tooltip="Disabled by default, matching TE-MAN local mode"),
                io.Boolean.Input("auto_unload", default=True,
                                 tooltip="Unload the Qwen model after generation to release VRAM; disable to reuse it on later executions"),
                io.Boolean.Input("retry_on_validation", default=False,
                                 tooltip="Retry once when JSON/image/language validation fails; reloads the model and can nearly double runtime"),
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
        context_length: int,
        seed: int,
        image_max_pixels: int,
        enable_thinking: bool,
        auto_unload: bool,
        retry_on_validation: bool,
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
        if config["backend"] != "qwen3.5":
            raise ValueError("The cached Qwen Image 2.1 runtime currently requires backend qwen3.5")
        
        frames = cls._autogrow_images(images, dynamic_inputs)
        normalized_prompt = cls._normalize_image_references(prompt, len(frames))
        effective_mode = "i2i" if frames else "t2i"
        if mode == "i2i" and not frames:
            raise ValueError("I2I enhancement requires at least one image")

        official_system_prompt = T2I_SYSTEM_PROMPT if effective_mode == "t2i" else I2I_SYSTEM_PROMPT
        output_language = "Simplified Chinese" if language == "zh" else "English"
        system_prompt = cls._system_prompt_for_language(
            official_system_prompt, mode=effective_mode, language=language,
            output_language=output_language,
        )
        if frames:
            image_mapping = "\n".join(
                f"Picture {index} in this message is <image{index}> in the official rewrite rules."
                for index in range(1, len(frames) + 1)
            )
            user_prompt = (
                f"{image_mapping}\nAll {len(frames)} images are present above this text and must be inspected.\n"
                f"User Raw Input Prompt: {normalized_prompt}"
            )
        else:
            user_prompt = f"User Raw Input Prompt: {normalized_prompt}"
        logging.info(
            "Qwen Image 2.1 rewrite sending mode=%s image_count=%d language=%s model=%s mmproj=%s",
            effective_mode, len(frames), language, selection.model_path.name, selection.mmproj_path.name,
        )
        context_length = int(context_length)
        estimated_required_context = cls._estimated_required_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_count=len(frames),
            max_new_tokens=int(max_new_tokens),
        )
        if context_length < estimated_required_context:
            raise ValueError(
                f"context_length={context_length} is likely too small for {len(frames)} image(s), "
                f"the official rules, and max_new_tokens={max_new_tokens}; set it to at least "
                f"{estimated_required_context}"
            )

        start_time = time.time()
        try:
            generation = {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "max_tokens": int(max_new_tokens),
                "seed": int(seed),
                "enable_thinking": bool(enable_thinking),
            }
            raw_result, model_reused = LOCAL_QWEN35_CACHE.generate(
                model_path=selection.model_path, mmproj_path=selection.mmproj_path,
                context_length=context_length, system_prompt=system_prompt,
                user_prompt=user_prompt, frames=frames, max_pixels=int(image_max_pixels),
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_tokens=max_new_tokens, seed=seed, thinking=enable_thinking,
            )
            enhanced_prompt, wh_ratio, ratio_follow, thinking, parse_ok = cls._parse_result(
                raw_result, allow_ratio_follow=(effective_mode == "i2i")
            )
            enhanced_prompt = cls._normalize_image_references(enhanced_prompt, len(frames))
            ratio_follow = cls._normalize_image_references(ratio_follow, len(frames))
            validation_issues = cls._validation_issues(
                original_prompt=normalized_prompt,
                rewritten_prompt=enhanced_prompt,
                language=language,
                image_count=len(frames),
                parse_ok=parse_ok,
            )
            retried = False
            language_only_failure = bool(validation_issues) and all("not in " in issue for issue in validation_issues)
            if validation_issues and (retry_on_validation or language_only_failure):
                retried = True
                correction_prompt = f"""Your previous response failed validation:
- {chr(10).join(validation_issues)}

Previous response:
<previous_response>
{raw_result}
</previous_response>

Follow the official Qwen Image 2.1 rules exactly and correct the validation failures. The entire descriptive prose in rewritten_prompt MUST be in {output_language}; do not translate exact user-supplied visible text, proper nouns or brand names. Re-read all {len(frames)} supplied images in their original order. In this message, Picture N is the same source as <imageN> in the official rules. Return exactly one valid JSON object on one line and stop immediately after the closing brace.

<user_request>
{normalized_prompt}
</user_request>"""
                raw_result, model_reused = LOCAL_QWEN35_CACHE.generate(
                    model_path=selection.model_path, mmproj_path=selection.mmproj_path,
                    context_length=context_length, system_prompt=system_prompt,
                    user_prompt=correction_prompt, frames=frames, max_pixels=int(image_max_pixels),
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    max_tokens=max_new_tokens, seed=seed, thinking=enable_thinking,
                )
                enhanced_prompt, wh_ratio, ratio_follow, thinking, parse_ok = cls._parse_result(
                    raw_result, allow_ratio_follow=(effective_mode == "i2i")
                )
                enhanced_prompt = cls._normalize_image_references(enhanced_prompt, len(frames))
                ratio_follow = cls._normalize_image_references(ratio_follow, len(frames))
                validation_issues = cls._validation_issues(
                    original_prompt=normalized_prompt,
                    rewritten_prompt=enhanced_prompt,
                    language=language,
                    image_count=len(frames),
                    parse_ok=parse_ok,
                )
            if validation_issues:
                raise ValueError(
                    "Qwen Image 2.1 rewrite failed validation"
                    + (" after correction" if retried else "")
                    + ": " + "; ".join(validation_issues)
                )
            coverage_warnings = cls._image_reference_warnings(enhanced_prompt, len(frames))
            
        except Exception as e:
            logging.error(f"Prompt enhancement failed: {e}")
            LOCAL_QWEN35_CACHE.unload()
            return io.NodeOutput(f"[Error: {e}]", "", "", "", "{}")
        finally:
            if auto_unload:
                LOCAL_QWEN35_CACHE.unload()
        
        # 构建完整结果
        full_result = {
            "enhanced_prompt": enhanced_prompt,
            "thinking": thinking,
            "wh_ratio": wh_ratio,
            "ratio_follow": ratio_follow,
            "requested_mode": mode,
            "effective_mode": effective_mode,
            "language": language,
            "raw_prompt": prompt,
            "normalized_prompt": normalized_prompt,
            "image_count": len(frames),
            "image_mapping": [f"Picture {index}=<image{index}>" for index in range(1, len(frames) + 1)],
            "parse_ok": parse_ok,
            "model_reused": bool(model_reused),
            "auto_unload": bool(auto_unload),
            "context_length": context_length,
            "estimated_required_context": estimated_required_context,
            "image_max_pixels": int(image_max_pixels),
            "retry_on_validation": bool(retry_on_validation),
            "retried": retried,
            "validation_issues": validation_issues,
            "coverage_warnings": coverage_warnings,
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
    def _system_prompt_for_language(base_prompt: str, *, mode: str, language: str,
                                    output_language: str) -> str:
        prompt = base_prompt
        if mode == "i2i":
            # The official edit prompt derives prose language from the request.
            # The node's explicit language selector replaces only that decision;
            # every enhancement/editing rule remains intact.
            prompt = re.sub(
                r"\*\*FIRST — there are TWO separate language decisions\.[\s\S]*?"
                r"(?=You are an expert at clarifying image editing instructions\.)",
                "",
                prompt,
                count=1,
            ).lstrip()
        elif language == "zh":
            prompt = re.sub(
                r"## Language\s+[\s\S]*?(?=## Output format)",
                "",
                prompt,
                count=1,
            )
        return (
            f"{prompt.rstrip()}\n\n## Node Output Policy (highest priority)\n"
            f"- Write all descriptive prose in `rewritten_prompt` in {output_language}.\n"
            "- The node language selector is authoritative; do not infer or change it from the user request.\n"
            "- Keep user-supplied visible text, proper nouns, brand names, and interface labels exactly as supplied.\n"
            "- Do not add readable image text unless the user supplied its exact wording or explicitly requested it.\n"
            "- Return exactly the JSON object required by the official task, on one line. Stop immediately after `}`.\n"
            "- Do not output analysis, explanations, Markdown, or phrases such as 'let me think'."
        )

    @staticmethod
    def _estimated_required_context(*, system_prompt: str, user_prompt: str,
                                    image_count: int, max_new_tokens: int) -> int:
        # Conservative preflight only; the user's context_length remains exact.
        # M-RoPE visual models cannot context-shift when the dialogue overflows.
        estimated_text_tokens = (len(system_prompt) + len(user_prompt) + 1) // 2
        estimated_vision_tokens = int(image_count) * 1344
        required = estimated_text_tokens + estimated_vision_tokens + int(max_new_tokens) + 512
        return min(131072, max(8192, ((required + 1023) // 1024) * 1024))

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
    def _normalize_image_references(text: str, image_count: int) -> str:
        value = str(text or "")
        chinese_numbers = ("一", "二", "三", "四", "五", "六", "七", "八",
                           "九", "十", "十一", "十二", "十三", "十四", "十五", "十六")
        for index in range(1, image_count + 1):
            tag = f"<image{index}>"
            patterns = [
                rf"<<\s*(?:image|picture)\s*{index}\s*>>",
                rf"<\s*(?:image|picture)\s*{index}\s*>",
                rf"(?<!<)\b(?:image|picture)\s*{index}\b(?!>)",
                rf"第\s*{index}\s*张\s*(?:图|图片|图像)",
                rf"(?:图片|图像|图)\s*{index}(?!\d)",
            ]
            if index <= len(chinese_numbers):
                patterns.append(rf"第\s*{chinese_numbers[index - 1]}\s*张\s*(?:图|图片|图像)")
            for pattern in patterns:
                value = re.sub(pattern, tag, value, flags=re.IGNORECASE)
        return value

    @staticmethod
    def _image_reference_warnings(rewritten_prompt: str, image_count: int) -> list[str]:
        if image_count < 2:
            return []
        missing = [f"<image{index}>" for index in range(1, image_count + 1)
                   if f"<image{index}>" not in rewritten_prompt]
        return (["Model output did not explicitly reference: " + ", ".join(missing)]
                if missing else [])

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
                io.Int.Input("context_length", default=8192, min=1024, max=65536, step=256),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Boolean.Input("auto_unload", default=True,
                                 tooltip="Unload the shared Qwen model after translation to release VRAM"),
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
        context_length: int,
        seed: int,
        auto_unload: bool,
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
        if config["backend"] != "qwen3.5":
            return io.NodeOutput("[Translator cached runtime requires qwen3.5]", detected)
        try:
            translated, _ = LOCAL_QWEN35_CACHE.generate(
                model_path=selection.model_path, mmproj_path=selection.mmproj_path,
                context_length=int(context_length),
                system_prompt=(
                    "You are a professional translator. Preserve meaning, tone, proper nouns, quoted text, "
                    "and formatting. Output only the translation with no explanation or JSON."
                ),
                user_prompt=translate_prompt, frames=(), max_pixels=1048576,
                temperature=0.1, top_p=0.9, top_k=20, max_tokens=2048,
                seed=seed, thinking=False, json_response=False,
            )
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            LOCAL_QWEN35_CACHE.unload()
            return io.NodeOutput(f"[Error: {e}]", detected)
        finally:
            if auto_unload:
                LOCAL_QWEN35_CACHE.unload()
        
        return io.NodeOutput(translated.strip(), detected)
    
    @staticmethod
    def _detect_language(text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'[\w\u4e00-\u9fff]', text))
        if total_chars == 0:
            return "en"
        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"

