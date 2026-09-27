"""Shared-config H3 long-video prompt skill compiler node."""

from __future__ import annotations

import json

import comfy.model_management
from comfy_api.latest import io, ui

from .director_backend import resolve_director_selection
from .director_config import HRDirectorConfig, normalize_qwen38_config
from .prompt_skill import (
    CONTINUITY_MODES, build_prompt_skill_request, build_typed_prompt_plan,
    normalize_prompt_plan, validate_prompt_plan_edit,
)
from .qwen35 import Qwen35ContinuityDirector
from .reference_set import HRReferenceSet, reference_images


HRH3EventLedger = io.Custom("HR_H3_EVENT_LEDGER")
HRH3PromptPlan = io.Custom("HR_H3_PROMPT_PLAN")


class HRH3PromptSkillCompiler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRH3PromptSkillCompiler",
            display_name="HR H3 Prompt Skill Compiler",
            category="model/sampling/custom",
            description=(
                "Compile an ordinary story into a repetition-resistant MiniMax H3 prompt and event-owned shot plan. "
                "Uses exactly the Qwen3.5/3.6/3.8 backend selected by the connected HR Qwen Director Config."
            ),
            inputs=[
                io.String.Input("story", multiline=True, dynamic_prompts=True),
                io.Float.Input("duration_seconds", default=10.0, min=0.21, max=3600.0, step=0.001),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                HRReferenceSet.Input("reference_set"),
                HRDirectorConfig.Input("director_config"),
                io.String.Input("style", default="cinematic realism", advanced=True),
                io.Combo.Input("shot_density", options=["low", "medium", "high"], default="medium", advanced=True),
                io.Combo.Input("continuity_mode", options=list(CONTINUITY_MODES), default="balanced"),
                io.Combo.Input("prompt_lang", options=["zh", "en"], default="zh"),
            ],
            outputs=[
                io.String.Output(display_name="H3 prompt"),
                io.String.Output(display_name="structured shot plan"),
                HRH3EventLedger.Output(display_name="initial event ledger"),
                io.String.Output(display_name="event ledger JSON"),
                io.String.Output(display_name="validation report"),
                io.Int.Output(display_name="planned frames"),
                HRH3PromptPlan.Output(display_name="prompt plan"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, story, duration_seconds, fps, reference_set, director_config,
                style="cinematic realism", shot_density="medium", continuity_mode="balanced", prompt_lang="zh"):
        images = reference_images(reference_set)
        if not images:
            raise ValueError("HR H3 Prompt Skill Compiler requires at least one identity/reference picture")
        config = normalize_qwen38_config(director_config)
        selection = resolve_director_selection(config["backend"], config["model"], config["mmproj"])
        if selection.model_path is None or selection.mmproj_path is None:
            raise ValueError("Prompt Skill Compiler requires a local matching Qwen model and mmproj")
        request = build_prompt_skill_request(
            story, duration_seconds=duration_seconds, fps=fps, image_count=len(images),
            style=style, shot_density=shot_density, continuity_mode=continuity_mode,
            prompt_lang=prompt_lang,
        )
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        director = Qwen35ContinuityDirector(
            selection.model_path, selection.mmproj_path,
            debug=config["debug"], mtp_enabled=config["mtp"],
            mtp_draft_tokens=config["mtp_draft_tokens"],
            reasoning_effort=config["reasoning_effort"],
            cpu_moe=config["cpu_moe"], n_cpu_moe=config["n_cpu_moe"],
            backend=config["backend"],
        )
        result = director.compile_prompt_skill(request, images)
        plan = result["shot_plan"]
        warnings = result.get("warnings", ())
        report = {
            "backend": config["backend"],
            "continuity_mode": continuity_mode,
            "subjects": len(plan.get("image_subjects", ())),
            "shots": len(plan.get("shots", ())),
            "events": sum(len(item.get("events", ())) for item in plan.get("shots", ())),
            "requested_duration_seconds": request["requested_duration_seconds"],
            "minimum_spoken_duration_seconds": request["minimum_spoken_duration_seconds"],
            "planned_duration_seconds": request["duration_seconds"],
            "duration_source": request["duration_source"],
            "warnings": list(warnings),
        }
        typed_plan = build_typed_prompt_plan(result, fps=fps)
        return io.NodeOutput(
            str(result["prompt"]),
            json.dumps(plan, ensure_ascii=False, indent=2),
            result["initial_event_ledger"],
            json.dumps(result["initial_event_ledger"], ensure_ascii=False, indent=2),
            json.dumps(report, ensure_ascii=False, indent=2),
            int(result["planned_frames"]),
            typed_plan,
            ui=ui.PreviewText(json.dumps(typed_plan, ensure_ascii=False, indent=2)),
        )


class HRH3PromptPlanEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRH3PromptPlanEditor",
            display_name="HR H3 Prompt Plan Editor",
            category="model/sampling/custom",
            description=(
                "Apply manually edited JSON to an HR H3 Prompt Plan without another Qwen call. "
                "Picture/Subject identity and exact dialogue ownership/text remain immutable."
            ),
            inputs=[
                HRH3PromptPlan.Input("prompt_plan"),
                io.String.Input(
                    "edited_plan_json", multiline=True, default="",
                    tooltip=(
                        "Paste either the full prompt-plan JSON or the compiler's structured shot-plan JSON. "
                        "Leave empty to pass through the connected plan."
                    ),
                ),
            ],
            outputs=[
                HRH3PromptPlan.Output(display_name="edited prompt plan"),
                io.String.Output(display_name="validated plan JSON"),
                io.String.Output(display_name="validation report"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, prompt_plan, edited_plan_json=""):
        fps = float(prompt_plan.get("fps", 0.0))
        total_frames = int(prompt_plan.get("total_frames", 0))
        original = normalize_prompt_plan(prompt_plan, fps=fps, total_frames=total_frames)
        text = str(edited_plan_json or "").strip()
        if text:
            try:
                supplied = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"edited_plan_json is invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
                ) from error
            if not isinstance(supplied, dict):
                raise ValueError("edited_plan_json root must be an object")
            candidate = supplied if supplied.get("type") else {**original, **supplied}
            edited = normalize_prompt_plan(candidate, fps=fps, total_frames=total_frames)
            validate_prompt_plan_edit(original, edited)
        else:
            edited = original
        report = {
            "status": "valid",
            "edited": bool(text),
            "fps": fps,
            "total_frames": total_frames,
            "subjects": len(edited["image_subjects"]),
            "shots": len(edited["shots"]),
            "dialogues": sum(len(shot.get("dialogues", ())) for shot in edited["shots"]),
        }
        validated_json = json.dumps(edited, ensure_ascii=False, indent=2)
        return io.NodeOutput(
            edited,
            validated_json,
            json.dumps(report, ensure_ascii=False, indent=2),
            ui=ui.PreviewText(validated_json),
        )
