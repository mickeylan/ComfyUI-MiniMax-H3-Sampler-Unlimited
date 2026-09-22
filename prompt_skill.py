"""Shared structured prompt skill contract for HR Endless Sampler."""

from __future__ import annotations

import re
from typing import Any

try:
    from .story_format import compile_h3_prompt, planned_frame_count, validate_storyboard_plan
except ImportError:  # Direct worker execution.
    from story_format import compile_h3_prompt, planned_frame_count, validate_storyboard_plan


CONTINUITY_MODES = ("strict", "balanced", "cinematic")
_DESCRIPTION_FIELD = re.compile(r"(?:detailed_description|integrated_multimodal_description)\s*:", re.IGNORECASE)
_DESCRIPTION_END = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)


def prompt_skill_messages(request: dict[str, Any]) -> tuple[str, str]:
    image_count = int(request.get("image_count", 0))
    total_frames = int(request["total_frames"])
    fps = float(request["fps"])
    continuity = str(request.get("continuity_mode", "balanced"))
    language = "Chinese" if request.get("prompt_lang", "zh") == "zh" else "English"
    inventory = "\n".join(f"- <Picture {i}>: connected identity/reference picture {i}" for i in range(1, image_count + 1)) or "- none"
    system = f"""You are an HR Endless Sampler prompt skill compiler for MiniMax H3.
Convert the user's ordinary story into a strict structured shot plan. Return exactly one JSON object, no markdown.
Write descriptions in {language}, while preserving dialogue, lyrics, visible text, H3 labels, and required field names exactly.

Hard rules:
- Map every named character to one stable <Subject N>; never assign one name to two subjects.
- Every event has one stable ID such as S2.V1 and may start in only one shot.
- Every shot has camera, start_state, events, end_state, forbidden_replays, audio, start_frame, and end_frame.
- Each event contains id, action, and phase (start, continue, or complete).
- A later shot must use already/completed language instead of restarting an earlier action.
- Adjacent shots must differ in at least one of camera side, scale, height, movement, or subject arrangement.
- Split compound actions connected by then, after, followed by, finally, or physical dependencies.
- Shots cover exactly [0,{total_frames}) with contiguous half-open frame intervals.
- Do not write [Shot N] markers or timestamps; the program owns them.
- Continuity mode is {continuity}: strict strongly forbids visual/action repetition; balanced permits necessary continuation; cinematic permits intentional match cuts only when declared.
"""
    user = f"""Story:
--- BEGIN STORY ---
{str(request['story']).strip()}
--- END STORY ---

Target: {total_frames} frames at {fps:g} fps.
Style: {request.get('style', 'cinematic realism')}.
Shot density: {request.get('shot_density', 'medium')}.
Connected pictures:
{inventory}

Return JSON with:
{{
  "image_subjects": [{{"picture":1,"name":"Name","observable_features":"visible identity details"}}],
  "summary": "complete story summary",
  "retention_analysis": "identity and continuity locks",
  "shots": [{{
    "start_frame": 0,
    "end_frame": {total_frames},
    "pictures": [1],
    "camera": "specific camera setup",
    "start_state": "visible opening state",
    "events": [{{"id":"S1.V1","action":"one observable event","phase":"start"}}],
    "end_state": "visible final state usable by the next shot",
    "forbidden_replays": ["completed event or old composition that must not return"],
    "audio": "synchronized sound",
    "description": "complete H3-ready shot prose using the camera, states, events, prohibitions, and audio"
  }}],
  "overall_soundscape": "global diegetic sound plan",
  "non_diegetic_music": "music plan or N/A",
  "warnings": []
}}
"""
    return system, user


def validate_prompt_skill_result(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    total_frames = int(request["total_frames"])
    image_count = int(request.get("image_count", 0))
    plan = validate_storyboard_plan(value, image_count=image_count, total_frames=total_frames)
    event_owner: dict[str, int] = {}
    previous_camera = None
    ledger_pending = []
    warnings = [str(item) for item in value.get("warnings", ())] if isinstance(value.get("warnings", ()), list) else []
    for index, (raw, normalized) in enumerate(zip(value["shots"], plan["shots"]), 1):
        for name in ("camera", "start_state", "end_state", "audio"):
            if not str(raw.get(name, "")).strip():
                raise ValueError(f"shots[{index}].{name} must be non-empty")
        events = raw.get("events")
        forbidden = raw.get("forbidden_replays")
        if not isinstance(events, list) or not events or not isinstance(forbidden, list):
            raise ValueError(f"shots[{index}] needs events and forbidden_replays arrays")
        normalized_events = []
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"shots[{index}].events items must be objects")
            event_id = str(event.get("id", "")).strip()
            action = str(event.get("action", "")).strip()
            phase = str(event.get("phase", "")).strip()
            if not re.fullmatch(r"S\d+\.[A-Z]\d+", event_id) or not action or phase not in {"start", "continue", "complete"}:
                raise ValueError(f"shots[{index}] has an invalid event")
            if phase == "start" and event_id in event_owner:
                raise ValueError(f"Event {event_id} starts in more than one shot")
            event_owner.setdefault(event_id, index)
            normalized_events.append({"id": event_id, "action": action, "phase": phase})
            ledger_pending.append({"id": event_id, "summary": action, "owner_shot": index})
        camera = str(raw["camera"]).strip()
        if previous_camera and camera.casefold() == previous_camera.casefold() and request.get("continuity_mode") == "strict":
            raise ValueError(f"Shots {index - 1} and {index} repeat the same camera in strict mode")
        previous_camera = camera
        normalized.update(
            camera=camera,
            start_state=str(raw["start_state"]).strip(),
            end_state=str(raw["end_state"]).strip(),
            events=normalized_events,
            forbidden_replays=[str(item).strip() for item in forbidden if str(item).strip()],
            audio=str(raw["audio"]).strip(),
        )
    plan["initial_event_ledger"] = {
        "completed": [], "active": [], "pending": ledger_pending, "forbidden": [],
    }
    plan["warnings"] = warnings
    return plan


def compile_prompt_skill(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    plan = validate_prompt_skill_result(value, request)
    return {
        "prompt": compile_h3_prompt(plan, fps=float(request["fps"])),
        "shot_plan": plan,
        "initial_event_ledger": plan["initial_event_ledger"],
        "warnings": plan["warnings"],
        "planned_frames": int(request["total_frames"]),
    }


def build_typed_prompt_plan(compiled: dict[str, Any], *, fps: float) -> dict[str, Any]:
    plan = compiled["shot_plan"]
    return {
        "type": "HR_H3_PROMPT_PLAN",
        "version": 1,
        "fps": float(fps),
        "total_frames": int(compiled["planned_frames"]),
        "image_subjects": [dict(item) for item in plan.get("image_subjects", ())],
        "shots": [dict(item) for item in plan.get("shots", ())],
        "summary": str(plan.get("summary", "")),
        "retention_analysis": str(plan.get("retention_analysis", "")),
        "overall_soundscape": str(plan.get("overall_soundscape", "")),
        "non_diegetic_music": str(plan.get("non_diegetic_music", "")),
        "warnings": list(compiled.get("warnings", ())),
    }


def normalize_prompt_plan(value: Any, *, fps: float, total_frames: int) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "HR_H3_PROMPT_PLAN":
        raise ValueError("prompt_plan must come from HR H3 Prompt Skill Compiler or a compatible adapter")
    if value.get("version") != 1:
        raise ValueError(f"prompt_plan version {value.get('version')} is not supported")
    if abs(float(value.get("fps", fps)) - float(fps)) > 1e-6:
        raise ValueError("prompt_plan FPS does not match HR Endless Sampler FPS")
    if int(value.get("total_frames", 0)) != int(total_frames):
        raise ValueError("prompt_plan total_frames does not match the MiniMax H3 latent")
    subjects, shots = value.get("image_subjects", ()), value.get("shots", ())
    if not isinstance(subjects, (list, tuple)) or not isinstance(shots, (list, tuple)) or not shots:
        raise ValueError("prompt_plan requires image_subjects and at least one shot")
    declared_pictures = {
        int(item.get("picture", 0)) for item in subjects if isinstance(item, dict) and int(item.get("picture", 0) or 0) > 0
    }
    normalized_shots, previous_end = [], 0
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            raise ValueError(f"prompt_plan shot {index} must be an object")
        start, end = int(shot.get("start_frame", -1)), int(shot.get("end_frame", -1))
        if start != previous_end or end <= start or end > total_frames:
            raise ValueError(f"prompt_plan shot {index} has invalid interval [{start},{end})")
        required = ("camera", "start_state", "events", "end_state", "forbidden_replays", "audio", "description")
        if any(name not in shot for name in required):
            raise ValueError(f"prompt_plan shot {index} is missing required fields")
        pictures = tuple(int(item) for item in shot.get("pictures", ()))
        if any(picture not in declared_pictures for picture in pictures):
            raise ValueError(f"prompt_plan shot {index} references an undeclared picture")
        normalized_shots.append({
            **shot, "pictures": pictures, "start_frame": start, "end_frame": end,
            "cut": bool(shot.get("cut", True)),
        })
        previous_end = end
    if previous_end != total_frames:
        raise ValueError("prompt_plan shots do not cover the complete latent")
    return {**value, "image_subjects": [dict(item) for item in subjects], "shots": normalized_shots}


def prompt_plan_shots(plan: dict[str, Any]) -> list[tuple[int, int, int, str, bool]]:
    return [
        (index, shot["start_frame"], shot["end_frame"], str(shot["description"]).strip(), bool(shot["cut"]))
        for index, shot in enumerate(plan["shots"])
    ]


def _description_text(prompt: str) -> str:
    field = _DESCRIPTION_FIELD.search(prompt)
    if field is None:
        return prompt.strip()
    end = _DESCRIPTION_END.search(prompt, field.end())
    return prompt[field.end():(end.start() if end is not None else len(prompt))].strip()


def active_prompt_plan_pictures(plan: dict[str, Any], *, frame_start: int, frame_end: int) -> tuple[int, ...]:
    return tuple(sorted({
        int(picture)
        for shot in plan["shots"] if shot["start_frame"] < frame_end and shot["end_frame"] > frame_start
        for picture in shot.get("pictures", ()) if isinstance(picture, int) or str(picture).isdigit()
    }))


def filter_prompt_plan_picture_items(items: Any, active_pictures: tuple[int, ...], *, kind_key: str) -> list[Any]:
    allowed, picture_index, result = set(active_pictures), 0, []
    for item in items or ():
        if isinstance(item, dict) and item.get(kind_key) == "image":
            picture_index += 1
            if picture_index not in allowed:
                continue
        result.append(item)
    return result


def localize_prompt_from_plan(prompt: str, plan: dict[str, Any], *, frame_start: int, frame_end: int) -> str:
    active = [shot for shot in plan["shots"] if shot["start_frame"] < frame_end and shot["end_frame"] > frame_start]
    active_pictures = active_prompt_plan_pictures(plan, frame_start=frame_start, frame_end=frame_end)
    picture_map = {picture: index for index, picture in enumerate(active_pictures, 1)}
    def remap_picture(match):
        old = int(match.group(1))
        return f"<Picture {picture_map[old]}>" if old in picture_map else ""
    local_description = re.sub(r"<Picture\s+(\d+)>", remap_picture, _description_text(prompt), flags=re.IGNORECASE)
    subjects = []
    for item in plan["image_subjects"]:
        picture = int(item.get("picture", 0) or 0)
        if picture not in picture_map:
            continue
        subjects.append(
            f"<Subject {int(item.get('subject', picture))}> is {str(item.get('name', '')).strip()} "
            f"from <Picture {picture_map[picture]}>: {str(item.get('observable_features', '')).strip()}."
        )
    summary = " ".join(str(shot["description"]).strip() for shot in active if str(shot["description"]).strip())
    retention = []
    for shot in active:
        retention.extend((
            f"Camera contract: {str(shot['camera']).strip()}.",
            f"Opening state: {str(shot['start_state']).strip()}.",
            f"Required ending state: {str(shot['end_state']).strip()}.",
        ))
        retention.extend(
            f"Forbidden replay: {str(item).strip()}."
            for item in shot.get("forbidden_replays", ()) if str(item).strip()
        )
    retention.append(
        f"Only events scheduled inside frames [{frame_start},{frame_end}) may appear; "
        "subjects and events owned only by later intervals must remain absent."
    )
    soundscape = " ".join(str(shot["audio"]).strip() for shot in active if str(shot["audio"]).strip()) or "N/A"
    return "\n\n".join((
        "subject_definitions:\n" + ("\n".join(subjects) or "None."),
        "summary:\n" + (summary or "Continue only the established current interval."),
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\n" + local_description,
        "overall_soundscape:\n" + soundscape,
        "non_diegetic_music:\n" + str(plan.get("non_diegetic_music", "N/A") or "N/A").strip(),
    ))


def build_prompt_skill_request(story: str, *, duration_seconds: float, fps: float, image_count: int,
                               style: str, shot_density: str, continuity_mode: str, prompt_lang: str) -> dict[str, Any]:
    if not isinstance(story, str) or not story.strip():
        raise ValueError("Prompt Skill Compiler requires a non-empty story")
    if continuity_mode not in CONTINUITY_MODES:
        raise ValueError(f"Unknown continuity mode: {continuity_mode}")
    if prompt_lang not in {"zh", "en"}:
        raise ValueError(f"Unknown prompt language: {prompt_lang}")
    return {
        "story": story.strip(), "duration_seconds": float(duration_seconds), "fps": float(fps),
        "total_frames": planned_frame_count(duration_seconds, fps), "image_count": int(image_count),
        "style": str(style), "shot_density": str(shot_density),
        "continuity_mode": continuity_mode, "prompt_lang": prompt_lang,
    }
