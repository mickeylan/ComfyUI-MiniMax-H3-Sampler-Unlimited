"""Shared structured prompt skill contract for HR Endless Sampler."""

from __future__ import annotations

import json
import math
import re
from typing import Any

try:
    from .dialogue_timing import slice_dialogue_for_interval
    from .story_format import compile_h3_prompt, planned_frame_count, validate_storyboard_plan
except ImportError:  # Direct worker execution.
    from dialogue_timing import slice_dialogue_for_interval
    from story_format import compile_h3_prompt, planned_frame_count, validate_storyboard_plan


CONTINUITY_MODES = ("strict", "balanced", "cinematic")
_DESCRIPTION_FIELD = re.compile(r"(?:detailed_description|integrated_multimodal_description)\s*:", re.IGNORECASE)
_DESCRIPTION_END = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)
_DIALOGUE_TAG = re.compile(r"<d>(.*?)</d>", re.IGNORECASE | re.DOTALL)
_SPOKEN_QUOTE = re.compile(
    r"(?:说|说道|问|询问|喊|喊道|回答|答道|低语|耳语|旁白|独白|心声|says?|asks?|shouts?|replies?|whispers?|voiceover|monologue)\s*[：:]\s*[\"“](.*?)[\"”]",
    re.IGNORECASE | re.DOTALL,
)
_SPEAKER_SUBJECT = re.compile(r"(<Subject\s+\d+>)\s*\(\s*(S\d+)\s*\)", re.IGNORECASE)
_PICTURE_DECLARATION = re.compile(r"<Picture\s+(\d+)>\s*(?:是|is)\s*([^；;\r\n]+)", re.IGNORECASE)
_NAMED_SPOKEN_QUOTE = re.compile(
    r"([^\s，。；：:\"“”<>]{1,32})\s*(?:说|说道|问|询问|喊|喊道|回答|答道|低语|耳语)\s*[：:]\s*[\"“](.*?)[\"”]",
    re.DOTALL,
)


def _spoken_lines(story: str) -> tuple[str, ...]:
    matches = []
    for pattern_order, pattern in enumerate((_DIALOGUE_TAG, _SPOKEN_QUOTE)):
        for match in pattern.finditer(story):
            text = re.sub(r"^\s*\[[^\]]+\]\s*", "", match.group(1)).strip()
            if text:
                matches.append((match.start(), pattern_order, text))
    return tuple(text for _start, _pattern_order, text in sorted(matches))


def _line_spoken_duration_seconds(line: str) -> float:
    text = re.sub(r"^\s*\[[^\]]+\]\s*", "", line).strip()
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))
    punctuation_pause = sum(
        0.4 if character in "。！？!?；;" else 0.2
        for character in text if character in "，,。！？!?；;：:"
    )
    return cjk_count / 4.0 + word_count / 2.5 + punctuation_pause


def _spoken_duration_seconds(lines: tuple[str, ...]) -> float:
    if not lines:
        return 0.0
    duration = 2.0 + 0.5 * (len(lines) - 1)
    duration += sum(_line_spoken_duration_seconds(line) for line in lines)
    return math.ceil(duration * 10.0) / 10.0


def _speaker_subjects(story: str) -> dict[str, str]:
    result = {}
    for subject, speaker_id in _SPEAKER_SUBJECT.findall(story):
        subject = re.sub(r"\s+", " ", subject).title().replace("Subject ", "Subject ")
        speaker_id = speaker_id.upper()
        if speaker_id in result and result[speaker_id].casefold() != subject.casefold():
            raise ValueError(f"Source story maps speaker {speaker_id} to more than one character")
        result[speaker_id] = subject
    return result


def _picture_declarations(story: str) -> dict[int, str]:
    return {
        int(number): re.sub(r"\s+", " ", name).strip()
        for number, name in _PICTURE_DECLARATION.findall(story)
        if str(name).strip()
    }


def _named_spoken_subjects(story: str) -> tuple[tuple[str, str, str], ...]:
    declarations = _picture_declarations(story)
    result = []
    for match in _NAMED_SPOKEN_QUOTE.finditer(story):
        name, text = match.group(1).strip(), match.group(2).strip()
        picture = next((number for number, declared in declarations.items() if declared.casefold().startswith(name.casefold())), None)
        if picture is not None and text:
            result.append((text, f"<Subject {picture}>", name))
    return tuple(result)


def _source_dialogue_contract(story: str) -> tuple[dict[str, str], list[str], dict[int, str]]:
    explicit = _speaker_subjects(story)
    line_subjects = _named_spoken_subjects(story)
    subject_ids = {subject.casefold(): speaker_id for speaker_id, subject in explicit.items()}
    next_id = 1
    while f"S{next_id}" in explicit:
        next_id += 1
    subjects = []
    speaker_names = {}
    for _text, subject, name in line_subjects:
        key = subject.casefold()
        if key not in subject_ids:
            speaker_id = f"S{next_id}"
            next_id += 1
            subject_ids[key] = speaker_id
            explicit[speaker_id] = subject
        subjects.append(subject)
        speaker_names[int(re.search(r"\d+", subject).group())] = name
    return explicit, subjects, speaker_names


def prompt_skill_messages(request: dict[str, Any]) -> tuple[str, str]:
    image_count = int(request.get("image_count", 0))
    total_frames = int(request["total_frames"])
    fps = float(request["fps"])
    continuity = str(request.get("continuity_mode", "balanced"))
    language = "Chinese" if request.get("prompt_lang", "zh") == "zh" else "English"
    source_contract = {
        str(item["entity_id"]): item
        for item in request.get("source_image_contract", ())
        if isinstance(item, dict) and str(item.get("entity_id", "")).strip()
    }
    inventory = "\n".join(
        (
            f"- entity_id=asset_{i}: immutable source name={source_contract[f'asset_{i}']['name']!r}; "
            + (f"kind={source_contract[f'asset_{i}']['kind']}" if source_contract[f'asset_{i}'].get("kind") else "classify kind from this image")
            if f"asset_{i}" in source_contract else f"- entity_id=asset_{i}: connected reference {i}"
        )
        for i in range(1, image_count + 1)
    ) or "- none"
    spoken_lines = request.get("required_spoken_lines", ())
    spoken_inventory = "\n".join(f"{index}. {line}" for index, line in enumerate(spoken_lines, 1)) or "- none"
    requested_duration = float(request.get("requested_duration_seconds", request.get("duration_seconds", 0.0)))
    spoken_duration = float(request.get("minimum_spoken_duration_seconds", 0.0))
    speaker_inventory = "\n".join(
        f"- {speaker_id} is permanently bound to {subject}"
        for speaker_id, subject in request.get("required_speaker_subjects", {}).items()
    ) or "- no explicit source binding"
    system = f"""You are an HR Endless Sampler prompt skill compiler for MiniMax H3.
Convert the user's ordinary story into a strict structured shot plan. Return exactly one JSON object, no markdown.
Write descriptions in {language}, while preserving dialogue, lyrics, visible text, H3 labels, and required field names exactly.

Hard rules:
- Classify every connected asset as kind=character, scene, or prop in image_subjects, keyed only by its supplied entity_id.
- Source-declared entity names are immutable. Never move a name to another entity_id, swap two assets, or replace a declared name based on visual analysis.
- Never emit <Picture N>, <Subject N>, numeric picture/subject fields, or infer an asset from a speaker number. Those H3 labels are private compiler output.
- In every shot, pictures is an array of entity_id strings. Dialogue speaker and each visual event actor are entity_id strings.
- In prose fields, reference an asset only as <Entity asset_N>; never combine a human name with another entity marker.
- Speaker IDs such as S1 and S2 are voice identities only and have no relationship to asset_N.
- Every event has one stable ID such as S2.V1 and may start in only one shot.
- Every shot has camera, start_state, events, dialogues, end_state, forbidden_replays, audio, start_frame, and end_frame.
- Each event contains id, action, and phase (start, continue, or complete).
- Preserve every user-provided spoken line verbatim; never translate, paraphrase, shorten, or invent dialogue. A long line may be split into consecutive dialogue fragments across adjacent shots only when those fragments concatenate to the exact original line.
- The numbered mandatory spoken-line list is chronological and authoritative. dialogues across shots and within each shot must follow that exact global order; never swap speakers or reorder fragments for dramatic effect.
- Each dialogue contains id, kind, speaker, speaker_id, language, text, and delivery. kind is dialogue, monologue, or voiceover. Use stable speaker IDs S1, S2, ... across all shots.
- dialogue.text contains only the exact spoken words without quotation marks or <d> tags. dialogue.language names the spoken language, regardless of prompt_lang.
- Visible referenced speakers use speaker="asset_N". That entity must be classified as kind=character and included in the same shot's pictures list. A scene or prop can never speak.
- For kind=voiceover, the compiled prompt will state that the corresponding on-screen speaker's lips remain completely closed.
- kind=dialogue is spoken to another character; kind=monologue is audible self-directed speech with visible lip movement; kind=voiceover is off-screen narration or internal narration with no lip movement.
- Dialogue is concurrent with visual events, not a replacement visual event. Do not repeat dialogue in audio or overall_soundscape.
- A visible speaker must already be present in that shot's start_state, pictures, and description before their first spoken fragment begins. Never schedule a character to enter in a later shot after they have already spoken.
- Allocate enough shot duration for natural speech at approximately 4 Chinese characters per second or 2.5 English words per second, plus punctuation pauses. When dialogue exists, its complete natural delivery duration owns the total timeline even when shorter or longer than the requested duration. The sum of dialogue delivery time assigned to one shot must never exceed that shot's frame interval. Split long lines across consecutive shots instead of accelerating or overlapping them.
- In a two-person conversation, classify addressed questions and replies as dialogue, not monologue. Keep the question on its actual asker and the reply on its actual respondent.
- A later shot must use already/completed language instead of restarting an earlier action.
- Preserve the same camera across adjacent shots when the story action or dialogue is continuous. Change camera side, scale, height, movement, or subject arrangement only for an intentional story cut; never invent a cut merely to make adjacent camera strings different.
- Split compound actions connected by then, after, followed by, finally, or physical dependencies.
- Shots cover exactly [0,{total_frames}) with contiguous half-open frame intervals.
- Do not write [Shot N] markers or timestamps; the program owns them.
- Continuity mode is {continuity}: strict strongly forbids visual/action repetition; balanced permits necessary continuation; cinematic permits intentional match cuts only when declared.
"""
    repair = ""
    if request.get("prompt_skill_structure_repair"):
        repair = f"""

STRUCTURE CORRECTION REQUIRED
The previous response was invalid. Never return an empty object {{}}. Return the complete object with non-empty image_subjects and shots arrays.
The previous JSON failed deterministic validation: {request.get('prompt_skill_validation_error', 'invalid shot structure')}
Return one complete replacement JSON object, not a patch. Preserve the same story, dialogue order, speaker bindings, picture bindings, and creative intent.
Shot intervals must be contiguous 0-based half-open ranges that cover exactly [0,{total_frames}):
- shots[0].start_frame must equal 0
- every shot end_frame must equal the next shot start_frame
- every shot must have end_frame > start_frame
- the final shot end_frame must equal {total_frames}
- no start_frame or end_frame may exceed {total_frames}
Do not double the requested duration. Do not append a shot beginning at {total_frames}.
Reallocate long dialogue across consecutive shots so every shot has enough frames for natural delivery. Every visible speaker must be present before speaking; do not place their entrance after their first line.
Previous invalid JSON:
{str(request.get('prompt_skill_previous_response', ''))}
"""
    user = f"""Story:
--- BEGIN STORY ---
{str(request['story']).strip()}
--- END STORY ---

Target: {total_frames} frames at {fps:g} fps ({total_frames / fps:.3f} seconds on the H3 frame grid).
User-requested duration: {requested_duration:g} seconds.
Estimated duration for the exact spoken content, including natural pauses and visual lead-in/out: {spoken_duration:g} seconds.
Duration authority: {request.get('duration_source', 'user')}. When dialogue is present, its estimated natural duration owns the complete timeline and the user-requested duration is only a reference; the result may be shorter or longer. When dialogue is absent, preserve the user-requested duration.
Allocate every dialogue fragment enough frames for natural delivery. Do not repeat actions, shots, or camera moves merely to fill the user-requested duration.
Style: {request.get('style', 'cinematic realism')}.
Shot density: {request.get('shot_density', 'medium')}.
Connected pictures:
{inventory}

Mandatory spoken lines detected verbatim in the story, in authoritative chronological order. Preserve every numbered occurrence in dialogues.text and this exact global order. A long line may be divided into consecutive fragments across adjacent shots, but concatenating all fragments must reproduce the original lines exactly. Do not omit, reorder, or rewrite any character or punctuation:
{spoken_inventory}

Authoritative speaker-to-character bindings extracted from the source story. Apply these bindings to every fragment and every shot; never remap a speaker ID:
{speaker_inventory}

Return JSON with:
{{
  "image_subjects": [{{"entity_id":"asset_1","kind":"character","name":"immutable source name","observable_features":"exact visible identity, face, hair, headwear, clothing design and clothing colors"}}],
  "summary": "complete story summary",
  "retention_analysis": "identity and continuity locks",
  "shots": [{{
    "start_frame": 0,
    "end_frame": {total_frames},
    "pictures": ["asset_1"],
    "camera": "specific camera setup",
    "start_state": "visible opening state",
    "events": [{{"id":"S1.V1","actor":"asset_1","action":"one observable event using <Entity asset_1>","phase":"start"}}],
    "dialogues": [{{"id":"S1.D1","kind":"dialogue","speaker":"asset_1","speaker_id":"S1","language":"Chinese","text":"exact user-provided words","delivery":"quietly with a warm voice"}}],
    "end_state": "visible final state usable by the next shot",
    "forbidden_replays": ["completed event or old composition that must not return"],
    "audio": "ambient, physical, and non-verbal synchronized sound only; exclude dialogue",
    "description": "complete H3-ready visual shot prose using the camera, states, events, prohibitions, and non-dialogue audio"
  }}],
  "overall_soundscape": "global diegetic sound plan",
  "non_diegetic_music": "music plan or N/A",
  "warnings": []
}}
{repair}
"""
    return system, user


def _merge_trailing_zero_length_shots(value: Any, total_frames: int) -> tuple[Any, list[str]]:
    if not isinstance(value, dict) or not isinstance(value.get("shots"), list) or len(value["shots"]) < 2:
        return value, []
    shots = [dict(item) if isinstance(item, dict) else item for item in value["shots"]]
    merged = []
    while len(shots) > 1:
        tail = shots[-1]
        if not isinstance(tail, dict):
            break
        try:
            start, end = int(tail["start_frame"]), int(tail["end_frame"])
        except (KeyError, TypeError, ValueError, OverflowError):
            break
        if start < total_frames or end != start:
            break
        previous = shots[-2]
        if not isinstance(previous, dict):
            break
        combined = dict(previous)
        for name in ("pictures", "events", "dialogues", "forbidden_replays"):
            left = previous.get(name, [])
            right = tail.get(name, [])
            if isinstance(left, list) and isinstance(right, list):
                combined[name] = [*left, *right]
        for name in ("audio", "description"):
            parts = [str(item.get(name, "")).strip() for item in (previous, tail)]
            combined[name] = " ".join(part for part in parts if part)
        if str(tail.get("end_state", "")).strip():
            combined["end_state"] = str(tail["end_state"]).strip()
        shots[-2:] = [combined]
        merged.append(start)
    if not merged:
        return value, []
    normalized = dict(value)
    normalized["shots"] = shots
    return normalized, [
        "Merged zero-length trailing Qwen shot(s) at frame "
        + ", ".join(str(frame) for frame in merged)
        + " into the preceding shot without dropping events, dialogue, narration, audio, or description."
    ]


def _normalize_shot_intervals(value: Any, total_frames: int) -> tuple[Any, list[str]]:
    value, merge_warnings = _merge_trailing_zero_length_shots(value, total_frames)
    if not isinstance(value, dict) or not isinstance(value.get("shots"), list) or not value["shots"]:
        return value, merge_warnings
    shots = value["shots"]
    try:
        starts = [int(item["start_frame"]) for item in shots]
        ends = [int(item["end_frame"]) for item in shots]
    except (KeyError, TypeError, ValueError, OverflowError):
        return value, []
    warnings = list(merge_warnings)
    if starts[0] == 1 and ends[-1] == total_frames:
        starts = [start - 1 for start in starts]
        warnings.append("Converted Qwen shot intervals from 1-based inclusive coordinates to 0-based half-open ranges.")
    if any(start < 0 or start >= total_frames for start in starts) or any(left >= right for left, right in zip(starts, starts[1:])):
        return value, []
    if starts[0] != 0:
        warnings.append(f"Normalized Qwen first shot start_frame {starts[0]} to timeline origin 0.")
        starts[0] = 0
    expected_ends = [*starts[1:], total_frames]
    if ends == expected_ends and all(int(item["start_frame"]) == start for item, start in zip(shots, starts)):
        return value, warnings
    normalized = dict(value)
    normalized["shots"] = [dict(item, start_frame=start, end_frame=end) for item, start, end in zip(shots, starts, expected_ends)]
    changes = [f"shot {index} [{old_start},{old_end}) -> [{start},{end})" for index, (old_start, old_end, start, end) in enumerate(zip((int(item["start_frame"]) for item in shots), ends, starts, expected_ends), 1) if old_start != start or old_end != end]
    warnings.append("Normalized Qwen shot intervals to contiguous half-open ranges: " + ", ".join(changes) + ".")
    return normalized, warnings


def _restore_required_dialogues(value: Any, required: tuple[str, ...]) -> tuple[Any, list[str]]:
    if not required or not isinstance(value, dict) or not isinstance(value.get("shots"), list):
        return value, []
    slots = []
    for shot_index, shot in enumerate(value["shots"]):
        if not isinstance(shot, dict) or not isinstance(shot.get("dialogues", []), list):
            return value, []
        for dialogue_index, dialogue in enumerate(shot.get("dialogues", [])):
            if isinstance(dialogue, dict):
                slots.append((shot_index, dialogue_index, dialogue))
    returned = [str(dialogue.get("text", "")).strip() for _shot, _index, dialogue in slots]
    if returned == list(required) or "".join(returned) == "".join(required):
        return value, []
    if len(slots) < len(required):
        return value, []
    normalized = dict(value)
    normalized["shots"] = [dict(shot, dialogues=[dict(item) for item in shot.get("dialogues", [])]) for shot in value["shots"]]
    for required_index, text in enumerate(required):
        shot_index, dialogue_index, _dialogue = slots[required_index]
        normalized["shots"][shot_index]["dialogues"][dialogue_index]["text"] = text
    for shot_index, dialogue_index, _dialogue in reversed(slots[len(required):]):
        del normalized["shots"][shot_index]["dialogues"][dialogue_index]
    return normalized, [
        "Restored mandatory spoken lines verbatim from the source story in authoritative order; "
        "Qwen dialogue wording was not used."
    ]


def _normalize_dialogue_order(value: Any, required: tuple[str, ...]) -> tuple[Any, list[str]]:
    if not required or not isinstance(value, dict) or not isinstance(value.get("shots"), list):
        return value, []
    shot_dialogues = []
    returned = []
    for shot in value["shots"]:
        dialogues = shot.get("dialogues", []) if isinstance(shot, dict) else []
        if not isinstance(dialogues, list):
            return value, []
        shot_dialogues.append(dialogues)
        returned.extend(dialogues)
    texts = [str(item.get("text", "")).strip() if isinstance(item, dict) else "" for item in returned]
    if sorted(texts) != sorted(required) or texts == list(required):
        return value, []
    pools = {}
    for item, text in zip(returned, texts):
        pools.setdefault(text, []).append(item)
    ordered = [pools[text].pop(0) for text in required]
    normalized = dict(value)
    normalized_shots = []
    offset = 0
    for shot, dialogues in zip(value["shots"], shot_dialogues):
        count = len(dialogues)
        normalized_shots.append(dict(shot, dialogues=ordered[offset:offset + count]))
        offset += count
    normalized["shots"] = normalized_shots
    return normalized, ["Normalized Qwen spoken lines to the exact chronological order of the source story."]


def _dialogue_prefix_for_frames(text: str, available_frames: int, fps: float) -> int:
    if available_frames <= 0:
        return 0
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        frames = math.ceil(_line_spoken_duration_seconds(text[:middle]) * fps)
        if frames <= available_frames:
            low = middle
        else:
            high = middle - 1
    return low


def _redistribute_dialogues(value: Any, request: dict[str, Any]) -> tuple[Any, list[str]]:
    if float(request.get("minimum_spoken_duration_seconds", 0.0)) <= 0.0:
        return value, []
    if not isinstance(value, dict) or not isinstance(value.get("shots"), list):
        return value, []
    shots = value["shots"]
    dialogues = []
    for shot_index, shot in enumerate(shots):
        if not isinstance(shot, dict) or not isinstance(shot.get("dialogues", []), list):
            return value, []
        for dialogue in shot.get("dialogues", []):
            if isinstance(dialogue, dict) and str(dialogue.get("text", "")).strip():
                dialogues.append((shot_index, dict(dialogue)))
    if not dialogues:
        return value, []
    required_lines = [str(item) for item in request.get("required_spoken_lines", ())]
    required_subjects = [str(item) for item in request.get("required_spoken_subjects", ())]
    if required_lines and len(required_subjects) == len(required_lines):
        speaker_ids = {
            subject.casefold(): speaker_id
            for speaker_id, subject in request.get("required_speaker_subjects", {}).items()
        }
        templates = dialogues
        dialogues = []
        for line_index, (text, subject) in enumerate(zip(required_lines, required_subjects)):
            original_shot, template = templates[min(line_index, len(templates) - 1)]
            dialogue = dict(template)
            dialogue["text"] = text
            dialogue["speaker"] = subject
            dialogue["speaker_id"] = speaker_ids[subject.casefold()]
            dialogues.append((original_shot, dialogue))
    normalized = dict(value)
    normalized_shots = [dict(shot, dialogues=[]) for shot in shots]
    normalized["shots"] = normalized_shots
    fps = float(request["fps"])
    used = [0] * len(shots)
    current_shot = 0
    split_count = 0
    for _original_shot, dialogue in dialogues:
        remaining = str(dialogue["text"]).strip()
        while remaining:
            while current_shot < len(shots):
                capacity = int(shots[current_shot]["end_frame"]) - int(shots[current_shot]["start_frame"]) - used[current_shot]
                if capacity > 0:
                    break
                current_shot += 1
            if current_shot >= len(shots):
                raise ValueError("The complete video timeline is too short for the mandatory spoken dialogue at natural speed")
            required_frames = math.ceil(_line_spoken_duration_seconds(remaining) * fps)
            capacity = int(shots[current_shot]["end_frame"]) - int(shots[current_shot]["start_frame"]) - used[current_shot]
            if required_frames <= capacity:
                fragment = remaining
            else:
                if not re.search(r"[\u3400-\u9fff]", remaining):
                    current_shot += 1
                    continue
                cut = _dialogue_prefix_for_frames(remaining, capacity, fps)
                if cut <= 0:
                    current_shot += 1
                    continue
                fragment = remaining[:cut]
                split_count += 1
            item = dict(dialogue)
            item["text"] = fragment
            item["id"] = f"S{current_shot + 1}.D{len(normalized_shots[current_shot]['dialogues']) + 1}"
            normalized_shots[current_shot]["dialogues"].append(item)
            used[current_shot] += math.ceil(_line_spoken_duration_seconds(fragment) * fps)
            remaining = remaining[len(fragment):]
            if remaining:
                current_shot += 1
    returned = "".join(
        str(item.get("text", ""))
        for shot in normalized_shots for item in shot["dialogues"]
    )
    required = "".join(str(item) for item in request.get("required_spoken_lines", ()))
    if required and returned != required:
        raise ValueError("Deterministic dialogue redistribution did not preserve the exact mandatory spoken text")
    if normalized_shots == shots:
        return value, []
    return normalized, [
        f"Redistributed mandatory dialogue across shot frame capacity at natural speech speed; split {split_count} fragment(s)."
    ]


def _normalize_event_id(value: Any, shot_index: int, event_index: int) -> str:
    text = str(value or "").strip().upper().replace("_", ".").replace("-", ".")
    match = re.fullmatch(r"S?(\d+)\.?([A-Z])?(\d+)", text)
    if match:
        shot_number, kind, number = match.groups()
        return f"S{shot_number}.{kind or 'V'}{number}"
    if not text:
        return f"S{shot_index}.V{event_index}"
    return text


def _normalize_event_phase(value: Any, *, already_seen: bool) -> str:
    phase = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "begin": "start", "begins": "start", "started": "start", "new": "start",
        "ongoing": "continue", "continues": "continue", "continuing": "continue", "in progress": "continue",
        "completed": "complete", "completes": "complete", "finish": "complete", "finished": "complete", "end": "complete",
    }
    if not phase:
        return "continue" if already_seen else "start"
    return aliases.get(phase, phase)


def _normalize_boundary_states(value: Any) -> tuple[Any, list[str]]:
    if not isinstance(value, dict) or not isinstance(value.get("shots"), list):
        return value, []
    shots = value["shots"]
    normalized_shots = None
    warnings = []
    for index in range(len(shots) - 1):
        shot, next_shot = shots[index], shots[index + 1]
        if not isinstance(shot, dict) or not isinstance(next_shot, dict):
            continue
        if str(shot.get("end_state", "")).strip():
            continue
        boundary_state = str(next_shot.get("start_state", "")).strip()
        if not boundary_state:
            continue
        if normalized_shots is None:
            normalized_shots = [dict(item) if isinstance(item, dict) else item for item in shots]
        normalized_shots[index]["end_state"] = boundary_state
        warnings.append(
            f"Normalized shot {index + 1} end_state from shot {index + 2} start_state at their shared frame boundary."
        )
    if normalized_shots is None:
        return value, []
    normalized = dict(value)
    normalized["shots"] = normalized_shots
    return normalized, warnings


def _dialogue_description(item: dict[str, str]) -> str:
    speaker = item["speaker"]
    speaker_id = item["speaker_id"]
    delivery = item["delivery"]
    tagged = f"<d>[{item['language']}] {item['text']}</d>"
    if item["kind"] == "voiceover":
        return f"{speaker} ({speaker_id}) says in an off-screen voiceover, {delivery}: {tagged} while the corresponding on-screen character's lips remain completely closed."
    if item["kind"] == "monologue":
        return f"{speaker} ({speaker_id}) speaks an audible monologue, {delivery}: {tagged} with synchronized visible lip movement."
    return f"{speaker} ({speaker_id}) says, {delivery}: {tagged} with synchronized visible lip movement."


def _resolve_entity_contract(value: Any, request: dict[str, Any]) -> tuple[Any, list[str]]:
    contract = {
        str(item["entity_id"]): dict(item)
        for item in request.get("source_image_contract", ())
        if isinstance(item, dict) and str(item.get("entity_id", "")).strip()
    }
    if not contract or not isinstance(value, dict) or not isinstance(value.get("image_subjects"), list):
        raise ValueError("Prompt Skill requires the immutable source entity contract and image_subjects")

    def entity(entity_id: Any, label: str) -> dict[str, Any]:
        key = str(entity_id).strip()
        if key not in contract:
            raise ValueError(f"{label} references unknown entity_id {key!r}")
        return contract[key]

    warnings = []
    names = {}
    for entity_id, source in contract.items():
        name = str(source.get("name", "")).strip()
        if name:
            names.setdefault(name.casefold(), []).append(entity_id)

    def compile_text(text: Any, label: str) -> str:
        result = str(text).strip()
        if re.search(r"<(?:Subject|Picture)\s+\d+>", result, re.IGNORECASE):
            raise ValueError(f"{label} contains compiler-owned Subject/Picture labels")
        for entity_id, source in contract.items():
            name = str(source.get("name", "")).strip()
            if not name:
                continue
            matches = list(re.finditer(re.escape(name), result, re.IGNORECASE))
            for match in reversed(matches):
                prefix = result[:match.start()]
                marker = re.search(r"<Entity\s+([^>]+)>\s*$", prefix, re.IGNORECASE)
                if marker:
                    marked_id = marker.group(1).strip()
                    if marked_id != entity_id:
                        raise ValueError(
                            f"{label} binds {name!r} to <Entity {marked_id}> instead of <Entity {entity_id}>"
                        )
                    continue
                if len(names[name.casefold()]) != 1:
                    raise ValueError(f"{label} contains ambiguous bare source name {name!r}")
                result = result[:match.start()] + f"<Entity {entity_id}> " + result[match.start():]
                warning = f"Restored canonical <Entity {entity_id}> marker before {name!r} in {label}."
                if warning not in warnings:
                    warnings.append(warning)
        for marker in re.findall(r"<Entity\s+([^>]+)>", result, re.IGNORECASE):
            source = entity(marker, label)
            replacement = f"<Subject {int(source['picture'])}>"
            if str(source.get("name", "")).strip():
                replacement += f" {str(source['name']).strip()}"
            result = re.sub(rf"<Entity\s+{re.escape(marker)}>", replacement, result, flags=re.IGNORECASE)
        return result

    def subject_object(raw: Any, index: int) -> dict[str, Any]:
        candidate = raw
        for _ in range(2):
            if isinstance(candidate, list) and len(candidate) == 1:
                candidate = candidate[0]
                continue
            if isinstance(candidate, str):
                text = candidate.strip()
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, (dict, list)):
                    candidate = decoded
                    continue
                entity_match = re.search(r"\basset_\d+\b", text, re.IGNORECASE)
                kind_match = re.search(r"(?:^|[\s,:;|])(character|scene|prop|角色|场景|道具)(?:$|[\s,:;|])", text, re.IGNORECASE)
                if entity_match:
                    entity_id = entity_match.group().lower()
                    source = entity(entity_id, f"image_subjects[{index}]")
                    kind = str(source.get("kind") or "").strip().lower()
                    if kind_match:
                        kind = {"角色": "character", "场景": "scene", "道具": "prop"}.get(
                            kind_match.group(1), kind_match.group(1).lower()
                        )
                    if kind:
                        return {
                            "entity_id": entity_id,
                            "kind": kind,
                            "name": str(source.get("name", "")).strip(),
                            "observable_features": text,
                        }
                break
            break
        if not isinstance(candidate, dict):
            raise ValueError(
                f"image_subjects[{index}] must be an object or an unambiguous serialized object; got {raw!r}"
            )
        return candidate

    normalized = dict(value)
    normalized_subjects = []
    seen = set()
    for index, item in enumerate(value["image_subjects"], 1):
        raw = subject_object(item, index)
        if "picture" in raw or "subject" in raw:
            raise ValueError(f"image_subjects[{index}] must use entity_id; Picture/Subject numbers are compiler-owned")
        source = entity(raw.get("entity_id"), f"image_subjects[{index}]")
        entity_id = str(source["entity_id"])
        if entity_id in seen:
            raise ValueError(f"image_subjects repeats entity_id {entity_id}")
        seen.add(entity_id)
        source_name = str(source.get("name", "")).strip()
        model_name = str(raw.get("name", "")).strip()
        if source_name and model_name.casefold() != source_name.casefold():
            raise ValueError(
                f"image_subjects[{index}] renamed immutable {entity_id} from {source_name!r} to {model_name!r}"
            )
        kind = str(source.get("kind") or raw.get("kind", "")).strip().lower()
        if source.get("kind") and str(raw.get("kind", "")).strip().lower() != kind:
            warnings.append(f"Restored {entity_id} kind={kind} from its source-story binding.")
        normalized_subjects.append({
            "entity_id": entity_id,
            "picture": int(source["picture"]),
            "kind": kind,
            "name": source_name or model_name,
            "observable_features": str(raw.get("observable_features", raw.get("description", ""))).strip(),
        })
    missing = sorted(set(contract) - seen)
    if missing:
        raise ValueError("Prompt Skill omitted source entities: " + ", ".join(missing))
    resolved_contract = {item["entity_id"]: item for item in normalized_subjects}

    shots = []
    for shot_index, raw in enumerate(value.get("shots", ()), 1):
        if not isinstance(raw, dict):
            raise ValueError(f"shots[{shot_index}] must be an object")
        shot = dict(raw)
        for field in ("events", "forbidden_replays"):
            items = raw.get(field)
            if items is None:
                shot[field] = []
                warnings.append(f"Normalized missing shots[{shot_index}].{field} to an empty array.")
            elif not isinstance(items, list):
                raise ValueError(f"shots[{shot_index}].{field} must be an array")
        if not isinstance(raw.get("dialogues", []), list):
            raise ValueError(f"shots[{shot_index}].dialogues must be an array")
        shot["pictures"] = [int(entity(item, f"shots[{shot_index}].pictures")["picture"]) for item in raw.get("pictures", ())]
        shot["camera"] = compile_text(raw.get("camera", ""), f"shots[{shot_index}].camera")
        shot["start_state"] = compile_text(raw.get("start_state", ""), f"shots[{shot_index}].start_state")
        shot["end_state"] = compile_text(raw.get("end_state", ""), f"shots[{shot_index}].end_state")
        shot["description"] = compile_text(raw.get("description", ""), f"shots[{shot_index}].description")
        shot["audio"] = compile_text(raw.get("audio", ""), f"shots[{shot_index}].audio")
        shot["forbidden_replays"] = [
            compile_text(item, f"shots[{shot_index}].forbidden_replays") for item in raw.get("forbidden_replays", ())
        ]
        events = []
        shot_entity_ids = [
            item["entity_id"] for item in resolved_contract.values()
            if int(item["picture"]) in shot["pictures"] and item.get("kind") == "character"
        ]
        for event_index, raw_event in enumerate(raw.get("events", ()), 1):
            event = dict(raw_event)
            actor_id = event.get("actor")
            if not str(actor_id or "").strip() and len(shot_entity_ids) == 1:
                actor_id = shot_entity_ids[0]
                warnings.append(
                    f"Inferred missing shots[{shot_index}].events[{event_index}].actor={actor_id} from the shot's only character."
                )
            actor = resolved_contract[entity(actor_id, f"shots[{shot_index}].events[{event_index}].actor")["entity_id"]]
            if str(actor.get("kind") or "").lower() != "character":
                raise ValueError(f"shots[{shot_index}].events[{event_index}] actor {actor['entity_id']} is not a character")
            event["actor"] = str(actor["entity_id"])
            if int(actor["picture"]) not in shot["pictures"]:
                shot["pictures"].append(int(actor["picture"]))
            action = compile_text(
                event.get("action", event.get("description", "")),
                f"shots[{shot_index}].events[{event_index}].action",
            )
            event["action"] = action
            events.append(event)
        shot["events"] = events
        dialogues = []
        for dialogue_index, raw_dialogue in enumerate(raw.get("dialogues", ()), 1):
            dialogue = dict(raw_dialogue)
            speaker_id = str(dialogue.get("speaker", "")).strip()
            if speaker_id:
                legacy_subject = re.fullmatch(r"<Subject\s+(\d+)>", speaker_id, re.IGNORECASE)
                if legacy_subject:
                    speaker_picture = int(legacy_subject.group(1))
                    speaker_id = next((
                        item["entity_id"] for item in resolved_contract.values()
                        if int(item["picture"]) == speaker_picture
                    ), speaker_id)
                speaker = resolved_contract[entity(speaker_id, f"shots[{shot_index}].dialogues[{dialogue_index}].speaker")["entity_id"]]
                if str(speaker.get("kind") or "").lower() != "character":
                    raise ValueError(f"shots[{shot_index}].dialogues[{dialogue_index}] speaker {speaker['entity_id']} is not a character")
                dialogue["speaker"] = f"<Subject {int(speaker['picture'])}>"
            dialogues.append(dialogue)
        shot["dialogues"] = dialogues
        shot["description"] = " ".join(filter(None, (
            f"Camera: {shot['camera']}.",
            f"Opening state: {shot['start_state']}.",
            *(event["action"] for event in events if event["action"]),
            f"Required ending state: {shot['end_state']}.",
            "Forbidden replay: " + "; ".join(shot["forbidden_replays"]) + "." if shot["forbidden_replays"] else "",
        )))
        shots.append(shot)

    normalized.update(
        image_subjects=normalized_subjects,
        shots=shots,
        summary=compile_text(value.get("summary", ""), "summary"),
        retention_analysis=compile_text(value.get("retention_analysis", ""), "retention_analysis"),
        overall_soundscape=compile_text(value.get("overall_soundscape", ""), "overall_soundscape"),
        non_diegetic_music=compile_text(value.get("non_diegetic_music", ""), "non_diegetic_music"),
    )
    return normalized, warnings


def validate_prompt_skill_result(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    total_frames = int(request["total_frames"])
    image_count = int(request.get("image_count", 0))
    value, source_image_warnings = _resolve_entity_contract(value, request)
    value, interval_warnings = _normalize_shot_intervals(value, total_frames)
    value, boundary_state_warnings = _normalize_boundary_states(value)
    required_spoken = tuple(str(item).strip() for item in request.get("required_spoken_lines", ()) if str(item).strip())
    value, dialogue_restore_warnings = _restore_required_dialogues(value, required_spoken)
    value, dialogue_order_warnings = _normalize_dialogue_order(value, required_spoken)
    value, dialogue_timing_warnings = _redistribute_dialogues(value, request)
    plan = validate_storyboard_plan(value, image_count=image_count, total_frames=total_frames)
    required_character_subjects = {
        int(match.group(1))
        for subject in request.get("required_speaker_subjects", {}).values()
        if (match := re.fullmatch(r"<Subject\s+(\d+)>", str(subject), re.IGNORECASE))
    }
    subject_kinds = {}
    character_subjects = []
    subject_kind_warnings = []
    for index, (raw_subject, subject) in enumerate(zip(value["image_subjects"], plan["image_subjects"]), 1):
        kind = str(raw_subject.get("kind", "")).strip().lower()
        if kind not in {"character", "scene", "prop"}:
            raise ValueError(f"image_subjects[{index}].kind must be character, scene, or prop")
        subject_number = int(subject["subject"])
        if subject_number in required_character_subjects and kind != "character":
            subject_kind_warnings.append(
                f"Normalized <Subject {subject_number}> from kind={kind} to character because the source story explicitly assigns spoken dialogue to it."
            )
            kind = "character"
        subject["kind"] = kind
        subject_kinds[subject_number] = kind
        if kind == "character":
            character_subjects.append(f"<Subject {subject_number}>")
    entity_labels = {
        str(item.get("entity_id", "")): " ".join(filter(None, (
            f"<Subject {int(item['subject'])}>", str(item.get("name", "")).strip(),
        )))
        for item in plan["image_subjects"] if str(item.get("entity_id", "")).strip()
    }
    event_owner: dict[str, int] = {}
    event_actions: dict[str, str] = {}
    speaker_subjects: dict[str, str] = {}
    subject_speakers: dict[str, str] = {}
    dialogue_ids = set()
    previous_camera = None
    ledger_pending = []
    warnings = [str(item) for item in value.get("warnings", ())] if isinstance(value.get("warnings", ()), list) else []
    warnings.extend(source_image_warnings)
    warnings.extend(interval_warnings)
    warnings.extend(boundary_state_warnings)
    warnings.extend(dialogue_restore_warnings)
    warnings.extend(dialogue_order_warnings)
    warnings.extend(dialogue_timing_warnings)
    warnings.extend(subject_kind_warnings)
    for index, (raw, normalized) in enumerate(zip(value["shots"], plan["shots"]), 1):
        for name in ("camera", "start_state", "end_state"):
            if not str(raw.get(name, "")).strip():
                raise ValueError(f"shots[{index}].{name} must be non-empty")
        audio = str(raw.get("audio", "")).strip()
        if not audio:
            audio = "N/A"
            warnings.append(f"Normalized empty shots[{index}].audio to N/A; no non-dialogue sound was invented.")
        events = raw.get("events")
        dialogues = raw.get("dialogues", [])
        forbidden = raw.get("forbidden_replays")
        if not isinstance(events, list) or not isinstance(forbidden, list):
            raise ValueError(f"shots[{index}] needs events and forbidden_replays arrays")
        if not isinstance(dialogues, list):
            raise ValueError(f"shots[{index}].dialogues must be an array")
        normalized_events = []
        for event_index, event in enumerate(events, 1):
            if not isinstance(event, dict):
                raise ValueError(f"shots[{index}].events items must be objects")
            original_id = str(event.get("id", "")).strip()
            original_phase = str(event.get("phase", "")).strip()
            event_id = _normalize_event_id(original_id, index, event_index)
            action = str(event.get("action", event.get("description", ""))).strip()
            phase = _normalize_event_phase(original_phase, already_seen=event_id in event_owner)
            if not action and event_id in event_actions and phase in {"continue", "complete"}:
                action = event_actions[event_id]
                warnings.append(
                    f"Inherited missing action for {phase} event {event_id} in shot {index} from its first occurrence."
                )
            if not re.fullmatch(r"S\d+\.[A-Z]\d+", event_id) or not action or phase not in {"start", "continue", "complete"}:
                raise ValueError(
                    f"shots[{index}] has an invalid event: id={original_id!r}, phase={original_phase!r}, action_present={bool(action)}"
                )
            if event_id != original_id or phase != original_phase:
                warnings.append(
                    f"Normalized shot {index} event {event_index} from id={original_id!r}, phase={original_phase!r} "
                    f"to id={event_id}, phase={phase}."
                )
            if phase == "start" and event_id in event_owner:
                phase = "continue"
                warnings.append(
                    f"Event {event_id} was marked start again in shot {index}; normalized to continue from shot {event_owner[event_id]}."
                )
            if event_id not in event_owner:
                event_owner[event_id] = index
                event_actions[event_id] = action
                ledger_pending.append({"id": event_id, "summary": action, "owner_shot": index})
            normalized_events.append({"id": event_id, "action": action, "phase": phase})
        normalized_dialogues = []
        for dialogue_index, dialogue in enumerate(dialogues, 1):
            if not isinstance(dialogue, dict):
                raise ValueError(f"shots[{index}].dialogues items must be objects")
            item = {name: str(dialogue.get(name, "")).strip() for name in (
                "id", "kind", "speaker", "speaker_id", "language", "text", "delivery"
            )}
            repaired_fields = []
            if not item["id"]:
                item["id"] = f"S{index}.D{dialogue_index}"
                repaired_fields.append("id")
            if not item["kind"]:
                item["kind"] = "dialogue"
                repaired_fields.append("kind")
            if not item["language"] and item["text"]:
                item["language"] = "Chinese" if re.search(r"[\u3400-\u9fff]", item["text"]) else "English"
                repaired_fields.append("language")
            if not item["delivery"]:
                item["delivery"] = "自然清晰地" if item["language"] == "Chinese" else "naturally and clearly"
                repaired_fields.append("delivery")
            if not item["speaker"] and item["speaker_id"]:
                item["speaker"] = request.get("required_speaker_subjects", {}).get(item["speaker_id"], "")
                if item["speaker"]:
                    repaired_fields.append("speaker")
            if not item["speaker_id"] and item["speaker"]:
                item["speaker_id"] = subject_speakers.get(item["speaker"], "")
                if not item["speaker_id"]:
                    item["speaker_id"] = next((
                        speaker_id for speaker_id, subject in request.get("required_speaker_subjects", {}).items()
                        if subject.casefold() == item["speaker"].casefold()
                    ), "")
                if item["speaker_id"]:
                    repaired_fields.append("speaker_id")
            if repaired_fields:
                warnings.append(f"Normalized shot {index} dialogue {dialogue_index} fields: {', '.join(repaired_fields)}.")
            if not re.fullmatch(r"S\d+\.D\d+", item["id"]) or item["id"] in dialogue_ids:
                raise ValueError(f"shots[{index}] has an invalid or repeated dialogue id")
            if item["kind"] not in {"dialogue", "monologue", "voiceover"}:
                raise ValueError(f"shots[{index}] has an invalid dialogue kind")
            missing = [name for name in ("speaker", "speaker_id", "language", "text", "delivery") if not item[name]]
            if not re.fullmatch(r"S\d+", item["speaker_id"]) or missing:
                raise ValueError(f"shots[{index}] has incomplete dialogue metadata: missing={missing}, speaker_id={item['speaker_id']!r}")
            required_subject = request.get("required_speaker_subjects", {}).get(item["speaker_id"])
            if required_subject and item["speaker"].casefold() != required_subject.casefold():
                warnings.append(
                    f"Speaker {item['speaker_id']} was remapped from {item['speaker']} to authoritative source binding {required_subject}."
                )
                item["speaker"] = required_subject
            subject_match = re.fullmatch(r"<Subject\s+(\d+)>", item["speaker"], re.IGNORECASE)
            subject_number = int(subject_match.group(1)) if subject_match else 0
            if subject_kinds.get(subject_number) != "character" and not required_subject:
                speaker_number = int(item["speaker_id"][1:])
                if 1 <= speaker_number <= len(character_subjects):
                    inferred_subject = character_subjects[speaker_number - 1]
                    warnings.append(
                        f"Speaker {item['speaker_id']} was remapped from non-character {item['speaker']} "
                        f"to character picture {inferred_subject} by character order."
                    )
                    item["speaker"] = inferred_subject
                    subject_number = int(re.search(r"\d+", inferred_subject).group())
            if subject_kinds.get(subject_number) != "character":
                raise ValueError(
                    f"shots[{index}] dialogue speaker {item['speaker']} is not a character picture; "
                    "speaker IDs S1/S2 must not be confused with Subject/Picture numbers"
                )
            if "<d>" in item["text"].casefold() or "</d>" in item["text"].casefold():
                raise ValueError(f"shots[{index}] dialogue text must not contain <d> tags")
            if item["speaker"] in subject_speakers and subject_speakers[item["speaker"]] != item["speaker_id"]:
                original_speaker_id = item["speaker_id"]
                item["speaker_id"] = subject_speakers[item["speaker"]]
                warnings.append(
                    f"Normalized {item['speaker']} voice id from {original_speaker_id} to its established {item['speaker_id']}."
                )
            if item["speaker_id"] in speaker_subjects and speaker_subjects[item["speaker_id"]] != item["speaker"]:
                original_speaker_id = item["speaker_id"]
                number = 1
                while f"S{number}" in speaker_subjects:
                    number += 1
                item["speaker_id"] = f"S{number}"
                warnings.append(
                    f"Speaker id {original_speaker_id} was already bound to {speaker_subjects[original_speaker_id]}; "
                    f"normalized {item['speaker']} to unused voice id {item['speaker_id']}."
                )
            speaker_subjects[item["speaker_id"]] = item["speaker"]
            subject_speakers[item["speaker"]] = item["speaker_id"]
            dialogue_ids.add(item["id"])
            ledger_pending.append({"id": item["id"], "summary": item["text"], "owner_shot": index, "kind": item["kind"]})
            normalized_dialogues.append(item)
            if subject_number not in normalized["pictures"]:
                normalized["pictures"].append(subject_number)
        normalized["pictures"] = sorted(set(normalized["pictures"]))
        shot_start, shot_end = int(normalized["start_frame"]), int(normalized["end_frame"])
        shot_frames = shot_end - shot_start
        event_count = len(normalized_events)
        for event_index, event in enumerate(normalized_events):
            event["start_frame"] = shot_start + round(shot_frames * event_index / event_count)
            event["end_frame"] = shot_start + round(shot_frames * (event_index + 1) / event_count)
        dialogue_cursor = shot_start
        for item in normalized_dialogues:
            item["start_frame"] = dialogue_cursor
            item["end_frame"] = min(
                shot_end,
                dialogue_cursor + math.ceil(_line_spoken_duration_seconds(item["text"]) * float(request["fps"])),
            )
            dialogue_cursor = item["end_frame"]
        dialogue_frames = sum(item["end_frame"] - item["start_frame"] for item in normalized_dialogues)
        if float(request.get("minimum_spoken_duration_seconds", 0.0)) > 0.0 and dialogue_frames > shot_frames:
            raise ValueError(
                f"shots[{index}] assigns {dialogue_frames} natural-speech frames to a {shot_frames}-frame shot; "
                "split the dialogue across consecutive shots without overlap or acceleration"
            )
        camera = str(raw["camera"]).strip()
        if previous_camera and camera.casefold() == previous_camera.casefold() and request.get("continuity_mode") == "strict":
            warnings.append(
                f"Shots {index - 1} and {index} preserve the same camera in strict continuity mode; no corrective model retry was requested."
            )
        previous_camera = camera
        event_descriptions = []
        for raw_event, event in zip(events, normalized_events):
            actor_label = entity_labels.get(str(raw_event.get("actor", "")).strip(), "")
            event_descriptions.append(" ".join(filter(None, (actor_label + ":" if actor_label else "", event["action"]))))
        visual_description = " ".join(filter(None, (
            f"Camera: {camera}.",
            f"Opening state: {str(raw['start_state']).strip()}.",
            *event_descriptions,
            f"Required ending state: {str(raw['end_state']).strip()}.",
            "Forbidden replay: " + "; ".join(str(item).strip() for item in forbidden if str(item).strip()) + "." if forbidden else "",
        )))
        normalized.update(
            camera=camera,
            start_state=str(raw["start_state"]).strip(),
            end_state=str(raw["end_state"]).strip(),
            events=normalized_events,
            dialogues=normalized_dialogues,
            visual_description=visual_description,
            description=" ".join(filter(None, (
                visual_description,
                *(_dialogue_description(item) for item in normalized_dialogues),
            ))),
            forbidden_replays=[str(item).strip() for item in forbidden if str(item).strip()],
            audio=audio,
        )
    returned_spoken = [item["text"] for shot in plan["shots"] for item in shot.get("dialogues", ())]
    exact_spoken = returned_spoken == list(required_spoken)
    split_spoken = "".join(returned_spoken) == "".join(required_spoken)
    if not exact_spoken and not split_spoken:
        raise ValueError(
            "Prompt Skill must preserve every mandatory spoken-line occurrence in exact story order: "
            + " | ".join(required_spoken)
        )
    if split_spoken and not exact_spoken:
        warnings.append("Accepted chronologically adjacent dialogue fragments whose concatenation exactly preserves the source spoken text.")
    plan["summary"] = " ".join(
        str(shot.get("visual_description", "")).strip() for shot in plan["shots"]
        if str(shot.get("visual_description", "")).strip()
    )
    plan["retention_analysis"] = "\n".join(
        f"Shot {index}: opening={shot['start_state']}; ending={shot['end_state']}; forbidden="
        + ("; ".join(shot["forbidden_replays"]) or "none")
        for index, shot in enumerate(plan["shots"], 1)
    )
    plan["overall_soundscape"] = " ".join(
        str(shot["audio"]).strip() for shot in plan["shots"]
        if str(shot["audio"]).strip() and str(shot["audio"]).strip().upper() != "N/A"
    ) or "N/A"
    plan["initial_event_ledger"] = {
        "completed": [], "active": [], "pending": ledger_pending, "forbidden": [],
    }
    plan["warnings"] = warnings
    return plan


def validate_h3_identity_contract(prompt: str, plan: dict[str, Any]) -> None:
    subjects = {
        int(item["subject"]): item for item in plan.get("image_subjects", ())
        if isinstance(item, dict) and int(item.get("subject", 0) or 0) > 0
    }
    referenced = {int(number) for number in re.findall(r"<Subject\s+(\d+)>", str(prompt), re.IGNORECASE)}
    unknown = sorted(referenced - set(subjects))
    if unknown:
        raise ValueError("H3 prompt references undeclared Subject(s): " + ", ".join(map(str, unknown)))
    visual_text = re.sub(r"<d>.*?</d>", "", str(prompt), flags=re.IGNORECASE | re.DOTALL)
    for number, item in subjects.items():
        name = str(item.get("name", "")).strip()
        if name:
            for match in re.finditer(re.escape(name), visual_text, re.IGNORECASE):
                prefix = visual_text[:match.start()]
                marker = re.search(r"<Subject\s+(\d+)>\s*[\(\[]?\s*$", prefix, re.IGNORECASE)
                if marker is None:
                    continue
                if int(marker.group(1)) != number:
                    raise ValueError(
                        f"H3 identity contract violation: {name!r} is bound to Subject {number}, not Subject {marker.group(1)}"
                    )
    kinds = {number: str(item.get("kind", "")).strip().lower() for number, item in subjects.items()}
    for number in re.findall(r"<Subject\s+(\d+)>\s*\(S\d+\)", str(prompt), re.IGNORECASE):
        if kinds.get(int(number)) != "character":
            raise ValueError(f"H3 identity contract violation: Subject {number} speaks but is not a character")


def compile_prompt_skill(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    plan = validate_prompt_skill_result(value, request)
    prompt = compile_h3_prompt(plan, fps=float(request["fps"]))
    validate_h3_identity_contract(prompt, plan)
    return {
        "prompt": prompt,
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


def project_prompt_plan_interval(plan: dict[str, Any], *, frame_start: int, frame_end: int) -> dict[str, Any]:
    total_frames = int(plan.get("total_frames", plan["shots"][-1]["end_frame"]))
    if frame_start < 0 or frame_end <= frame_start or frame_end > total_frames:
        raise ValueError(f"invalid prompt-plan projection interval [{frame_start},{frame_end})")
    completed, active, pending = [], [], []
    projected_shots = []
    for shot in plan["shots"]:
        shot_start, shot_end = int(shot["start_frame"]), int(shot["end_frame"])
        for event in shot.get("events", ()):
            event_start = int(event.get("start_frame", shot_start))
            event_end = int(event.get("end_frame", shot_end))
            projected = {**event, "start_frame": event_start, "end_frame": event_end}
            if event_end <= frame_start:
                completed.append(projected)
            elif event_start >= frame_end:
                pending.append(projected)
            else:
                phase = "start" if frame_start <= event_start else "continue"
                if event_end <= frame_end:
                    phase = "complete" if phase == "continue" else "start_complete"
                active.append({**projected, "interval_phase": phase})
        if shot_start >= frame_end or shot_end <= frame_start:
            continue
        projected_shots.append({
            **shot,
            "projection_start": max(frame_start, shot_start),
            "projection_end": min(frame_end, shot_end),
        })
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "shots": projected_shots,
        "completed": completed,
        "active": active,
        "pending": pending,
        "forbidden": [f"Do not restart completed event {item['id']}." for item in completed],
    }


def _localized_shot_description(shot: dict[str, Any], frame_start: int, frame_end: int, fps: float,
                                active_events: tuple[dict[str, Any], ...]) -> str:
    parts = []
    if frame_start > int(shot["start_frame"]):
        parts.append(
            "Continue the already established shot without a cut, reframing, zoom, or restart of completed action."
        )
    if active_events:
        parts.extend(str(event["action"]).strip() for event in active_events if str(event.get("action", "")).strip())
    elif frame_start <= int(shot["start_frame"]):
        visual = str(shot.get("visual_description", shot.get("description", ""))).strip()
        if visual:
            parts.append(visual)
    else:
        parts.append("Maintain the established post-action body orientation, positions, and composition; allow only natural breathing, lip movement, and subtle expression changes.")
    for dialogue in shot.get("dialogues", ()):
        if not isinstance(dialogue, dict) or not str(dialogue.get("text", "")).strip():
            continue
        dialogue_start = int(dialogue.get("start_frame", shot["start_frame"]))
        dialogue_end = int(dialogue.get(
            "end_frame",
            min(int(shot["end_frame"]), dialogue_start + math.ceil(_line_spoken_duration_seconds(str(dialogue["text"])) * float(fps))),
        ))
        overlap_start = max(dialogue_start, int(frame_start))
        overlap_end = min(dialogue_end, int(frame_end))
        if overlap_start < overlap_end:
            fragment = slice_dialogue_for_interval(
                _dialogue_description(dialogue), dialogue_start, dialogue_end, overlap_start, overlap_end
            )
            if fragment:
                parts.append(fragment)
    return " ".join(parts)


def localize_prompt_from_plan(prompt: str, plan: dict[str, Any], *, frame_start: int, frame_end: int) -> str:
    projection = project_prompt_plan_interval(plan, frame_start=frame_start, frame_end=frame_end)
    active = projection["shots"]
    subjects = []
    for item in plan["image_subjects"]:
        picture = int(item.get("picture", 0) or 0)
        features = str(item.get("observable_features", "")).strip()
        definition = (
            f"<Subject {int(item.get('subject', picture))}> is {str(item.get('name', '')).strip()} "
            f"from <Picture {picture}>"
        )
        subjects.append(definition + (f": {features}." if features else "."))
    subjects.extend(
        line.strip()
        for line in str(prompt).splitlines()
        if re.match(r"^\s*<(?:Video|Audio)\s+\d+>\s+is\b", line, re.IGNORECASE)
    )
    local_shots = []
    localized_descriptions = []
    for index, shot in enumerate(active, 1):
        marker = f"[Shot {index}]"
        if index > 1:
            milliseconds = round(max(0, int(shot["start_frame"]) - int(frame_start)) * 1000 / float(plan["fps"]))
            minutes, remainder = divmod(milliseconds, 60000)
            seconds, milliseconds = divmod(remainder, 1000)
            marker += f" At {minutes:02d}:{seconds:02d}.{milliseconds:03d},"
        shot_events = tuple(
            event for event in projection["active"]
            if int(shot["start_frame"]) <= int(event["start_frame"]) < int(shot["end_frame"])
        )
        description = _localized_shot_description(shot, frame_start, frame_end, float(plan["fps"]), shot_events)
        localized_descriptions.append(description)
        local_shots.append(f"{marker} {description}")
    local_description = "\n".join(local_shots)
    summary = " ".join(
        str(event["action"]).strip() for event in projection["active"] if str(event.get("action", "")).strip()
    )
    retention = []
    for shot in active:
        retention.append(f"Camera contract: {str(shot['camera']).strip()}.")
        if int(frame_start) <= int(shot["start_frame"]):
            retention.append(f"Opening state: {str(shot['start_state']).strip()}.")
        else:
            retention.append("Current state: preserve the established post-action orientation, positions, and composition from the continuation frames.")
        if int(shot["end_frame"]) <= int(frame_end):
            retention.append(f"Required ending state: {str(shot['end_state']).strip()}.")
        retention.extend(
            f"Forbidden replay: {str(item).strip()}."
            for item in shot.get("forbidden_replays", ()) if str(item).strip()
        )
    retention.extend(projection["forbidden"])
    retention.append(
        f"Only the active events scheduled inside frames [{frame_start},{frame_end}) may occur; "
        "pending events must not begin and completed events must not restart."
    )
    soundscape = " ".join(
        str(shot["audio"]).strip()
        for shot in active
        if int(frame_start) <= int(shot["start_frame"]) < int(frame_end)
        and str(shot["audio"]).strip()
    ) or "N/A"
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
    spoken_lines = _spoken_lines(story)
    speaker_subjects, spoken_subjects, speaker_names = _source_dialogue_contract(story)
    picture_declarations = _picture_declarations(story)
    source_image_contract = [
        {
            "entity_id": f"asset_{picture}",
            "picture": picture,
            "name": speaker_names.get(picture, picture_declarations.get(picture, "")),
            "kind": "character" if picture in speaker_names else None,
        }
        for picture in range(1, int(image_count) + 1)
    ]
    minimum_spoken_duration = _spoken_duration_seconds(spoken_lines)
    duration_source = "dialogue" if spoken_lines else "user"
    effective_duration = minimum_spoken_duration if spoken_lines else float(duration_seconds)
    total_frames = planned_frame_count(effective_duration, fps)
    return {
        "story": story.strip(),
        "requested_duration_seconds": float(duration_seconds),
        "minimum_spoken_duration_seconds": minimum_spoken_duration,
        "duration_seconds": total_frames / float(fps),
        "duration_source": duration_source,
        "fps": float(fps),
        "total_frames": total_frames,
        "image_count": int(image_count),
        "style": str(style), "shot_density": str(shot_density),
        "continuity_mode": continuity_mode, "prompt_lang": prompt_lang,
        "required_spoken_lines": list(spoken_lines),
        "required_spoken_subjects": spoken_subjects,
        "required_speaker_subjects": speaker_subjects,
        "source_image_contract": source_image_contract,
    }
