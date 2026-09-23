from __future__ import annotations

import math
import re


_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.IGNORECASE | re.DOTALL)
_LANGUAGE = re.compile(r"^(\s*\[[^\]]+\]\s*)(.*)$", re.DOTALL)


def dialogue_duration_seconds(content: str) -> float:
    match = _DIALOGUE.search(content)
    if match is None:
        return 0.0
    language = _LANGUAGE.match(match.group(1))
    text = (language.group(2) if language else match.group(1)).strip()
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))
    pauses = sum(
        0.4 if character in "。！？!?；;" else 0.2
        for character in text if character in "，,。！？!?；;：:"
    )
    return max(0.5, cjk_count / 4.0 + word_count / 2.5 + pauses)


def dialogue_frame_count(content: str, fps: float) -> int:
    return max(1, math.ceil(dialogue_duration_seconds(content) * float(fps)))


def slice_dialogue_for_interval(content: str, source_start: int, source_end: int,
                                overlap_start: int, overlap_end: int) -> str:
    match = _DIALOGUE.search(content)
    duration = source_end - source_start
    if match is None or duration <= 0 or overlap_start <= source_start and overlap_end >= source_end:
        return content
    tagged = match.group(1)
    language = _LANGUAGE.match(tagged)
    prefix, text = (language.group(1), language.group(2)) if language else ("", tagged)
    length = len(text)
    first = max(0, min(length, math.floor(length * (overlap_start - source_start) / duration)))
    last = max(first, min(length, math.floor(length * (overlap_end - source_start) / duration)))
    if overlap_end >= source_end:
        last = length
    fragment = text[first:last]
    if not fragment:
        return ""
    before = content[:match.start()]
    after = content[match.end():]
    if overlap_start > source_start:
        before = re.sub(
            r"(?i)\b(?:says|asks|replies|shouts|whispers)(?:\s*,[^:]*)?\s*:\s*$",
            "continues speaking: ", before,
        )
        if before == content[:match.start()]:
            before = before.rstrip() + " continues speaking: "
    if overlap_end < source_end:
        after = re.sub(r"^\s*with synchronized visible lip movement\.?", "", after, flags=re.IGNORECASE)
        after = " while the same utterance continues into the next chunk." + after
    return (before + f"<d>{prefix}{fragment}</d>" + after).strip()
