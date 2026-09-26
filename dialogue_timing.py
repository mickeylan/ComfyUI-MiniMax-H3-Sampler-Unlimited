from __future__ import annotations

import math
import re


_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.IGNORECASE | re.DOTALL)
_LANGUAGE = re.compile(r"^(\s*\[[^\]]+\]\s*)(.*)$", re.DOTALL)
_TRAILING_PUNCTUATION = "，,。！？!?；;：:、"


def _dialogue_split_index(text: str, position: int) -> int:
    position = max(0, min(len(text), int(position)))
    candidates = []
    for index in range(max(0, position - 4), min(len(text), position + 4)):
        if text[index] in _TRAILING_PUNCTUATION:
            candidates.append(index + 1)
        elif text[index].isspace():
            candidates.append(index)
    if candidates:
        position = min(candidates, key=lambda index: (abs(index - position), index < position))
    if position == 1:
        return 0
    if position == len(text) - 1:
        return len(text)
    return position


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
    carries_in = text.startswith("<scenetrans>")
    carries_out = text.endswith("<scenetrans>")
    text = re.sub(r"^<scenetrans>\s*|\s*<scenetrans>$", "", text)
    length = len(text)
    first = max(0, min(length, math.floor(length * (overlap_start - source_start) / duration)))
    last = max(first, min(length, math.floor(length * (overlap_end - source_start) / duration)))
    if overlap_start > source_start:
        first = _dialogue_split_index(text, first)
    if overlap_end < source_end:
        last = _dialogue_split_index(text, last)
    else:
        last = length
    last = max(first, last)
    fragment = text[first:last]
    if not fragment:
        return ""
    fragment = (
        ("<scenetrans> " if carries_in and overlap_start <= source_start else "")
        + fragment
        + (" <scenetrans>" if carries_out and overlap_end >= source_end else "")
    )
    before = content[:match.start()]
    after = content[match.end():]
    if overlap_start > source_start:
        before = re.sub(
            r"(?is)^(.+?\(S\d+\))\s+.*:\s*$",
            r"\1 continues the same uninterrupted utterance from the previous chunk: ",
            before,
        )
        if before == content[:match.start()]:
            before = before.rstrip() + " continues the same uninterrupted utterance from the previous chunk: "
    if overlap_end < source_end:
        after = re.sub(r"^\s*with synchronized visible lip movement\.?", "", after, flags=re.IGNORECASE)
        after = re.sub(r"^\s*<scenetrans>\s*", "", after, flags=re.IGNORECASE)
        after = " while the same utterance continues into the next chunk without a pause or restart." + after
    return (before + f"<d>{prefix}{fragment}</d>" + after).strip()
