"""Boundary extraction and typed data contracts for H3 video bridging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torchaudio

import comfy.model_management
import folder_paths
from comfy_api.latest import io

try:
    from .director_backend import resolve_director_selection
    from .director_config import HRDirectorConfig, normalize_qwen38_config
    from .qwen35 import Qwen35ContinuityDirector, Qwen35ObservationError, _image_url, _run_worker_once
    from .reference_set import (
        HRReferenceSet, HRMiniMaxH3ReferenceConditioning, _encode_audio, _resize,
        normalize_reference_set, reference_images,
    )
    from .video_io import HREndlessTimeline, normalize_timeline
except ImportError:  # Direct lightweight test loading.
    from director_backend import resolve_director_selection
    from director_config import HRDirectorConfig, normalize_qwen38_config
    from qwen35 import Qwen35ContinuityDirector, Qwen35ObservationError, _image_url, _run_worker_once
    from reference_set import (
        HRReferenceSet, HRMiniMaxH3ReferenceConditioning, _encode_audio, _resize,
        normalize_reference_set, reference_images,
    )
    from video_io import HREndlessTimeline, normalize_timeline


HRVideoBridgeSource = io.Custom("HR_VIDEO_BRIDGE_SOURCE")
HRVideoBridgePlan = io.Custom("HR_VIDEO_BRIDGE_PLAN")


def _save_qwen35_bridge_error(error: Qwen35ObservationError) -> Path:
    directory = Path(folder_paths.get_output_directory()) / "hr_endless_sampler"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "video_bridge_last_qwen35_error.json"
    raw = str(error.raw_json or "")
    try:
        response = json.loads(raw)
    except json.JSONDecodeError:
        response = raw
    path.write_text(json.dumps({
        "error": str(error),
        "response": response,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
BRIDGE_SOURCE_VERSION = 1
H3_BOUNDARY_FRAMES = 22


def _validate_images(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError(f"{name} must be an NHWC IMAGE batch")
    if int(value.shape[0]) < H3_BOUNDARY_FRAMES:
        raise ValueError(f"{name} must contain at least {H3_BOUNDARY_FRAMES} frames")
    return value[..., :3]


def _validate_fps(value: Any, name: str) -> float:
    fps = float(value)
    if not 1.0 <= fps <= 240.0:
        raise ValueError(f"{name} must be between 1 and 240")
    return fps


def _resample_frames(images: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    if abs(source_fps - target_fps) <= 1e-6:
        return images
    output_count = max(1, round(int(images.shape[0]) * target_fps / source_fps))
    indices = torch.linspace(0, int(images.shape[0]) - 1, output_count, device=images.device).round().long()
    return images.index_select(0, indices).contiguous()


def _audio_window(audio: Any, frame_count: int, fps: float, *, tail: bool):
    if audio is None:
        return None
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Bridge audio inputs must be ComfyUI AUDIO data")
    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 0))
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Bridge audio waveform must have shape [1, channels, samples]")
    if sample_rate <= 0:
        raise ValueError("Bridge audio sample_rate must be positive")
    samples = min(int(waveform.shape[-1]), max(1, round(frame_count * sample_rate / fps)))
    window = waveform[..., -samples:] if tail else waveform[..., :samples]
    return {"waveform": window.contiguous(), "sample_rate": sample_rate}


def normalize_bridge_source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "HR_VIDEO_BRIDGE_SOURCE":
        raise ValueError("bridge_source must be produced by HR Video Bridge Extract")
    if int(value.get("version", -1)) != BRIDGE_SOURCE_VERSION:
        raise ValueError("Unsupported HR video bridge source version")
    _validate_images(value.get("video_a"), "bridge_source video_a")
    _validate_images(value.get("video_b"), "bridge_source video_b")
    _validate_images(value.get("a_tail"), "bridge_source a_tail")
    _validate_images(value.get("b_head"), "bridge_source b_head")
    _validate_fps(value.get("fps"), "bridge_source fps")
    return value


class HRVideoBridgeExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRVideoBridgeExtract",
            display_name="HR Video Bridge Extract",
            category="image/video",
            description=(
                "Extract A's final 22 frames and B's first 22 frames for identity-aware H3 bridge planning. "
                "The original videos remain in the typed bridge source for final assembly."
            ),
            inputs=[
                io.Image.Input("video_a"),
                io.Image.Input("video_b"),
                io.Float.Input("fps_a", default=24.0, min=1.0, max=240.0, step=0.001),
                io.Float.Input("fps_b", default=24.0, min=1.0, max=240.0, step=0.001),
                io.Combo.Input(
                    "fps_policy",
                    options=["reject", "resample_b_to_a", "resample_a_to_b"],
                    default="reject",
                ),
                io.Audio.Input("audio_a", optional=True),
                io.Audio.Input("audio_b", optional=True),
            ],
            outputs=[
                HRVideoBridgeSource.Output(display_name="bridge_source"),
                io.Image.Output(display_name="A tail (22 frames)"),
                io.Image.Output(display_name="B head (22 frames)"),
                io.Audio.Output(display_name="A tail audio"),
                io.Audio.Output(display_name="B head audio"),
                io.String.Output(display_name="diagnostic"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, video_a, video_b, fps_a=24.0, fps_b=24.0, fps_policy="reject", audio_a=None, audio_b=None):
        video_a = _validate_images(video_a, "video_a")
        video_b = _validate_images(video_b, "video_b")
        fps_a = _validate_fps(fps_a, "fps_a")
        fps_b = _validate_fps(fps_b, "fps_b")
        if fps_policy not in {"reject", "resample_b_to_a", "resample_a_to_b"}:
            raise ValueError(f"Unknown bridge FPS policy: {fps_policy}")
        if abs(fps_a - fps_b) > 1e-6:
            if fps_policy == "reject":
                raise ValueError(f"Video bridge FPS mismatch: A={fps_a:g}, B={fps_b:g}")
            if fps_policy == "resample_b_to_a":
                video_b = _resample_frames(video_b, fps_b, fps_a)
                fps_b = fps_a
            else:
                video_a = _resample_frames(video_a, fps_a, fps_b)
                fps_a = fps_b
        if int(video_a.shape[0]) < H3_BOUNDARY_FRAMES or int(video_b.shape[0]) < H3_BOUNDARY_FRAMES:
            raise ValueError("FPS conversion left fewer than 22 bridge boundary frames")

        fps = fps_a
        a_tail = video_a[-H3_BOUNDARY_FRAMES:].contiguous()
        b_head = video_b[:H3_BOUNDARY_FRAMES].contiguous()
        a_tail_audio = _audio_window(audio_a, H3_BOUNDARY_FRAMES, fps, tail=True)
        b_head_audio = _audio_window(audio_b, H3_BOUNDARY_FRAMES, fps, tail=False)
        source = {
            "type": "HR_VIDEO_BRIDGE_SOURCE",
            "version": BRIDGE_SOURCE_VERSION,
            "fps": fps,
            "analysis_frames": H3_BOUNDARY_FRAMES,
            "video_a": video_a,
            "video_b": video_b,
            "audio_a": audio_a,
            "audio_b": audio_b,
            "a_tail": a_tail,
            "b_head": b_head,
            "a_tail_audio": a_tail_audio,
            "b_head_audio": b_head_audio,
            "source_a_frame_count": int(video_a.shape[0]),
            "source_b_frame_count": int(video_b.shape[0]),
        }
        diagnostic = {
            "fps": fps,
            "analysis_frames": H3_BOUNDARY_FRAMES,
            "video_a": {"frames": int(video_a.shape[0]), "height": int(video_a.shape[1]), "width": int(video_a.shape[2])},
            "video_b": {"frames": int(video_b.shape[0]), "height": int(video_b.shape[1]), "width": int(video_b.shape[2])},
            "audio_a": a_tail_audio is not None,
            "audio_b": b_head_audio is not None,
        }
        return io.NodeOutput(source, a_tail, b_head, a_tail_audio, b_head_audio, json.dumps(diagnostic, ensure_ascii=False, indent=2))


def normalize_bridge_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "HR_VIDEO_BRIDGE_PLAN":
        raise ValueError("bridge_plan must be produced by HR Video Bridge Director")
    if int(value.get("version", -1)) != 1:
        raise ValueError("Unsupported HR video bridge plan version")
    frames = int(value.get("transition_frames", -1))
    if frames not in {22, 39, 56, 73}:
        raise ValueError("Bridge transition_frames must be 22, 39, 56, or 73")
    if not str(value.get("h3_prompt", "")).strip():
        raise ValueError("Bridge plan has an empty H3 prompt")
    return value


class HRVideoBridgeDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRVideoBridgeDirector",
            display_name="HR Video Bridge Director",
            category="model/sampling/custom",
            description="Use Qwen3.6/3.8 to analyze A's tail, B's head, and identity pictures, then write an H3 bridge plan.",
            inputs=[
                HRVideoBridgeSource.Input("bridge_source"),
                HRReferenceSet.Input("reference_set"),
                HRDirectorConfig.Input("director_config"),
                io.Combo.Input("transition_frames", options=[22, 39, 56, 73], default=39),
                io.Combo.Input("transition_strategy", options=["auto", "continuous_action", "foreground_occlusion", "whip_pan", "camera_follow", "light_transition", "match_cut"], default="auto"),
                io.Combo.Input("identity_policy", options=["strict", "balanced", "follow_video"], default="strict"),
                io.Combo.Input("clothing_policy", options=["follow_video", "lock_a", "arrive_at_b", "follow_reference"], default="follow_video"),
                io.Combo.Input("prompt_lang", options=["zh", "en"], default="zh"),
                io.String.Input("user_instruction", default="", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                HRVideoBridgePlan.Output(display_name="bridge_plan"),
                io.String.Output(display_name="H3 prompt"),
                io.String.Output(display_name="analysis JSON"),
                io.String.Output(display_name="risk report"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, bridge_source, reference_set, director_config, transition_frames=39,
                transition_strategy="auto", identity_policy="strict", clothing_policy="follow_video",
                prompt_lang="zh", user_instruction=""):
        source = normalize_bridge_source(bridge_source)
        refs = normalize_reference_set(reference_set)
        config = normalize_qwen38_config(director_config)
        if config["backend"] not in {"qwen3.5", "qwen3.6", "qwen3.8"}:
            raise ValueError("HR Video Bridge Director requires Qwen3.5, Qwen3.6, or Qwen3.8")
        transition_frames = int(transition_frames)
        if transition_frames not in {22, 39, 56, 73}:
            raise ValueError("Bridge transition_frames must be 22, 39, 56, or 73")
        pictures = reference_images(refs)
        selection = resolve_director_selection(config["backend"], config["model"], config["mmproj"])
        if selection.model_path is None or selection.mmproj_path is None:
            raise ValueError("Video Bridge Director requires a local Qwen GGUF and same-family mmproj")

        if config["backend"] == "qwen3.5":
            target_height, target_width = source["a_tail"].shape[1:3]
            destination = torch.nn.functional.interpolate(
                source["b_head"].movedim(-1, 1), size=(target_height, target_width),
                mode="bilinear", align_corners=False,
            ).movedim(1, -1)
            identity_images = [torch.nn.functional.interpolate(
                image.movedim(-1, 1), size=(target_height, target_width),
                mode="bilinear", align_corners=False,
            ).movedim(1, -1)[0] for image in pictures]
            paired_boundaries = torch.cat((source["a_tail"], destination), dim=2)
            identity_canvases = [torch.cat((image, image), dim=1).unsqueeze(0) for image in identity_images]
            observations = torch.cat((paired_boundaries, *identity_canvases), dim=0)
            output_language = "Chinese" if prompt_lang == "zh" else "English"
            instruction = (
                f"Generate a {transition_frames}-frame bridge beginning at video A's visible final state and ending at "
                f"video B's visible opening state. Each of the first 22 chronological comparison images has A on the "
                f"left and B on the right. Strategy={transition_strategy}; identity={identity_policy}; "
                f"clothing={clothing_policy}. Use the chronological frames only to infer A's outgoing motion and B's incoming "
                f"motion. The H3 generation itself is first/last-frame generation: A's final frame is the exact start anchor and "
                f"B's first frame is the exact end anchor. Plan only the minimum visible state change required between those two "
                f"anchors; do not invent a new shot, secondary action, flourish, or narrative event. Keep camera scale, lens "
                f"perspective, screen direction, horizon, and axis stable through the bridge. Do not use a push-in, pull-back, "
                f"crash zoom, rapid dolly, whip pan, orbit, or abrupt reframing merely to reconcile A and B. Prefer the shortest "
                f"continuous subject motion and posture adjustment that reaches B. Change framing only when the chronological A/B "
                f"boundary frames prove the same camera motion is already underway. End exactly at B's first visible frame and do "
                f"not perform or replay any action that occurs after B begins; B itself owns that continuation. The executable H3 "
                f"prompt must state the exact A-end state, exact B-start state, minimum required motion, fixed camera contract, and "
                f"the ban on replaying B's later action. Identity pictures are identity-only evidence: never show a reference sheet, "
                f"turnaround, four-view layout, split screen, panel, collage, character card, or the pictures themselves in "
                f"the generated video. Describe one continuous cinematic scene only. Write descriptive content in "
                f"{output_language}, but the executable H3 prompt must begin with the exact English label [Shot 1]. "
                f"{str(user_instruction or '').strip()}"
            )
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            director = Qwen35ContinuityDirector(
                selection.model_path, selection.mmproj_path, debug=config["debug"],
                mtp_enabled=False, backend="qwen3.5", context_tokens=131072,
            )
            try:
                directed = director.external_video_continuation(
                    instruction, observations,
                    source={
                        "fps": float(source["fps"]), "source_frames": int(source["source_a_frame_count"]),
                        "tail_frames": H3_BOUNDARY_FRAMES, "tail_observation_count": H3_BOUNDARY_FRAMES,
                        "destination_frame_count": H3_BOUNDARY_FRAMES,
                        "reference_image_count": len(identity_images), "transition_frames": transition_frames,
                        "prompt_lang": prompt_lang,
                    },
                    reference_summary=(
                        f"The first {H3_BOUNDARY_FRAMES} chronological comparison images pair A on the left with B on the right; "
                        f"then {len(identity_images)} identity-only pictures follow. B is the required destination. The identity "
                        f"pictures must never become visible content, framing, layout, panels, split screens, collages, or shots."
                    ),
                )
            except Qwen35ObservationError as error:
                path = _save_qwen35_bridge_error(error)
                raise Qwen35ObservationError(
                    f"{error}; raw response saved to {path}", raw_json=error.raw_json,
                ) from error
            plan = {
                "type": "HR_VIDEO_BRIDGE_PLAN", "version": 1,
                "transition_frames": transition_frames, "strategy": transition_strategy,
                "analysis": {
                    "confidence": directed.confidence,
                    "observed_end_state": directed.observed_end_state,
                    "transition_plan": directed.transition_plan,
                },
                "constraints": list(directed.observed_end_state.get("must_continue", ())),
                "h3_prompt": directed.h3_prompt,
                "risk_report": "",
                "reference_labels": {
                    "pictures": [f"<Picture {index}>" for index in range(1, len(pictures) + 1)],
                    "video_a": "<Video 2>", "video_b": "<Video 1>",
                },
                "source_fps": float(source["fps"]), "source": source,
            }
            return io.NodeOutput(
                plan, directed.h3_prompt,
                json.dumps(plan["analysis"], ensure_ascii=False, indent=2), "",
            )

        images = [image[0] for image in pictures]
        images.extend(frame for frame in source["a_tail"])
        images.extend(frame for frame in source["b_head"])
        request = {
            "operation": "video_bridge",
            "director_backend": config["backend"],
            "director_model_path": str(selection.model_path),
            "director_mmproj_path": str(selection.mmproj_path),
            "director_mtp": config["mtp"],
            "director_mtp_draft_tokens": config["mtp_draft_tokens"],
            "director_reasoning_effort": config["reasoning_effort"],
            "director_cpu_moe": config["cpu_moe"],
            "director_n_cpu_moe": config["n_cpu_moe"],
            "reference_image_count": len(pictures),
            "a_frame_count": H3_BOUNDARY_FRAMES,
            "b_frame_count": H3_BOUNDARY_FRAMES,
            "transition_frames": transition_frames,
            "transition_strategy": str(transition_strategy),
            "identity_policy": str(identity_policy),
            "clothing_policy": str(clothing_policy),
            "prompt_lang": str(prompt_lang),
            "user_instruction": str(user_instruction or "").strip(),
            "image_urls": [_image_url(frame) for frame in images],
        }
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        process, value = _run_worker_once(request, timeout=600)
        native_failure = value is None or value.get("error_type") not in {"Qwen35ObservationError", "Qwen35DependencyError", "ValueError"}
        if config["mtp"] and native_failure:
            request["director_mtp"] = False
            process, value = _run_worker_once(request, timeout=600)
        if value is None:
            detail = str(getattr(process, "stderr", "") or getattr(process, "stdout", "") or "no worker output")[-4000:]
            raise RuntimeError(f"Qwen video bridge worker returned no result: {detail}")
        if not value.get("ok"):
            raise RuntimeError(str(value.get("message", "Qwen video bridge worker failed")))
        result = dict(value["video_bridge"])
        plan = {
            "type": "HR_VIDEO_BRIDGE_PLAN",
            **result,
            "source_fps": float(source["fps"]),
            "source": source,
        }
        return io.NodeOutput(
            plan,
            str(result["h3_prompt"]),
            json.dumps(result.get("analysis", {}), ensure_ascii=False, indent=2),
            str(result.get("risk_report", "")),
        )


def _bridge_reference_set(reference_set: Any, source: dict[str, Any]) -> dict[str, Any]:
    refs = normalize_reference_set(reference_set)
    if len(reference_images(refs)) > 9:
        raise ValueError("MiniMax H3 supports at most 9 identity pictures")
    return {
        "version": 1,
        "images": refs["images"],
        "videos": (),
        "video_audios": (),
        "audios": (),
        "ref_image_size": refs["ref_image_size"],
        "ref_scale": refs["ref_scale"],
    }


class HRVideoBridgeConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRVideoBridgeConditioning",
            display_name="HR Video Bridge Conditioning",
            category="model/sampling/custom",
            description="Build H3 first/last-frame conditioning: A's final frame anchors the start and B's first frame anchors the destination.",
            inputs=[
                io.Clip.Input("clip"), io.Vae.Input("vae"), io.Vae.Input("audio_vae"),
                HRVideoBridgePlan.Input("bridge_plan"), HRReferenceSet.Input("reference_set"),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Combo.Input("audio_mode", options=["continue", "mute"], default="continue"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="latent"),
                io.Custom("HR_H3_EXTERNAL_CONTINUATION").Output(display_name="external_continuation"),
                HRReferenceSet.Output(display_name="bridge_reference_set"), io.String.Output(display_name="prompt"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, bridge_plan, reference_set, width=1344, height=768, audio_mode="continue"):
        plan = normalize_bridge_plan(bridge_plan)
        source = normalize_bridge_source(plan["source"])
        refs = _bridge_reference_set(reference_set, source)
        prompt = str(plan["h3_prompt"])
        conditioned = HRMiniMaxH3ReferenceConditioning.execute(
            clip, vae, audio_vae, prompt, int(width), int(height), int(plan["transition_frames"]), refs
        ).result
        positive, latent, normalized_refs = conditioned
        samples = latent["samples"].unbind()
        target_video = samples[0]
        resized_a = _resize(source["a_tail"][-1:], int(width), int(height), "disabled")
        resized_b = _resize(source["b_head"][:1], int(width), int(height), "disabled")
        video_context = vae.encode(resized_a)
        keyframes = [
            {"resolved_frame_index": 0, "latent": video_context},
            {"resolved_frame_index": int(plan["transition_frames"]) - 1, "latent": vae.encode(resized_b)},
        ]
        audio_context = None
        updated_positive = []
        for embedding, metadata in positive:
            updated = dict(metadata)
            existing_keyframes = [dict(item) for item in updated.get("minimax_keyframes", ())]
            anchor_indices = {item["resolved_frame_index"] for item in keyframes}
            existing_keyframes = [item for item in existing_keyframes if not (
                item.get("resolved_frame_index") in anchor_indices and item.get("latent") is not None
            )]
            updated["minimax_keyframes"] = [*existing_keyframes, *keyframes]
            updated_positive.append([embedding, updated])
        continuation = {
            "type": "HR_H3_EXTERNAL_CONTINUATION", "version": 1,
            "video_context": video_context, "video_context_start": 0,
            "audio_context": audio_context, "audio_context_start": 0,
            "audio_mode": "mute", "visual_references_only": True, "tail_images": resized_a,
            "source_audio_tail": source.get("a_tail_audio"), "prompt": prompt,
            "analysis": plan.get("analysis", {}), "target_frames": int(plan["transition_frames"]),
            "reference_set": normalized_refs,
            "source_fps": float(source["fps"]), "source_frames": int(source["source_a_frame_count"]),
            "tail_frames": 1,
        }
        if target_video.shape[2] < 2:
            raise ValueError("Bridge target latent is too short for MiniMax H3")
        return io.NodeOutput(updated_positive, latent, continuation, normalized_refs, prompt)


def _frame_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left[..., :3].movedim(-1, 0).unsqueeze(0).float()
    right = right[..., :3].movedim(-1, 0).unsqueeze(0).float()
    size = (min(64, left.shape[-2], right.shape[-2]), min(64, left.shape[-1], right.shape[-1]))
    left = torch.nn.functional.adaptive_avg_pool2d(left, size)
    right = torch.nn.functional.adaptive_avg_pool2d(right, size)
    color = (left.mean(dim=(-2, -1)) - right.mean(dim=(-2, -1))).abs().mean()
    return float((left - right).abs().mean() + 0.25 * color)


def _preserve_bridge_span(start: int, end: int, total: int, minimum: int) -> tuple[int, int]:
    minimum = min(max(1, int(minimum)), int(total))
    start = max(0, min(int(start), int(total)))
    end = max(start, min(int(end), int(total)))
    if end - start >= minimum:
        return start, end
    center = (start + end) // 2
    start = max(0, min(center - minimum // 2, int(total) - minimum))
    return start, start + minimum


def _best_seam(left: torch.Tensor, right: torch.Tensor, search: int) -> dict[str, Any]:
    left_start = max(0, int(left.shape[0]) - int(search))
    right_end = min(int(right.shape[0]), int(search))
    best = None
    for li in range(left_start, int(left.shape[0])):
        for ri in range(right_end):
            score = _frame_distance(left[li], right[ri])
            candidate = {"left_index": li, "right_index": ri, "score": score}
            if best is None or score < best["score"]:
                best = candidate
    if best is None:
        raise ValueError("Could not search an empty video seam")
    return best


def _best_sequence_seam(left: torch.Tensor, right: torch.Tensor, search: int) -> dict[str, Any]:
    left_start = max(0, int(left.shape[0]) - int(search))
    right_end = min(int(right.shape[0]), int(search))
    best = None
    for left_index in range(left_start, int(left.shape[0])):
        for right_start in range(right_end):
            count = min(int(left.shape[0]) - left_index, right_end - right_start)
            if count < 3:
                continue
            score = sum(_frame_distance(left[left_index + offset], right[right_start + offset])
                        for offset in range(count)) / count
            candidate = {
                "left_index": left_index + count - 1,
                "right_index": right_start + count - 1,
                "score": score,
                "overlap_frames": count,
            }
            if best is None or (score, -count) < (best["score"], -best["overlap_frames"]):
                best = candidate
    return best if best is not None else _best_seam(left, right, search)


def _audio_tensor(audio: Any, sample_rate: int, channels: int) -> torch.Tensor:
    if audio is None:
        return torch.zeros(channels, 0)
    waveform = audio["waveform"]
    source_rate = int(audio["sample_rate"])
    waveform = waveform[0] if waveform.ndim == 3 else waveform
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    if waveform.shape[0] == 1 and channels > 1:
        waveform = waveform.expand(channels, -1)
    if waveform.shape[0] != channels:
        raise ValueError("Bridge audio channel counts do not match")
    return waveform


def _audio_segment(audio: Any, start_frame: int, end_frame: int, fps: float, sample_rate: int, channels: int) -> torch.Tensor:
    expected = max(0, round((end_frame - start_frame) * sample_rate / fps))
    waveform = _audio_tensor(audio, sample_rate, channels)
    start = max(0, round(start_frame * sample_rate / fps))
    segment = waveform[:, start:start + expected]
    if segment.shape[-1] < expected:
        segment = torch.nn.functional.pad(segment, (0, expected - segment.shape[-1]))
    return segment


def _edge_fade(parts: list[torch.Tensor], fade_samples: int) -> list[torch.Tensor]:
    result = [part.clone() for part in parts]
    for index, part in enumerate(result):
        fade = min(fade_samples, part.shape[-1] // 2)
        if not fade:
            continue
        if index > 0:
            part[:, :fade] *= torch.linspace(0, 1, fade, device=part.device, dtype=part.dtype)
        if index + 1 < len(result):
            part[:, -fade:] *= torch.linspace(1, 0, fade, device=part.device, dtype=part.dtype)
    return result


def _crossfade_parts(parts, overlap, dim):
    overlaps = [int(overlap)] * (len(parts) - 1) if isinstance(overlap, int) else [int(value) for value in overlap]
    if len(overlaps) != len(parts) - 1:
        raise ValueError("Crossfade overlap count must match the number of seams")
    output = parts[0]
    used = []
    for part, requested in zip(parts[1:], overlaps):
        count = min(requested, int(output.shape[dim]), int(part.shape[dim]))
        used.append(count)
        if count <= 0:
            output = torch.cat((output, part), dim=dim)
            continue
        left = output.narrow(dim, output.shape[dim] - count, count)
        right = part.narrow(dim, 0, count).to(left)
        shape = [1] * left.ndim
        shape[dim] = count
        weight = (0.5 - 0.5 * torch.cos(
            torch.linspace(0.0, torch.pi, count + 2, device=left.device, dtype=left.dtype)[1:-1]
        )).reshape(shape)
        blended = left * (1.0 - weight) + right * weight
        output = torch.cat((output.narrow(dim, 0, output.shape[dim] - count), blended,
                            part.narrow(dim, count, part.shape[dim] - count)), dim=dim)
    return output, used


def _source_bridge_audio(audio_a, audio_b, sample_rate, channels, sample_count):
    left = _audio_tensor(audio_a, sample_rate, channels)
    right = _audio_tensor(audio_b, sample_rate, channels)
    left = left[:, -sample_count:]
    right = right[:, :sample_count]
    if left.shape[-1] < sample_count:
        left = torch.nn.functional.pad(left, (sample_count - left.shape[-1], 0))
    if right.shape[-1] < sample_count:
        right = torch.nn.functional.pad(right, (0, sample_count - right.shape[-1]))
    fade = torch.linspace(0.0, 1.0, sample_count, device=left.device, dtype=left.dtype).unsqueeze(0)
    return left * (1.0 - fade) + right.to(left) * fade


class HRVideoBridgeAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRVideoBridgeAssemble", display_name="HR Video Bridge Assemble", category="image/video",
            description="Find low-difference A/bridge/B seams, trim duplicate endpoint frames, and assemble synchronized audio.",
            inputs=[
                HRVideoBridgeSource.Input("bridge_source"), io.Image.Input("bridge_frames"),
                io.Audio.Input("bridge_audio", optional=True),
                io.Combo.Input("assemble_policy", options=["auto_seam", "hard_cut_debug"], default="auto_seam"),
                io.Int.Input("seam_search_frames", default=12, min=1, max=22, step=1),
                io.Combo.Input("audio_policy", options=["mute_bridge", "source_crossfade", "use_bridge_audio"], default="mute_bridge"),
                io.Float.Input("fade_seconds", default=0.25, min=0.0, max=2.0, step=0.01),
                io.Int.Input("video_blend_frames", default=3, min=0, step=1, tooltip="Requested crossfade length per video seam; execution clamps it to the frames actually available on both sides."),
                io.Int.Input("min_bridge_frames", default=22, min=5, max=73, step=1),
            ],
            outputs=[io.Image.Output(display_name="frames"), io.Audio.Output(display_name="audio"),
                     HREndlessTimeline.Output(display_name="timeline"), io.String.Output(display_name="seam report")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, bridge_source, bridge_frames, bridge_audio=None, assemble_policy="auto_seam",
                seam_search_frames=12, audio_policy="mute_bridge", fade_seconds=0.25,
                video_blend_frames=3, min_bridge_frames=22):
        source = normalize_bridge_source(bridge_source)
        bridge = _validate_images(bridge_frames, "bridge_frames")
        a, b = source["video_a"], source["video_b"]
        if a.shape[1:] != bridge.shape[1:] or b.shape[1:] != bridge.shape[1:]:
            raise ValueError("A, bridge, and B must have identical frame dimensions before assembly")
        if assemble_policy == "auto_seam":
            left = _best_seam(a, bridge, seam_search_frames)
            right = _best_sequence_seam(bridge, b, seam_search_frames)
            a_end = int(a.shape[0])
            bridge_start = left["right_index"] + 1
            bridge_end = right["left_index"] + 1
            b_start = right["right_index"] + 1
            bridge_start, bridge_end = _preserve_bridge_span(
                bridge_start, bridge_end, int(bridge.shape[0]), int(min_bridge_frames)
            )
        elif assemble_policy == "hard_cut_debug":
            a_end, bridge_start, bridge_end, b_start = int(a.shape[0]), 0, int(bridge.shape[0]), 0
            left = {"left_index": a_end - 1, "right_index": 0, "score": _frame_distance(a[-1], bridge[0])}
            right = {"left_index": bridge_end - 1, "right_index": 0, "score": _frame_distance(bridge[-1], b[0])}
        else:
            raise ValueError(f"Unknown bridge assemble policy: {assemble_policy}")
        parts = [a[:a_end], bridge[bridge_start:bridge_end], b[b_start:]]
        fps = float(source["fps"])
        blend_frames = int(video_blend_frames) if assemble_policy == "auto_seam" else 0
        seam_blends = [0, blend_frames]
        frames, video_overlaps = _crossfade_parts(parts, seam_blends, 0)

        if audio_policy == "source_crossfade":
            audio_inputs = [source.get("audio_a"), None, source.get("audio_b")]
        else:
            audio_inputs = [source.get("audio_a"), None if audio_policy == "mute_bridge" else bridge_audio, source.get("audio_b")]
        available = [item for item in audio_inputs if isinstance(item, dict) and isinstance(item.get("waveform"), torch.Tensor)]
        audio = None
        audio_info = None
        if available:
            sample_rate = int(available[0]["sample_rate"])
            channels = max(int(item["waveform"].shape[-2]) for item in available)
            ranges = [(0, a_end), (bridge_start, bridge_end), (b_start, int(b.shape[0]))]
            audio_parts = [_audio_segment(item, start, end, fps, sample_rate, channels)
                           for item, (start, end) in zip(audio_inputs, ranges)]
            if audio_policy == "source_crossfade":
                audio_parts[1] = _source_bridge_audio(
                    source.get("audio_a"), source.get("audio_b"), sample_rate, channels, audio_parts[1].shape[-1]
                )
            elif audio_policy == "mute_bridge":
                audio_parts[1].zero_()
            audio_parts = _edge_fade(audio_parts, round(float(fade_seconds) * sample_rate))
            audio_overlap = round(blend_frames * sample_rate / fps)
            waveform, audio_overlaps = _crossfade_parts(audio_parts, [0, audio_overlap], -1)
            waveform = waveform.unsqueeze(0)
            expected = round(int(frames.shape[0]) * sample_rate / fps)
            waveform = waveform[..., :expected]
            if waveform.shape[-1] < expected:
                waveform = torch.nn.functional.pad(waveform, (0, expected - waveform.shape[-1]))
            audio = {"waveform": waveform, "sample_rate": sample_rate}
            audio_info = {"policy": audio_policy, "sample_rate": sample_rate, "channels": channels,
                          "fade_seconds": float(fade_seconds), "video_blend_frames": blend_frames,
                          "overlap_samples": audio_overlaps, "total_samples": expected}

        lengths = [int(parts[0].shape[0]) - video_overlaps[0], int(parts[1].shape[0]),
                   int(parts[2].shape[0]) - video_overlaps[1]]
        starts = [0, lengths[0], lengths[0] + lengths[1]]
        timeline_value = {
            "schema_version": 1, "producer": "HR Video Bridge Assemble", "fps": fps,
            "total_frames": int(frames.shape[0]), "chunks": [], "shots": [],
            "segments": [
                {"type": "source_a", "start": starts[0], "end": starts[1] - 1, "source_start": 0, "source_end": a_end - 1},
                {"type": "generated_bridge", "start": starts[1], "end": starts[2] - 1, "source_start": bridge_start, "source_end": bridge_end - 1},
                {"type": "source_b", "start": starts[2], "end": int(frames.shape[0]) - 1, "source_start": b_start, "source_end": int(b.shape[0]) - 1},
            ],
            "seams": [
                {"side": "a_to_bridge", "frame": starts[1], "blend_frames": video_overlaps[0], **left},
                {"side": "bridge_to_b", "frame": starts[2], "blend_frames": video_overlaps[1], **right},
            ],
            "audio": audio_info,
        }
        timeline = normalize_timeline(timeline_value, fps=fps, total_frames=int(frames.shape[0]))
        report = {"left": left, "right": right, "ranges": {"a_end": a_end, "bridge_start": bridge_start,
                  "bridge_end": bridge_end, "b_start": b_start}, "video_blend_frames": video_overlaps,
                  "visible_bridge_frames": bridge_end - bridge_start,
                  "output_frames": int(frames.shape[0])}
        return io.NodeOutput(frames, audio, timeline, json.dumps(report, ensure_ascii=False, indent=2))
