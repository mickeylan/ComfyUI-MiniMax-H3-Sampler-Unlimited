"""Decode and assemble replay audio with measured, bounded seam alignment."""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
from comfy_api.latest import io

try:
    from .audio_seam_probe import analyze_audio_seam
except ImportError:  # Direct test execution.
    from audio_seam_probe import analyze_audio_seam


HREndlessTimeline = io.Custom("HRENDLESS_TIMELINE")
ALIGN_CORRELATION = 0.75
ALIGN_MAX_LAG_MS = 12.0
GAIN_LIMIT_DB = 3.0
GAIN_RELEASE_MS = 150.0


def _decoded_channels(audio_vae, latent):
    decoded = audio_vae.decode(latent[:1]).detach().cpu().float()
    if decoded.ndim == 3:
        decoded = decoded.movedim(-1, 1)
    elif decoded.ndim != 2:
        raise ValueError(f"HR Endless Audio Assemble expected decoded audio [B,L,C] or [C,L], got {tuple(decoded.shape)}")
    if decoded.ndim == 3:
        decoded = decoded[0]
    return decoded.numpy().astype(np.float64, copy=False)


def _fit_length(audio, length):
    if audio.shape[-1] >= length:
        return audio[..., :length]
    return np.pad(audio, ((0, 0), (0, length - audio.shape[-1])))


def _rms_gain_envelope(previous, current, sample_rate, credible):
    if not credible:
        return 1.0, None
    measure = min(round(0.1 * sample_rate), previous.shape[-1], current.shape[-1])
    if measure < 1:
        return 1.0, None
    left = float(np.sqrt(np.mean(np.square(previous[..., -measure:]))))
    right = float(np.sqrt(np.mean(np.square(current[..., :measure]))))
    if left < 1e-4 or right < 1e-4:
        return 1.0, None
    limit = 10.0 ** (GAIN_LIMIT_DB / 20.0)
    gain = float(np.clip(left / right, 1.0 / limit, limit))
    release = min(round(GAIN_RELEASE_MS / 1000.0 * sample_rate), current.shape[-1])
    envelope = np.ones(current.shape[-1], dtype=np.float64)
    if release:
        envelope[:release] = np.linspace(gain, 1.0, release, endpoint=True)
    return gain, envelope


def assemble_audio_chunks(decoded, frame_counts, trim_frames, fps, sample_rate):
    if len(decoded) != len(frame_counts) or len(decoded) != len(trim_frames):
        raise ValueError("HR Endless Audio Assemble received inconsistent chunk metadata")
    first_length = round(frame_counts[0] / fps * sample_rate)
    assembled = _fit_length(decoded[0], first_length).copy()
    seams = []
    delivered_frames = int(frame_counts[0])
    for index in range(1, len(decoded)):
        current = decoded[index]
        overlap = round(trim_frames[index] / fps * sample_rate)
        previous_length = round(frame_counts[index - 1] / fps * sample_rate)
        previous_mono = _fit_length(decoded[index - 1], previous_length).mean(axis=0)
        current_mono = current.mean(axis=0)
        result = analyze_audio_seam(previous_mono, current_mono, sample_rate, overlap)
        credible = (
            result["mean_correlation"] >= ALIGN_CORRELATION
            and result["mean_lag_ms"] is not None
            and abs(result["mean_lag_ms"]) <= ALIGN_MAX_LAG_MS
        )
        lag_samples = round(result["mean_lag_ms"] / 1000.0 * sample_rate) if credible else 0
        cut = max(0, min(current.shape[-1], overlap + lag_samples))
        fade = round((0.03 if credible else 0.01) * sample_rate)
        fade = min(fade, assembled.shape[-1], cut, current.shape[-1] - cut)
        new_frames = int(frame_counts[index]) - int(trim_frames[index])
        target_total = round((delivered_frames + new_frames) / fps * sample_rate)
        append_length = target_total - assembled.shape[-1]
        start = max(0, cut - fade)
        segment = _fit_length(current[..., start:], append_length + fade)
        gain, envelope = _rms_gain_envelope(assembled, segment[..., fade:], sample_rate, credible)
        if envelope is not None:
            segment[..., fade:fade + len(envelope)] *= envelope
            if fade:
                segment[..., :fade] *= gain
        if fade:
            phase = np.linspace(0.0, np.pi / 2.0, fade, endpoint=True)
            assembled[..., -fade:] = assembled[..., -fade:] * np.cos(phase) + segment[..., :fade] * np.sin(phase)
        assembled = np.concatenate((assembled, segment[..., fade:fade + append_length]), axis=-1)
        delivered_frames += new_frames
        seams.append({
            "from_chunk": index,
            "to_chunk": index + 1,
            "correlation": result["mean_correlation"],
            "lag_ms": result["mean_lag_ms"],
            "cut_samples": cut,
            "fade_samples": fade,
            "gain_match": gain,
            "gain_match_db": 20.0 * np.log10(gain),
            "gain_release_samples": 0 if envelope is None else int(np.count_nonzero(envelope != 1.0)),
            "aligned": credible,
            "alignment_reason": "high_correlation_bounded_lag" if credible else "unaligned_short_fade",
        })
    return assembled, seams


class HREndlessAudioSeamAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessAudioSeamAssemble",
            display_name="HR Endless Audio Seam Assemble",
            category="model/sampling/custom",
            description="Decode replay chunks, align correlated overlaps, and apply bounded equal-power seam fades.",
            inputs=[
                io.Vae.Input("audio_vae"),
                HREndlessTimeline.Input("timeline"),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, audio_vae, timeline, fps=24.0):
        from .nodes import _replay_cache_root, _replay_load_tensor_file

        root = _replay_cache_root()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        records = sorted(manifest.get("chunks", ()), key=lambda item: int(item["chunk"]))
        if len(records) < 2:
            raise ValueError("HR Endless Audio Seam Assemble requires at least two completed chunks")
        sample_rate = int(getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000)))
        decoded = []
        frame_counts = []
        trims = []
        for record in records:
            number = int(record["chunk"])
            state = _replay_load_tensor_file(root / "chunks" / f"chunk_{number:04d}.pt")
            decoded.append(_decoded_channels(audio_vae, state["sampled_audio"]))
            frame_counts.append(int(state["previous_frame_count"]))
            metadata = json.loads((root / record["metadata_path"]).read_text(encoding="utf-8"))
            trims.append(int(metadata.get("output_trim_frames", 0)))
        waveform, seams = assemble_audio_chunks(decoded, frame_counts, trims, float(fps), sample_rate)
        audio = torch.from_numpy(waveform.astype(np.float32, copy=False)).unsqueeze(0)
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        report_path = root / "audio_assembly_report.json"
        report_path.write_text(json.dumps({"run_id": manifest.get("run_id"), "seams": seams}, indent=2), encoding="utf-8")
        logging.info("HR Endless Audio Seam Assemble: wrote %d aligned seam decisions to %s", len(seams), report_path)
        return io.NodeOutput({"waveform": audio, "sample_rate": sample_rate})
