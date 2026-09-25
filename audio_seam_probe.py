"""Read-only audio seam diagnostics for the latest HR Endless replay."""

from __future__ import annotations

import json

import numpy as np
from comfy_api.latest import io

CORR_CREDIBLE = 0.6


def _mono(waveform):
    value = waveform.detach().cpu().numpy().astype(np.float64)
    while value.ndim > 2:
        value = value[0]
    if value.ndim == 2:
        value = value.mean(axis=0 if value.shape[0] <= value.shape[1] else 1)
    return value


def _normalized_correlation(window, reference):
    window = window - window.mean()
    reference = reference - reference.mean()
    window_norm = np.sqrt(np.square(window).sum())
    if window_norm < 1e-12:
        return None
    correlation = np.correlate(reference, window, mode="valid")
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(reference))))
    segment_energy = cumulative[len(window):] - cumulative[:-len(window)]
    return correlation / (window_norm * np.sqrt(np.maximum(segment_energy, 1e-12)))


def _tracked_peak(correlation, expected):
    best = float(correlation.max())
    indices = np.arange(1, len(correlation) - 1)
    peaks = indices[
        (correlation[indices] >= correlation[indices - 1])
        & (correlation[indices] >= correlation[indices + 1])
        & (correlation[indices] >= 0.9 * best)
    ]
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(correlation))])
    selected = int(peaks[np.argmin(np.abs(peaks - expected))])
    return selected, float(correlation[selected])


def _rms(value):
    return float(np.sqrt(np.mean(np.square(value)))) if len(value) else 0.0


def _floor_level(value, sample_rate):
    window = max(1, round(0.02 * sample_rate))
    count = len(value) // window
    if count < 3:
        return _rms(value)
    levels = np.sqrt(np.mean(np.square(value[:count * window].reshape(count, window)), axis=1))
    return float(np.percentile(levels, 10.0))


def _step_ratio(left, right):
    denominator = left + right
    return 0.0 if denominator <= 1e-12 else abs(left - right) / denominator


def analyze_audio_seam(previous, current, sample_rate, overlap_samples, window_ms=50.0, search_ms=40.0):
    if overlap_samples <= 0 or len(previous) < overlap_samples or len(current) <= overlap_samples:
        raise ValueError("Audio seam probe requires a non-empty overlap and post-overlap audio")
    window = max(1, round(window_ms / 1000.0 * sample_rate))
    hop = max(1, round(0.025 * sample_rate))
    search = max(1, round(search_ms / 1000.0 * sample_rate))
    tail_start = len(previous) - overlap_samples
    correlations = []
    lags = []
    offset = 0
    previous_lag = None
    while offset + window <= overlap_samples:
        nominal = tail_start + offset
        low = max(0, nominal - search)
        high = min(len(previous), nominal + window + search)
        reference = previous[low:high]
        if len(reference) <= window:
            break
        curve = _normalized_correlation(current[offset:offset + window], reference)
        if curve is None:
            offset += hop
            continue
        if previous_lag is None:
            selected = int(np.argmax(curve))
            correlation = float(curve[selected])
        else:
            selected, correlation = _tracked_peak(curve, nominal - previous_lag - low)
        lag = nominal - (low + selected)
        correlations.append(correlation)
        lags.append(lag / sample_rate * 1000.0)
        if correlation > CORR_CREDIBLE:
            previous_lag = lag
        offset += hop
    if not correlations:
        raise ValueError("Audio seam probe found no non-silent analysis windows")
    correlations = np.asarray(correlations)
    lags = np.asarray(lags)
    credible = correlations > CORR_CREDIBLE
    delivered = current[overlap_samples:]
    level_samples = min(round(0.5 * sample_rate), len(previous), len(delivered))
    if level_samples < round(0.05 * sample_rate):
        raise ValueError("Audio seam probe requires at least 50 ms around the delivered join")
    left = previous[-level_samples:]
    right = delivered[:level_samples]
    mean_correlation = float(correlations.mean())
    if mean_correlation < 0.3:
        reading = "not_continued"
    elif mean_correlation < CORR_CREDIBLE:
        reading = "imitation"
    elif credible.any() and abs(float(lags[credible].mean())) > 12.0:
        reading = "locked_but_offset"
    else:
        reading = "clean_continuation"
    return {
        "mean_correlation": mean_correlation,
        "credible_windows": int(credible.sum()),
        "window_count": int(len(correlations)),
        "mean_lag_ms": float(lags[credible].mean()) if credible.any() else None,
        "min_lag_ms": float(lags[credible].min()) if credible.any() else None,
        "max_lag_ms": float(lags[credible].max()) if credible.any() else None,
        "broadband_step": _step_ratio(_rms(left), _rms(right)),
        "floor_step": _step_ratio(_floor_level(left, sample_rate), _floor_level(right, sample_rate)),
        "reading": reading,
    }


class HREndlessAudioSeamProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessAudioSeamProbe",
            display_name="HR Endless Audio Seam Probe",
            category="model/sampling/custom",
            description="Decode and measure every audio join in the latest HR Endless replay without changing the render.",
            inputs=[
                io.Vae.Input("audio_vae"),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Float.Input("window_ms", default=50.0, min=5.0, max=500.0, step=1.0, advanced=True),
                io.Float.Input("search_ms", default=40.0, min=5.0, max=500.0, step=1.0, advanced=True),
            ],
            outputs=[io.String.Output(display_name="seam report")],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, audio_vae, fps=24.0, window_ms=50.0, search_ms=40.0):
        from .nodes import _replay_cache_root, _replay_load_tensor_file

        root = _replay_cache_root()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("HR Endless Audio Seam Probe found no latest replay")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunk_records = sorted(manifest.get("chunks", ()), key=lambda item: int(item["chunk"]))
        if len(chunk_records) < 2:
            raise ValueError("HR Endless Audio Seam Probe requires at least two completed chunks")
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        decoded = []
        frame_counts = []
        trims = []
        for record in chunk_records:
            number = int(record["chunk"])
            state = _replay_load_tensor_file(root / "chunks" / f"chunk_{number:04d}.pt")
            audio = state["sampled_audio"]
            waveform = audio_vae.decode(audio[:1])
            decoded.append(_mono(waveform))
            if "previous_frame_count" not in state:
                raise ValueError(f"Chunk {number} does not record its sampled frame count")
            frame_counts.append(int(state["previous_frame_count"]))
            metadata = json.loads((root / record["metadata_path"]).read_text(encoding="utf-8"))
            trims.append(int(metadata.get("output_trim_frames", 0)))
        results = []
        for index in range(1, len(decoded)):
            wanted = round(frame_counts[index - 1] / float(fps) * sample_rate)
            previous = decoded[index - 1]
            if len(previous) > wanted:
                previous = previous[:wanted]
            elif len(previous) < wanted:
                previous = np.concatenate((previous, np.zeros(wanted - len(previous))))
            overlap_samples = round(trims[index] / float(fps) * sample_rate)
            result = analyze_audio_seam(
                previous, decoded[index], sample_rate, overlap_samples,
                window_ms=window_ms, search_ms=search_ms,
            )
            result.update({"from_chunk": index, "to_chunk": index + 1, "overlap_frames": trims[index]})
            results.append(result)
        report = {
            "run_id": manifest.get("run_id"),
            "fps": float(fps),
            "sample_rate": sample_rate,
            "seams": results,
        }
        return io.NodeOutput(json.dumps(report, ensure_ascii=False, indent=2))
