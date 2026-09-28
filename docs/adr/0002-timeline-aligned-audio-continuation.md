# ADR 0002: Timeline-aligned H3 audio continuation

- Status: Accepted, amended after runtime validation
- Date: 2026-09-24

## Context

HR Endless Sampler generates MiniMax H3 video and audio jointly in physical chunks. Its established continuation path supplies a bounded tail of the preceding chunk as a synchronized `video_audio` reference and removes the repeated prefix when assembling chunk outputs.

This reference preserves prior audiovisual content, but H3 treats reference audio as material to imitate rather than samples occupying the target timeline immediately before the join. Long speech crossing short chunks can consequently restart, change delivery, or break at the boundary.

H3 Motion Context demonstrates a different conditioning contract: slice real audio steps directly from the preceding joint AV latent and place that window on the target timeline so it ends exactly at the carried video boundary. Correct placement requires fractional and usually negative `resolved_frame_index` values because H3 audio runs at 40 Hz while video runs at 24 fps. Legal H3 lengths also produce an audio-grid overhang of `-1/3`, `0`, or `+1/3` step.

Initial integration retained audio in the synchronized AV reference as a second condition. Runtime validation with long dialogue showed nearly every chunk restarting speech before jumping to its scheduled fragment. The duplicate reference-audio and timeline-audio conditions therefore conflict in practice.

## Decision

Add timeline-aligned prior-audio conditioning to the sampler's existing internal chunk loop.

1. Slice the audio tail directly from the preceding sampled H3 AV latent. Do not decode to PCM or re-encode it.
2. Use a fixed one-second window, 24 video-frame equivalents at 24 fps, independently of the 22-frame video continuation reference. If the preceding chunk is shorter, use all available audio steps.
3. Place the audio as a `minimax_keyframes` entry whose window ends at the end of the carried video prefix. Preserve fractional and negative `resolved_frame_index` values.
4. Account for the preceding latent's signed audio-grid overhang and snap the window end to the target audio grid before computing its start index.
5. Keep the existing continuation video reference but make it visual-only whenever timeline audio is active. Do not also expose the preceding audio as `<Audio N>` or a `video_audio` reference: the timeline keyframe is the single prior-audio condition.
6. Keep the existing latent assembly rule: preserve prior completed output and trim the repeated prefix from the newly sampled chunk. Do not add PCM crossfades or post-generation waveform welding.
7. Validate the required H3 `PackedLayout` behavior before using timeline audio: fractional/negative audio anchors must remain literal, reference blocks must not shift target-relative anchors, and stereo audio row counts must match the latent.
8. Cache a successful layout-contract check per process. Fail clearly if a future ComfyUI change violates it rather than silently rendering a displaced join.
9. Keep node inputs, output types, Prompt Skill, replay format, video conditioning, and ComfyUI core unchanged.

## Consequences

- Speech, music, and ambience receive real prior samples on the new chunk's timeline rather than only a sound-alike reference.
- Conditioning row count and VRAM use increase by approximately one second of stereo audio latent rows per continuation chunk.
- The implementation depends on documented arithmetic behavior of ComfyUI's H3 `PackedLayout`; the runtime contract check makes that dependency explicit.
- Replay fingerprints need no new user-facing field because the audio window is an implementation invariant. Replays created before this decision must not be used to judge the new behavior; validation renders should start from Chunk 1.
- Runtime evidence rejected simultaneous reference audio and timeline audio because dialogue restarted at chunk boundaries. The continuation reference now carries video only.
- Increasing `chunk_frames` remains useful because it reduces the number of joins, but it is not the continuity mechanism.

## Alternatives rejected

- **Reference audio only:** already used by the original sampler, but it asks H3 to imitate sound rather than continue samples on the target timeline.
- **PCM crossfade or waveform replacement:** operates after joint AV generation, cannot preserve H3 latent causality, and risks hiding rather than fixing timing errors.
- **Decode and re-encode the previous audio tail:** introduces avoidable loss and extra VAE work at every boundary.
- **Expose another node input immediately:** unnecessary until runtime evidence shows that the one-second invariant needs user control.
