# Streaming and WebSocket Servers

Use one `Resampler<T>` per audio stream. In a WebSocket server, that usually means one resampler per connected client, room participant, or upstream audio track.

Do not share one resampler across clients: the resampler owns filter history, IIR state when selected, ring-buffer state, and sample-rate phase position.

## Session State

```rust
use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};

struct ClientSession {
    resampler: Resampler<i16>,
    output: Vec<i16>,
}

impl ClientSession {
    fn new() -> fast_audio_resampler::Result<Self> {
        let config = ResamplerConfig {
            input_rate: 48_000,
            output_rate: 16_000,
            channels: 1,
            quality: Quality::Balanced,
            backend: Backend::Auto,
            max_input_frames_per_chunk: Some(960),
        };

        let resampler = Resampler::<i16>::new(config)?;
        let output = Vec::with_capacity(resampler.required_output_capacity(960));

        Ok(Self { resampler, output })
    }
}
```

`max_input_frames_per_chunk` is optional. Set it when your WebSocket packet size is predictable so the internal ring buffers can be sized up front.

## Processing Messages

```rust
# use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};
# let config = ResamplerConfig {
#     input_rate: 48_000,
#     output_rate: 16_000,
#     channels: 1,
#     quality: Quality::Balanced,
#     backend: Backend::Auto,
#     max_input_frames_per_chunk: Some(960),
# };
# let mut resampler = Resampler::<i16>::new(config)?;
# let mut output = Vec::with_capacity(resampler.required_output_capacity(960));
# let incoming_pcm: &[i16] = &[0; 960];
output.clear();
resampler.process(incoming_pcm, &mut output)?;

// Send `output` as the resampled audio payload.
# Ok::<(), Box<dyn std::error::Error>>(())
```

The input slice must contain complete interleaved frames. For stereo, the sample count must be divisible by `2`; for mono, every sample is one frame.

## Bounded Output Buffers

If your server uses fixed-size buffers, ask the resampler for a conservative capacity and use `process_into_slice`.

```rust
# use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};
# let config = ResamplerConfig {
#     input_rate: 48_000,
#     output_rate: 16_000,
#     channels: 1,
#     quality: Quality::Balanced,
#     backend: Backend::Auto,
#     max_input_frames_per_chunk: Some(960),
# };
# let mut resampler = Resampler::<i16>::new(config)?;
# let incoming_pcm: &[i16] = &[0; 960];
let mut output = vec![0i16; resampler.required_output_capacity(960)];
let stats = resampler.process_into_slice(incoming_pcm, &mut output)?;
let written_samples = stats.output_frames * config.channels;
let resampled_payload = &output[..written_samples];
# let _ = resampled_payload;
# Ok::<(), Box<dyn std::error::Error>>(())
```

If the output slice is too small, the call returns `Error::OutputTooSmall` and does not write partial output.

## Stream Boundaries

Call `flush` when an audio stream ends and you want the final filter tail. For exact `8_000 <-> 16_000` conversions at `Quality::Fast`, this also drains any pending IIR downsampling pair.

```rust
# use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};
# let config = ResamplerConfig {
#     input_rate: 48_000,
#     output_rate: 16_000,
#     channels: 1,
#     quality: Quality::Balanced,
#     backend: Backend::Auto,
#     max_input_frames_per_chunk: Some(960),
# };
# let mut resampler = Resampler::<i16>::new(config)?;
let mut tail = Vec::new();
resampler.flush(&mut tail)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Call `reset` when a connection starts a new independent audio stream or after a discontinuity.

```rust
# use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};
# let config = ResamplerConfig {
#     input_rate: 48_000,
#     output_rate: 16_000,
#     channels: 1,
#     quality: Quality::Balanced,
#     backend: Backend::Auto,
#     max_input_frames_per_chunk: Some(960),
# };
# let mut resampler = Resampler::<i16>::new(config)?;
resampler.reset();
```

## Complexity

For each WebSocket audio packet:

- Append cost: `O(N * C)`
- Resampling cost: `O(M * C * T)` for FIR paths; `O(M * C * S)` for the exact `8_000 <-> 16_000` IIR fast path
- Ring-buffer discard: `O(C)`

Where `N` is input frames, `M` is output frames, `C` is channel count, `T` is FIR taps for the selected quality, and `S` is the small fixed IIR all-pass stage count.

