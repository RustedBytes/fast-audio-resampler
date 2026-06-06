# fast-audio-resampler

[![Crates.io Version](https://img.shields.io/crates/v/fast-audio-resampler)](https://crates.io/crates/fast-audio-resampler)

Fast streaming audio resampling for Rust, focused on Intel and AMD x86/x86_64 CPUs.

The crate exposes a reusable library by default. WAV CLI support is optional and gated behind the `cli` feature so library users do not pull `hound`.

## Usage

```rust
use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};

let config = ResamplerConfig {
    input_rate: 44_100,
    output_rate: 48_000,
    channels: 2,
    quality: Quality::Balanced,
    backend: Backend::Auto,
    max_input_frames_per_chunk: None,
};

let mut resampler = Resampler::<f32>::new(config)?;
let mut output = Vec::new();
resampler.process(&input_samples, &mut output)?;
resampler.finish(&mut output)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

CLI:

```bash
cargo run --features cli -- --in input.wav --out output.wav --rate 48000
```

## Design Choices

- Uses windowed-sinc polyphase FIR resampling for arbitrary sample-rate ratios.
- Provides dedicated phase handling for exact `8000 <-> 16000` conversions.
- Supports `f32` and `i16` sample paths.
- Uses runtime CPU feature detection instead of CPU vendor checks.
- Uses AVX2/FMA and AVX-512 intrinsics for `f32` where available.
- Uses a Q15 fixed-point `i16` path with AVX2 `_mm256_madd_epi16` on supported CPUs.
- Stores FIR coefficients in phase-major aligned storage for cache-friendly reads.
- Uses per-channel ring buffers for streaming history, avoiding steady-state buffer shifting.
- Keeps the public API stable while hiding backend and buffer details internally.

## Complexity

Let:

- `N` = input frames processed
- `M` = output frames produced
- `C` = channel count
- `T` = FIR tap count selected by `Quality`

Construction:

- Time: `O(P * T)`, where `P` is the number of polyphase coefficient phases.
- Space: `O(P * T + C * T)` for coefficient tables and per-channel history.

Processing:

- Time: `O(M * C * T)` scalar work.
- SIMD reduces the constant factor by processing multiple taps per instruction.
- Streaming append is `O(N * C)`.
- Ring-buffer history discard is `O(C)` and does not move sample data.

Output size:

- Approximately `ceil(N * output_rate / input_rate)` frames after `finish`.

## Features

- Default: library only, no WAV dependency.
- `cli`: enables the WAV command-line tool and the optional `hound` dependency.
