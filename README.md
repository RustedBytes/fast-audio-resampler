# fast-audio-resampler

[![Crates.io Version](https://img.shields.io/crates/v/fast-audio-resampler)](https://crates.io/crates/fast-audio-resampler)

Fast streaming audio resampling for Rust, focused on x86/x86_64 and AArch64 ARM CPUs.

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

## Benchmarks

Criterion benchmark results from `cargo bench --bench resampler` on Windows 11, AMD Ryzen AI 7 350, Rust 1.96.0, `Balanced` quality. Each case processes one second of input audio. Times are medians; lower is better.

| Format | Ratio | Channels | Scalar | Auto | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `f32` | 8k -> 16k | 1 | 3.626 ms | 3.375 ms | 1.07x |
| `i16` | 8k -> 16k | 1 | 3.610 ms | 3.192 ms | 1.13x |
| `f32` | 8k -> 16k | 2 | 5.620 ms | 5.058 ms | 1.11x |
| `i16` | 8k -> 16k | 2 | 5.626 ms | 4.991 ms | 1.13x |
| `f32` | 16k -> 8k | 1 | 2.466 ms | 2.375 ms | 1.04x |
| `i16` | 16k -> 8k | 1 | 2.412 ms | 2.295 ms | 1.05x |
| `f32` | 16k -> 8k | 2 | 3.426 ms | 3.268 ms | 1.05x |
| `i16` | 16k -> 8k | 2 | 3.486 ms | 3.335 ms | 1.05x |
| `f32` | 44.1k -> 48k | 1 | 8.189 ms | 7.116 ms | 1.15x |
| `i16` | 44.1k -> 48k | 1 | 7.705 ms | 7.396 ms | 1.04x |
| `f32` | 44.1k -> 48k | 2 | 15.397 ms | 13.347 ms | 1.15x |
| `i16` | 44.1k -> 48k | 2 | 14.505 ms | 11.883 ms | 1.22x |
| `f32` | 48k -> 44.1k | 1 | 7.104 ms | 8.055 ms | 0.88x |
| `i16` | 48k -> 44.1k | 1 | 7.364 ms | 6.705 ms | 1.10x |
| `f32` | 48k -> 44.1k | 2 | 13.381 ms | 12.243 ms | 1.09x |
| `i16` | 48k -> 44.1k | 2 | 13.272 ms | 11.223 ms | 1.18x |

Streaming `f32` 48k -> 44.1k stereo with 64-frame chunks measured 12.476 ms with `Backend::Auto`.

## Design Choices

- Uses windowed-sinc polyphase FIR resampling for arbitrary sample-rate ratios.
- Provides dedicated phase handling for exact `8000 <-> 16000` conversions.
- Supports `f32` and `i16` sample paths.
- Uses runtime CPU feature detection instead of CPU vendor checks.
- Uses AVX2/FMA, AVX-512, and AArch64 NEON intrinsics for `f32` where available.
- Uses a Q15 fixed-point `i16` path with AVX2 `_mm256_madd_epi16` or AArch64 NEON widening multiply on supported CPUs.
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
