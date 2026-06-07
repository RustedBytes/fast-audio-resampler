# fast-audio-resampler

[![Crates.io Version](https://img.shields.io/crates/v/fast-audio-resampler)](https://crates.io/crates/fast-audio-resampler)

Fast streaming audio resampling for Rust, focused on x86/x86_64, AArch64 ARM, and RISC-V CPUs.

The crate exposes a reusable library by default. WAV CLI support is optional and gated behind the `cli` feature so library users do not pull `hound`.

## Usage

```rust
use fast_audio_resampler::{FirBackend, Quality, Resampler, ResamplerConfig};

let config = ResamplerConfig {
    input_rate: 44_100,
    output_rate: 48_000,
    channels: 2,
    quality: Quality::Balanced,
    backend: FirBackend::Auto,
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

Criterion benchmark results from `cargo bench --bench resampler` with `Balanced` quality. Each case processes one second of input audio. Times are medians; lower is better.

### x86_64

| Format | Ratio | Channels | FIR Backend | Median |
| --- | --- | ---: | --- | ---: |
| `f32` | 8k -> 16k | 1 | scalar | 3.360 ms |
| `i16` | 8k -> 16k | 1 | scalar | 3.271 ms |
| `f32` | 8k -> 16k | 1 | auto | 3.001 ms |
| `i16` | 8k -> 16k | 1 | auto | 3.033 ms |
| `f32` | 48k -> 44.1k | 2 | auto | 11.199 ms |
| `i16` | 48k -> 44.1k | 2 | auto | 10.743 ms |

Streaming `f32` 48k -> 44.1k stereo with 64-frame chunks measured 10.807 ms with `FirBackend::Auto`.

### AArch64 ARM

| Format | Ratio | Channels | FIR Backend | Median |
| --- | --- | ---: | --- | ---: |
| `f32` | 8k -> 16k | 1 | scalar | 2.762 ms |
| `i16` | 8k -> 16k | 1 | scalar | 2.730 ms |
| `f32` | 8k -> 16k | 1 | auto | 2.658 ms |
| `i16` | 8k -> 16k | 1 | auto | 2.685 ms |
| `i16` | 48k -> 44.1k | 2 | auto | 9.491 ms |

Streaming `f32` 48k -> 44.1k stereo with 64-frame chunks measured 10.360 ms with `FirBackend::Auto`.

## Design Choices

- Uses windowed-sinc polyphase FIR resampling for arbitrary sample-rate ratios.
- Provides dedicated phase handling for exact `8000 <-> 16000` conversions.
- Supports `f32` and `i16` sample paths.
- Uses runtime CPU feature detection instead of CPU vendor checks.
- Uses AVX2/FMA, AVX-512, and AArch64 NEON intrinsics for `f32` where available.
- Uses RISC-V RVV 1.0 kernels on `riscv64` builds compiled with `-C target-feature=+v`.
- Uses a Q15 fixed-point `i16` path with AVX2 `_mm256_madd_epi16`, AArch64 NEON widening multiply, or RISC-V RVV widening multiply-accumulate on supported CPUs.
- Keeps RISC-V RVV selection compile-time gated because stable Rust does not yet provide portable runtime detection for the vector extension.
- Stores FIR coefficients in phase-major aligned storage for cache-friendly reads.
- Uses per-channel ring buffers for streaming history, avoiding steady-state buffer shifting.
- Keeps the public API stable while hiding FIR backend and buffer details internally.

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
