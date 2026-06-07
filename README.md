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

Criterion benchmark results from `cargo bench --bench resampler`. Each one-shot case processes one second of input audio. Times are medians; lower is better.

Exact `8k <-> 16k` conversions use two different engines depending on quality:

- `Quality::Fast`: polyphase IIR all-pass path.
- `Quality::Balanced` and `Quality::Best`: FIR half-band path.
- Other ratios: windowed-sinc polyphase FIR path.

### x86_64

Focused `Quality::Fast` IIR results on x86_64:

| Format | Ratio | Channels | FIR Backend | Mode | Median |
| --- | --- | ---: | --- | --- | ---: |
| `f32` | 8k -> 16k | 1 | scalar | one-shot | 68 us |
| `i16` | 8k -> 16k | 1 | scalar | one-shot | 169 us |
| `f32` | 8k -> 16k | 2 | scalar | one-shot | 111 us |
| `i16` | 8k -> 16k | 2 | scalar | one-shot | 207 us |
| `f32` | 8k -> 16k | 2 | auto | streaming, 64-frame chunks | 129 us |
| `i16` | 8k -> 16k | 2 | auto | streaming, 64-frame chunks | 204 us |

General-ratio FIR benchmarks are still emitted by the same Criterion suite under labels such as `f32/44k1_to_48k/2ch/auto`.

### AArch64 ARM

AArch64 ARM benchmarks should be regenerated with the current IIR/FIR split. The crate includes NEON FIR kernels and an IIR stereo NEON backend path, both selected where supported.

## Design Choices

- Uses windowed-sinc polyphase FIR resampling for arbitrary sample-rate ratios.
- Uses FIR half-band filtering for exact `8000 <-> 16000` at `Quality::Balanced` and `Quality::Best`.
- Uses a polyphase IIR all-pass path for exact `8000 <-> 16000` at `Quality::Fast`.
- Supports `f32` and `i16` sample paths.
- Uses runtime CPU feature detection instead of CPU vendor checks.
- Uses AVX2/FMA, AVX-512, and AArch64 NEON intrinsics for FIR `f32` where available.
- Uses RISC-V RVV 1.0 FIR kernels on `riscv64` builds compiled with `-C target-feature=+v`.
- Uses a Q15 fixed-point `i16` path with AVX2 `_mm256_madd_epi16`, AArch64 NEON widening multiply, or RISC-V RVV widening multiply-accumulate on supported CPUs.
- Keeps FIR backend naming explicit with `FirBackend` and `SelectedFirBackend`; deprecated `Backend` aliases remain for compatibility.
- Keeps IIR backend selection separate from FIR backend selection. IIR currently has scalar, x86 SSE2, AArch64 NEON, and RVV-gated stereo all-pass kernels, with conservative auto-selection based on benchmark behavior.
- Keeps RISC-V RVV selection compile-time gated because stable Rust does not yet provide portable runtime detection for the vector extension.
- Stores FIR coefficients in phase-major aligned storage for cache-friendly reads.
- Uses per-channel ring buffers and IIR state for streaming history, avoiding steady-state buffer shifting.
- Keeps the public API stable while hiding FIR backend and buffer details internally.

## Complexity

Let:

- `N` = input frames processed
- `M` = output frames produced
- `C` = channel count
- `T` = FIR tap count selected by `Quality`
- `S` = fixed IIR all-pass stage count for the exact-ratio fast path

Construction:

- Time: `O(P * T)`, where `P` is the number of polyphase coefficient phases.
- Space: `O(P * T + C * T)` for FIR coefficient tables and per-channel history.
- Exact `8k <-> 16k` `Quality::Fast` IIR construction uses fixed coefficient/state storage per channel instead of a phase table.

Processing:

- FIR time: `O(M * C * T)` scalar work.
- IIR exact-ratio fast time: `O(M * C * S)`, where `S` is small and fixed.
- FIR SIMD reduces the constant factor by processing multiple taps per instruction.
- IIR stereo backends can process left/right all-pass lanes together where the target architecture supports it.
- Streaming append is `O(N * C)`.
- Ring-buffer history discard is `O(C)` and does not move sample data.

Output size:

- Approximately `ceil(N * output_rate / input_rate)` frames after `finish`.

## Features

- Default: library only, no WAV dependency.
- `cli`: enables the WAV command-line tool and the optional `hound` dependency.
