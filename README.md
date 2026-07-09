# fast-audio-resampler

[![PyPI - Version](https://img.shields.io/pypi/v/fast-audio-resampler)](https://pypi.org/project/fast-audio-resampler/)
[![Crates.io Version](https://img.shields.io/crates/v/fast-audio-resampler)](https://crates.io/crates/fast-audio-resampler)
[![CI](https://github.com/RustedBytes/fast-audio-resampler/actions/workflows/ci.yml/badge.svg)](https://github.com/RustedBytes/fast-audio-resampler/actions/workflows/ci.yml)

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

Python bindings are available through PyO3 for Python 3.9+:

```bash
maturin develop --features python-extension
```

```python
from fast_audio_resampler import F32Resampler

resampler = F32Resampler(44_100, 48_000, 2, quality="balanced", backend="auto")
output, stats = resampler.process(input_samples)
tail, tail_stats = resampler.finish()
output.extend(tail)
```

## Benchmarks

Criterion benchmark results from `cargo bench --bench resampler`. Each one-shot case processes one second of input audio. Times are medians; lower is better.

Exact-ratio special conversions use compact engines where available:

- Exact `8k <-> 16k` at `Quality::Fast`: polyphase IIR all-pass path.
- Exact `8k <-> 16k` at `Quality::Balanced` and `Quality::Best`: FIR half-band path.
- Exact `24k -> 8k`: sparse FIR third-band path.
- Other ratios: windowed-sinc polyphase FIR path.

### x86_64

`Quality::Fast` IIR results on x86_64:

| Format | Ratio | Channels | FIR Backend | Mode | Median |
| --- | --- | ---: | --- | --- | ---: |
| `f32` | 8k -> 16k | 1 | scalar | one-shot | 72.741 us |
| `i16` | 8k -> 16k | 1 | scalar | one-shot | 122.27 us |
| `f32` | 8k -> 16k | 1 | auto | one-shot | 75.437 us |
| `i16` | 8k -> 16k | 1 | auto | one-shot | 122.11 us |
| `f32` | 8k -> 16k | 2 | scalar | one-shot | 113.57 us |
| `i16` | 8k -> 16k | 2 | scalar | one-shot | 215.01 us |
| `f32` | 8k -> 16k | 2 | auto | one-shot | 114.21 us |
| `i16` | 8k -> 16k | 2 | auto | one-shot | 209.43 us |
| `f32` | 16k -> 8k | 1 | scalar | one-shot | 132.63 us |
| `i16` | 16k -> 8k | 1 | scalar | one-shot | 187.26 us |
| `f32` | 16k -> 8k | 1 | auto | one-shot | 133.90 us |
| `i16` | 16k -> 8k | 1 | auto | one-shot | 177.19 us |
| `f32` | 16k -> 8k | 2 | scalar | one-shot | 190.97 us |
| `i16` | 16k -> 8k | 2 | scalar | one-shot | 264.89 us |
| `f32` | 16k -> 8k | 2 | auto | one-shot | 192.58 us |
| `i16` | 16k -> 8k | 2 | auto | one-shot | 241.18 us |
| `f32` | 8k -> 16k | 2 | auto | streaming, 64-frame chunks | 183.40 us |
| `i16` | 8k -> 16k | 2 | auto | streaming, 64-frame chunks | 273.88 us |
| `f32` | 16k -> 8k | 2 | auto | streaming, 64-frame chunks | 288.59 us |
| `i16` | 16k -> 8k | 2 | auto | streaming, 64-frame chunks | 381.17 us |

`Quality::Balanced` FIR half-band results on x86_64:

| Format | Ratio | Channels | FIR Backend | Mode | Median |
| --- | --- | ---: | --- | --- | ---: |
| `f32` | 8k -> 16k | 1 | scalar | one-shot | 410.61 us |
| `i16` | 8k -> 16k | 1 | scalar | one-shot | 422.92 us |
| `f32` | 8k -> 16k | 1 | auto | one-shot | 407.42 us |
| `i16` | 8k -> 16k | 1 | auto | one-shot | 411.30 us |
| `f32` | 8k -> 16k | 2 | scalar | one-shot | 784.18 us |
| `i16` | 8k -> 16k | 2 | scalar | one-shot | 854.23 us |
| `f32` | 8k -> 16k | 2 | auto | one-shot | 804.98 us |
| `i16` | 8k -> 16k | 2 | auto | one-shot | 890.49 us |
| `f32` | 16k -> 8k | 1 | scalar | one-shot | 403.95 us |
| `i16` | 16k -> 8k | 1 | scalar | one-shot | 417.35 us |
| `f32` | 16k -> 8k | 1 | auto | one-shot | 427.03 us |
| `i16` | 16k -> 8k | 1 | auto | one-shot | 412.09 us |
| `f32` | 16k -> 8k | 2 | scalar | one-shot | 794.36 us |
| `i16` | 16k -> 8k | 2 | scalar | one-shot | 861.91 us |
| `f32` | 16k -> 8k | 2 | auto | one-shot | 834.62 us |
| `i16` | 16k -> 8k | 2 | auto | one-shot | 853.38 us |
| `f32` | 8k -> 16k | 2 | auto | streaming, 64-frame chunks | 1.0462 ms |
| `i16` | 8k -> 16k | 2 | auto | streaming, 64-frame chunks | 1.2621 ms |
| `f32` | 16k -> 8k | 2 | auto | streaming, 64-frame chunks | 1.1097 ms |
| `i16` | 16k -> 8k | 2 | auto | streaming, 64-frame chunks | 1.0886 ms |

General-ratio `Quality::Balanced` FIR results on x86_64:

| Format | Ratio | Channels | FIR Backend | Mode | Median |
| --- | --- | ---: | --- | --- | ---: |
| `f32` | 44.1k -> 48k | 1 | scalar | one-shot | 6.3880 ms |
| `i16` | 44.1k -> 48k | 1 | scalar | one-shot | 6.4852 ms |
| `f32` | 44.1k -> 48k | 1 | auto | one-shot | 5.8639 ms |
| `i16` | 44.1k -> 48k | 1 | auto | one-shot | 5.6396 ms |
| `f32` | 44.1k -> 48k | 2 | scalar | one-shot | 13.026 ms |
| `i16` | 44.1k -> 48k | 2 | scalar | one-shot | 14.342 ms |
| `f32` | 44.1k -> 48k | 2 | auto | one-shot | 14.945 ms |
| `i16` | 44.1k -> 48k | 2 | auto | one-shot | 10.145 ms |
| `f32` | 48k -> 44.1k | 1 | scalar | one-shot | 5.6738 ms |
| `i16` | 48k -> 44.1k | 1 | scalar | one-shot | 6.0230 ms |
| `f32` | 48k -> 44.1k | 1 | auto | one-shot | 5.2813 ms |
| `i16` | 48k -> 44.1k | 1 | auto | one-shot | 6.0168 ms |
| `f32` | 48k -> 44.1k | 2 | scalar | one-shot | 12.333 ms |
| `i16` | 48k -> 44.1k | 2 | scalar | one-shot | 10.919 ms |
| `f32` | 48k -> 44.1k | 2 | auto | one-shot | 10.069 ms |
| `i16` | 48k -> 44.1k | 2 | auto | one-shot | 9.4516 ms |
| `f32` | 48k -> 44.1k | 2 | auto | streaming, 64-frame chunks | 13.213 ms |

### AArch64 ARM

AArch64 ARM benchmarks should be regenerated with the current IIR/FIR split. The crate includes NEON FIR kernels and an IIR stereo NEON backend path, both selected where supported.

## Design Choices

- Uses windowed-sinc polyphase FIR resampling for arbitrary sample-rate ratios.
- Uses FIR half-band filtering for exact `8000 <-> 16000` at `Quality::Balanced` and `Quality::Best`.
- Uses sparse FIR third-band filtering for exact `24000 -> 8000`.
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
- Exact `24k -> 8k` construction uses a compact sparse third-band coefficient table instead of a full phase table.

Processing:

- FIR time: `O(M * C * T)` scalar work.
- IIR exact-ratio fast time: `O(M * C * S)`, where `S` is small and fixed.
- Exact `24k -> 8k` sparse FIR time: `O(M * C * T_sparse)`, where zero-valued third-band taps are omitted.
- FIR SIMD reduces the constant factor by processing multiple taps per instruction.
- IIR stereo backends can process left/right all-pass lanes together where the target architecture supports it.
- Streaming append is `O(N * C)`.
- Ring-buffer history discard is `O(C)` and does not move sample data.

Output size:

- Approximately `ceil(N * output_rate / input_rate)` frames after `finish`.

## Features

- Default: library only, no WAV dependency.
- `cli`: enables the WAV command-line tool and the optional `hound` dependency.
- `python`: enables testable PyO3 bindings with the Python 3.9 stable ABI.
- `python-extension`: enables `python` plus PyO3 extension-module linking for Python package builds.
