# Default Resampler Usage

This guide covers the standard library API for file conversion, batch jobs, and simple chunked audio processing.

For WebSocket or long-lived server sessions, see `docs/streaming.md`.

## Basic `f32` Resampling

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

Input and output samples are interleaved by channel. For stereo, the input layout is:

```text
L0, R0, L1, R1, L2, R2, ...
```

The input slice must contain complete frames. For example, stereo input must have an even number of samples.

## Basic `i16` Resampling

```rust
use fast_audio_resampler::{FirBackend, Quality, Resampler, ResamplerConfig};

let config = ResamplerConfig {
    input_rate: 48_000,
    output_rate: 16_000,
    channels: 1,
    quality: Quality::Balanced,
    backend: FirBackend::Auto,
    max_input_frames_per_chunk: None,
};

let mut resampler = Resampler::<i16>::new(config)?;
let mut output = Vec::new();

resampler.process(&input_pcm, &mut output)?;
resampler.finish(&mut output)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The `i16` path uses fixed-point Q15 coefficients and uses AVX2, AArch64 NEON, or RISC-V RVV integer multiply instructions on supported CPUs.

## Chunked Processing

You can feed input in chunks as long as you reuse the same resampler.

```rust
# use fast_audio_resampler::{FirBackend, Quality, Resampler, ResamplerConfig};
# let config = ResamplerConfig {
#     input_rate: 44_100,
#     output_rate: 48_000,
#     channels: 2,
#     quality: Quality::Balanced,
#     backend: FirBackend::Auto,
#     max_input_frames_per_chunk: None,
# };
# let mut resampler = Resampler::<f32>::new(config)?;
# let chunks: Vec<&[f32]> = Vec::new();
let mut output = Vec::new();

for chunk in chunks {
    resampler.process(chunk, &mut output)?;
}

resampler.finish(&mut output)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`finish` flushes the final filter tail. Call it once when the input stream is complete.

## Output Capacity

For simple use, passing a `Vec` is enough. The resampler reserves as needed.

If you want to reserve ahead of time:

```rust
# use fast_audio_resampler::{FirBackend, Quality, Resampler, ResamplerConfig};
# let config = ResamplerConfig {
#     input_rate: 44_100,
#     output_rate: 48_000,
#     channels: 2,
#     quality: Quality::Balanced,
#     backend: FirBackend::Auto,
#     max_input_frames_per_chunk: None,
# };
# let resampler = Resampler::<f32>::new(config)?;
# let input_frames = 44_100;
let mut output = Vec::with_capacity(resampler.required_output_capacity(input_frames));
# Ok::<(), Box<dyn std::error::Error>>(())
```

`required_output_capacity` returns a conservative sample count, not a frame count.

## Quality and FIR Backend

`Quality` controls the FIR tap count:

- `Fast`: fewer taps, lower CPU cost.
- `Balanced`: default tradeoff for general use.
- `Best`: more taps, higher CPU cost.

`FirBackend::Auto` is recommended. It selects the fastest supported FIR backend at runtime:

- scalar fallback on all targets
- AVX2/FMA on supported x86/x86_64 CPUs
- AVX-512 for `f32` where available
- AArch64 NEON on ARM CPUs
- RISC-V RVV on `riscv64` builds compiled with `-C target-feature=+v`

FIR backend dispatch is based on CPU features, not CPU vendor strings. RISC-V RVV selection is compile-time gated on stable Rust because portable runtime detection for the vector extension is not stable yet.

## CLI

The WAV CLI is optional and requires the `cli` feature.

```bash
cargo run --features cli -- --in input.wav --out output.wav --rate 48000
```

Supported WAV formats:

- `f32`
- `i16` PCM

