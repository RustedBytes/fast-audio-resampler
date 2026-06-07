use criterion::{Criterion, criterion_group, criterion_main};
use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};

fn config(input_rate: u32, output_rate: u32, channels: usize, backend: Backend) -> ResamplerConfig {
    ResamplerConfig {
        input_rate,
        output_rate,
        channels,
        quality: Quality::Balanced,
        backend,
        max_input_frames_per_chunk: Some(1024),
    }
}

fn f32_input(frames: usize, channels: usize) -> Vec<f32> {
    (0..frames)
        .flat_map(|frame| {
            (0..channels).map(move |channel| {
                (((frame as f32) * 0.01 + channel as f32 * 0.17).sin() * 0.25).clamp(-1.0, 1.0)
            })
        })
        .collect()
}

fn i16_input(frames: usize, channels: usize) -> Vec<i16> {
    f32_input(frames, channels)
        .into_iter()
        .map(|sample| (sample * i16::MAX as f32) as i16)
        .collect()
}

fn bench_resampler(c: &mut Criterion) {
    let ratios = [
        ("8k_to_16k_halfband", 8_000, 16_000),
        ("16k_to_8k_halfband", 16_000, 8_000),
        ("44k1_to_48k", 44_100, 48_000),
        ("48k_to_44k1", 48_000, 44_100),
    ];
    let backends = [("scalar", Backend::Scalar), ("auto", Backend::Auto)];

    for (ratio_name, input_rate, output_rate) in ratios {
        for channels in [1, 2] {
            let frames = input_rate as usize;
            let input_f32 = f32_input(frames, channels);
            let input_i16 = i16_input(frames, channels);

            for (backend_name, backend) in backends {
                c.bench_function(
                    &format!("f32/{ratio_name}/{channels}ch/{backend_name}"),
                    |b| {
                        b.iter(|| {
                            let mut resampler = Resampler::<f32>::new(config(
                                input_rate,
                                output_rate,
                                channels,
                                backend,
                            ))
                            .unwrap();
                            let mut output = Vec::new();
                            resampler.process(&input_f32, &mut output).unwrap();
                            resampler.finish(&mut output).unwrap();
                            output
                        })
                    },
                );

                c.bench_function(
                    &format!("i16/{ratio_name}/{channels}ch/{backend_name}"),
                    |b| {
                        b.iter(|| {
                            let mut resampler = Resampler::<i16>::new(config(
                                input_rate,
                                output_rate,
                                channels,
                                backend,
                            ))
                            .unwrap();
                            let mut output = Vec::new();
                            resampler.process(&input_i16, &mut output).unwrap();
                            resampler.finish(&mut output).unwrap();
                            output
                        })
                    },
                );
            }
        }
    }

    let streaming_input = f32_input(48_000, 2);
    c.bench_function("f32/48k_to_44k1/2ch/auto/streaming_64_frames", |b| {
        b.iter(|| {
            let mut resampler =
                Resampler::<f32>::new(config(48_000, 44_100, 2, Backend::Auto)).unwrap();
            let mut output = Vec::new();
            for chunk in streaming_input.chunks(64 * 2) {
                resampler.process(chunk, &mut output).unwrap();
            }
            resampler.finish(&mut output).unwrap();
            output
        })
    });

    for (ratio_name, input_rate, output_rate) in [
        ("8k_to_16k_halfband", 8_000, 16_000),
        ("16k_to_8k_halfband", 16_000, 8_000),
    ] {
        let input_f32 = f32_input(input_rate as usize, 2);
        c.bench_function(
            &format!("f32/{ratio_name}/2ch/auto/streaming_64_frames"),
            |b| {
                b.iter(|| {
                    let mut resampler =
                        Resampler::<f32>::new(config(input_rate, output_rate, 2, Backend::Auto))
                            .unwrap();
                    let mut output = Vec::new();
                    for chunk in input_f32.chunks(64 * 2) {
                        resampler.process(chunk, &mut output).unwrap();
                    }
                    resampler.finish(&mut output).unwrap();
                    output
                })
            },
        );

        let input_i16 = i16_input(input_rate as usize, 2);
        c.bench_function(
            &format!("i16/{ratio_name}/2ch/auto/streaming_64_frames"),
            |b| {
                b.iter(|| {
                    let mut resampler =
                        Resampler::<i16>::new(config(input_rate, output_rate, 2, Backend::Auto))
                            .unwrap();
                    let mut output = Vec::new();
                    for chunk in input_i16.chunks(64 * 2) {
                        resampler.process(chunk, &mut output).unwrap();
                    }
                    resampler.finish(&mut output).unwrap();
                    output
                })
            },
        );
    }
}

criterion_group!(benches, bench_resampler);
criterion_main!(benches);
