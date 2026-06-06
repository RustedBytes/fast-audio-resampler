use criterion::{Criterion, criterion_group, criterion_main};
use fast_audio_resampler::{Backend, Quality, Resampler, ResamplerConfig};

fn bench_resampler(c: &mut Criterion) {
    let input: Vec<f32> = (0..48_000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.25)
        .collect();

    for (name, input_rate, output_rate) in [
        ("8k_to_16k", 8_000, 16_000),
        ("16k_to_8k", 16_000, 8_000),
        ("44k1_to_48k", 44_100, 48_000),
    ] {
        c.bench_function(name, |b| {
            b.iter(|| {
                let config = ResamplerConfig {
                    input_rate,
                    output_rate,
                    channels: 1,
                    quality: Quality::Balanced,
                    backend: Backend::Auto,
                };
                let mut resampler = Resampler::<f32>::new(config).unwrap();
                let mut output = Vec::new();
                resampler.process(&input, &mut output).unwrap();
                resampler.finish(&mut output).unwrap();
                output
            })
        });
    }
}

criterion_group!(benches, bench_resampler);
criterion_main!(benches);
