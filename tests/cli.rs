#![cfg(feature = "cli")]

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn cli_resamples_i16_wav_with_hound() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("fast-audio-resampler-{stamp}"));
    std::fs::create_dir_all(&dir).unwrap();
    let input_path = dir.join("input.wav");
    let output_path = dir.join("output.wav");

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&input_path, spec).unwrap();
    for i in 0..80 {
        let sample = (((i as f32) * 0.2).sin() * i16::MAX as f32 * 0.5) as i16;
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_fast-audio-resampler"))
        .args([
            "--in",
            input_path.to_str().unwrap(),
            "--out",
            output_path.to_str().unwrap(),
            "--rate",
            "16000",
            "--backend",
            "scalar",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let mut reader = hound::WavReader::open(&output_path).unwrap();
    let out_spec = reader.spec();
    assert_eq!(out_spec.channels, 1);
    assert_eq!(out_spec.sample_rate, 16_000);
    assert_eq!(out_spec.bits_per_sample, 16);
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>().unwrap();
    assert_eq!(samples.len(), 160);
    assert!(samples.iter().any(|&sample| sample != 0));

    let _ = std::fs::remove_dir_all(&dir);
}
