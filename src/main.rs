use std::env;
use std::path::PathBuf;

use fast_audio_resampler::{Error, FirBackend, Quality, Resampler, ResamplerConfig};

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    rate: u32,
    quality: Quality,
    backend: FirBackend,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let mut reader = hound::WavReader::open(&args.input)?;
    let spec = reader.spec();
    let config = ResamplerConfig {
        input_rate: spec.sample_rate,
        output_rate: args.rate,
        channels: spec.channels as usize,
        quality: args.quality,
        backend: args.backend,
        max_input_frames_per_chunk: None,
    };

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            let input: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>()?;
            let mut resampler = Resampler::<f32>::new(config)?;
            let mut output = Vec::new();
            resampler.process(&input, &mut output)?;
            resampler.finish(&mut output)?;
            let out_spec = hound::WavSpec {
                sample_rate: args.rate,
                ..spec
            };
            let mut writer = hound::WavWriter::create(&args.output, out_spec)?;
            for sample in output {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        (hound::SampleFormat::Int, 16) => {
            let input: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;
            let mut resampler = Resampler::<i16>::new(config)?;
            let mut output = Vec::new();
            resampler.process(&input, &mut output)?;
            resampler.finish(&mut output)?;
            let out_spec = hound::WavSpec {
                sample_rate: args.rate,
                ..spec
            };
            let mut writer = hound::WavWriter::create(&args.output, out_spec)?;
            for sample in output {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        _ => {
            return Err(Box::new(Error::Cli(format!(
                "unsupported WAV format: {:?} with {} bits per sample; supported formats are f32 and i16 PCM",
                spec.sample_format, spec.bits_per_sample
            ))));
        }
    }

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;
    let mut rate = None;
    let mut quality = Quality::Balanced;
    let mut backend = FirBackend::Auto;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--in" | "-i" => input = args.next().map(PathBuf::from),
            "--out" | "-o" => output = args.next().map(PathBuf::from),
            "--rate" | "-r" => rate = Some(args.next().ok_or("--rate requires a value")?.parse()?),
            "--quality" | "-q" => {
                quality = parse_quality(&args.next().ok_or("--quality requires a value")?)?
            }
            "--backend" | "-b" => {
                backend = parse_backend(&args.next().ok_or("--backend requires a value")?)?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        input: input.ok_or("--in is required")?,
        output: output.ok_or("--out is required")?,
        rate: rate.ok_or("--rate is required")?,
        quality,
        backend,
    })
}

fn parse_quality(value: &str) -> Result<Quality, Box<dyn std::error::Error>> {
    match value {
        "fast" => Ok(Quality::Fast),
        "balanced" => Ok(Quality::Balanced),
        "best" => Ok(Quality::Best),
        _ => Err(format!("unknown quality: {value}").into()),
    }
}

fn parse_backend(value: &str) -> Result<FirBackend, Box<dyn std::error::Error>> {
    match value {
        "auto" => Ok(FirBackend::Auto),
        "scalar" => Ok(FirBackend::Scalar),
        "avx2" => Ok(FirBackend::Avx2),
        "avx512" => Ok(FirBackend::Avx512),
        "neon" => Ok(FirBackend::Neon),
        "rvv" => Ok(FirBackend::Rvv),
        _ => Err(format!("unknown backend: {value}").into()),
    }
}

fn print_help() {
    println!(
        "fast-audio-resampler --in input.wav --out output.wav --rate 48000 [--quality balanced] [--backend auto]\n\
         \n\
         quality: fast | balanced | best\n\
         backend: auto | scalar | avx2 | avx512 | neon | rvv"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_accepts_neon() {
        assert_eq!(parse_backend("neon").unwrap(), FirBackend::Neon);
    }

    #[test]
    fn parse_backend_accepts_rvv() {
        assert_eq!(parse_backend("rvv").unwrap(), FirBackend::Rvv);
    }
}
