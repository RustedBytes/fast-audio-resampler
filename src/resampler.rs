use std::marker::PhantomData;

use crate::backend::{self, SelectedBackend};
use crate::error::frame_alignment_error;
use crate::filter::FilterBank;
use crate::{ResamplerConfig, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecialRatio {
    General,
    Up2,
    Down2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessStats {
    pub input_frames: usize,
    pub output_frames: usize,
    pub backend: SelectedBackend,
}

#[derive(Debug, Clone)]
struct Core {
    config: ResamplerConfig,
    backend: SelectedBackend,
    filter: FilterBank,
    channels: Vec<Vec<f32>>,
    start_frame: i64,
    total_input_frames: i64,
    next_source_pos: f64,
    output_frames_emitted: i64,
    step: f64,
    special_ratio: SpecialRatio,
    scratch: Vec<f32>,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct Resampler<T> {
    core: Core,
    _sample: PhantomData<T>,
}

impl<T> Resampler<T> {
    #[inline]
    pub fn selected_backend(&self) -> SelectedBackend {
        self.core.backend
    }
}

impl Resampler<f32> {
    pub fn new(config: ResamplerConfig) -> Result<Self> {
        Ok(Self {
            core: Core::new(config)?,
            _sample: PhantomData,
        })
    }

    #[inline]
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<ProcessStats> {
        self.core.process_f32(input, output)
    }

    #[inline]
    pub fn finish(&mut self, output: &mut Vec<f32>) -> Result<ProcessStats> {
        self.core.finish_f32(output)
    }
}

impl Resampler<i16> {
    pub fn new(config: ResamplerConfig) -> Result<Self> {
        Ok(Self {
            core: Core::new(config)?,
            _sample: PhantomData,
        })
    }

    pub fn process(&mut self, input: &[i16], output: &mut Vec<i16>) -> Result<ProcessStats> {
        let mut converted = Vec::with_capacity(input.len());
        converted.extend(input.iter().map(|&sample| sample as f32 / 32768.0));
        let mut tmp = Vec::new();
        let stats = self.core.process_f32(&converted, &mut tmp)?;
        output.reserve(tmp.len());
        output.extend(tmp.into_iter().map(f32_to_i16));
        Ok(stats)
    }

    pub fn finish(&mut self, output: &mut Vec<i16>) -> Result<ProcessStats> {
        let mut tmp = Vec::new();
        let stats = self.core.finish_f32(&mut tmp)?;
        output.reserve(tmp.len());
        output.extend(tmp.into_iter().map(f32_to_i16));
        Ok(stats)
    }
}

impl Core {
    fn new(config: ResamplerConfig) -> Result<Self> {
        config.validate()?;
        let backend = config.backend.select()?;
        let filter = FilterBank::new(config.input_rate, config.output_rate, config.quality);
        let taps = filter.taps();
        let channels = (0..config.channels)
            .map(|_| Vec::with_capacity(taps * 4))
            .collect();
        let step = config.input_rate as f64 / config.output_rate as f64;
        let special_ratio = match (config.input_rate, config.output_rate) {
            (8_000, 16_000) => SpecialRatio::Up2,
            (16_000, 8_000) => SpecialRatio::Down2,
            _ => SpecialRatio::General,
        };
        Ok(Self {
            config,
            backend,
            filter,
            channels,
            start_frame: 0,
            total_input_frames: 0,
            next_source_pos: 0.0,
            output_frames_emitted: 0,
            step,
            special_ratio,
            scratch: vec![0.0; taps],
            finished: false,
        })
    }

    fn process_f32(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<ProcessStats> {
        if self.finished {
            return Ok(ProcessStats {
                input_frames: 0,
                output_frames: 0,
                backend: self.backend,
            });
        }
        let input_frames = self.append_input(input)?;
        let before = output.len();
        self.render_available(output, false);
        self.discard_consumed();
        Ok(ProcessStats {
            input_frames,
            output_frames: (output.len() - before) / self.config.channels,
            backend: self.backend,
        })
    }

    fn finish_f32(&mut self, output: &mut Vec<f32>) -> Result<ProcessStats> {
        let before = output.len();
        self.render_available(output, true);
        self.finished = true;
        Ok(ProcessStats {
            input_frames: 0,
            output_frames: (output.len() - before) / self.config.channels,
            backend: self.backend,
        })
    }

    fn append_input(&mut self, input: &[f32]) -> Result<usize> {
        if !input.len().is_multiple_of(self.config.channels) {
            return Err(frame_alignment_error(input.len(), self.config.channels));
        }
        let frames = input.len() / self.config.channels;
        for frame in input.chunks_exact(self.config.channels) {
            for (channel, &sample) in frame.iter().enumerate() {
                self.channels[channel].push(sample);
            }
        }
        self.total_input_frames += frames as i64;
        Ok(frames)
    }

    fn render_available(&mut self, output: &mut Vec<f32>, flush: bool) {
        let out_frames = if flush {
            expected_output_frames(
                self.total_input_frames,
                self.config.input_rate,
                self.config.output_rate,
            )
        } else {
            i64::MAX
        };
        output.reserve(self.frames_ready_to_render(flush, out_frames) * self.config.channels);
        loop {
            if flush && self.output_frames_emitted >= out_frames {
                break;
            }
            if !flush && !self.has_future_samples(self.next_source_pos) {
                break;
            }
            self.render_frame_into(self.next_source_pos, output);
            self.next_source_pos += self.step;
            self.output_frames_emitted += 1;
        }
    }

    fn has_future_samples(&self, source_pos: f64) -> bool {
        let center = source_pos as i64;
        center + (self.filter.half_taps() as i64) < self.total_input_frames
    }

    fn frames_ready_to_render(&self, flush: bool, out_frames: i64) -> usize {
        if flush {
            return out_frames.saturating_sub(self.output_frames_emitted).max(0) as usize;
        }

        let max_center = self.total_input_frames - self.filter.half_taps() as i64 - 1;
        let current_center = self.next_source_pos as i64;
        if current_center > max_center {
            return 0;
        }
        (((max_center as f64 - self.next_source_pos) / self.step) as usize).saturating_add(1)
    }

    fn render_frame_into(&mut self, source_pos: f64, output: &mut Vec<f32>) {
        let center = source_pos as i64;
        let fraction = source_pos - center as f64;
        let coeffs = match self.special_ratio {
            SpecialRatio::Up2 | SpecialRatio::Down2 | SpecialRatio::General => {
                self.filter.coeffs_for_fraction(fraction)
            }
        };
        let first_tap_frame = center - self.filter.half_taps() as i64 + 1;
        let channels = self.config.channels;
        let taps = self.filter.taps();
        for channel in 0..channels {
            for tap in 0..taps {
                let absolute = first_tap_frame + tap as i64;
                self.scratch[tap] = self.sample_at(channel, absolute);
            }
            output.push(backend::dot_f32(
                self.backend,
                &self.scratch[..taps],
                coeffs,
            ));
        }
    }

    #[inline(always)]
    fn sample_at(&self, channel: usize, absolute_frame: i64) -> f32 {
        if absolute_frame < self.start_frame || absolute_frame >= self.total_input_frames {
            return 0.0;
        }
        let local = (absolute_frame - self.start_frame) as usize;
        self.channels[channel][local]
    }

    fn discard_consumed(&mut self) {
        let keep_from = (self.next_source_pos as i64 - self.filter.half_taps() as i64 - 2)
            .max(self.start_frame);
        let remove = (keep_from - self.start_frame) as usize;
        if remove == 0 {
            return;
        }
        for channel in &mut self.channels {
            channel.drain(0..remove);
        }
        self.start_frame = keep_from;
    }
}

#[inline(always)]
fn f32_to_i16(sample: f32) -> i16 {
    let scaled = (sample.clamp(-1.0, 1.0) * 32767.0).round();
    scaled as i16
}

#[inline]
fn expected_output_frames(input_frames: i64, input_rate: u32, output_rate: u32) -> i64 {
    let input_frames = input_frames.max(0) as u64;
    let input_rate = input_rate as u64;
    let output_rate = output_rate as u64;
    (input_frames * output_rate).div_ceil(input_rate) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, Error, Quality, ResamplerConfig};

    fn cfg(input_rate: u32, output_rate: u32, channels: usize) -> ResamplerConfig {
        ResamplerConfig {
            input_rate,
            output_rate,
            channels,
            quality: Quality::Fast,
            backend: Backend::Scalar,
        }
    }

    #[test]
    fn rejects_unaligned_input() {
        let mut resampler = Resampler::<f32>::new(cfg(48_000, 44_100, 2)).unwrap();
        let err = resampler
            .process(&[0.0, 1.0, 2.0], &mut Vec::new())
            .unwrap_err();
        assert_eq!(
            err,
            Error::InputNotFrameAligned {
                samples: 3,
                channels: 2
            }
        );
    }

    #[test]
    fn chunked_matches_one_shot() {
        let input: Vec<f32> = (0..512)
            .flat_map(|i| {
                let s = ((i as f32) * 0.03).sin();
                [s, -s]
            })
            .collect();
        let mut one = Resampler::<f32>::new(cfg(44_100, 48_000, 2)).unwrap();
        let mut one_out = Vec::new();
        one.process(&input, &mut one_out).unwrap();
        one.finish(&mut one_out).unwrap();

        let mut chunked = Resampler::<f32>::new(cfg(44_100, 48_000, 2)).unwrap();
        let mut chunked_out = Vec::new();
        for chunk in input.chunks(62) {
            let len = chunk.len() - (chunk.len() % 2);
            chunked.process(&chunk[..len], &mut chunked_out).unwrap();
        }
        chunked.finish(&mut chunked_out).unwrap();

        assert_eq!(one_out.len(), chunked_out.len());
        for (a, b) in one_out.iter().zip(chunked_out.iter()) {
            assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
        }
    }

    #[test]
    fn supports_8k_to_16k_and_back() {
        let input: Vec<f32> = (0..160).map(|i| ((i as f32) * 0.2).sin()).collect();
        let mut up = Resampler::<f32>::new(cfg(8_000, 16_000, 1)).unwrap();
        let mut up_out = Vec::new();
        up.process(&input, &mut up_out).unwrap();
        up.finish(&mut up_out).unwrap();
        assert_eq!(up_out.len(), 320);

        let mut down = Resampler::<f32>::new(cfg(16_000, 8_000, 1)).unwrap();
        let mut down_out = Vec::new();
        down.process(&up_out, &mut down_out).unwrap();
        down.finish(&mut down_out).unwrap();
        assert_eq!(down_out.len(), 160);
    }

    #[test]
    fn i16_path_clamps_and_outputs_expected_length() {
        let input = vec![0i16; 441];
        let mut resampler = Resampler::<i16>::new(cfg(44_100, 48_000, 1)).unwrap();
        let mut output = Vec::new();
        resampler.process(&input, &mut output).unwrap();
        resampler.finish(&mut output).unwrap();
        assert_eq!(output.len(), 480);
    }
}
