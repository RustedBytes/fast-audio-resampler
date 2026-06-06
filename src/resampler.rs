use std::marker::PhantomData;

use crate::backend::{self, SelectedBackend};
use crate::error::frame_alignment_error;
use crate::filter::FilterBank;
use crate::ring::RingBuffer;
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
enum Inner {
    F32(CoreF32),
    I16(CoreI16),
}

#[derive(Debug, Clone)]
pub struct Resampler<T> {
    inner: Inner,
    _sample: PhantomData<T>,
}

impl<T> Resampler<T> {
    #[inline]
    pub fn selected_backend(&self) -> SelectedBackend {
        match &self.inner {
            Inner::F32(core) => core.backend,
            Inner::I16(core) => core.backend,
        }
    }
}

impl Resampler<f32> {
    pub fn new(config: ResamplerConfig) -> Result<Self> {
        Ok(Self {
            inner: Inner::F32(CoreF32::new(config)?),
            _sample: PhantomData,
        })
    }

    #[inline]
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<ProcessStats> {
        match &mut self.inner {
            Inner::F32(core) => core.process(input, output),
            Inner::I16(_) => unreachable!("f32 resampler cannot hold i16 core"),
        }
    }

    #[inline]
    pub fn finish(&mut self, output: &mut Vec<f32>) -> Result<ProcessStats> {
        match &mut self.inner {
            Inner::F32(core) => core.finish(output),
            Inner::I16(_) => unreachable!("f32 resampler cannot hold i16 core"),
        }
    }
}

impl Resampler<i16> {
    pub fn new(config: ResamplerConfig) -> Result<Self> {
        Ok(Self {
            inner: Inner::I16(CoreI16::new(config)?),
            _sample: PhantomData,
        })
    }

    #[inline]
    pub fn process(&mut self, input: &[i16], output: &mut Vec<i16>) -> Result<ProcessStats> {
        match &mut self.inner {
            Inner::I16(core) => core.process(input, output),
            Inner::F32(_) => unreachable!("i16 resampler cannot hold f32 core"),
        }
    }

    #[inline]
    pub fn finish(&mut self, output: &mut Vec<i16>) -> Result<ProcessStats> {
        match &mut self.inner {
            Inner::I16(core) => core.finish(output),
            Inner::F32(_) => unreachable!("i16 resampler cannot hold f32 core"),
        }
    }
}

#[derive(Debug, Clone)]
struct CoreF32 {
    config: ResamplerConfig,
    backend: SelectedBackend,
    filter: FilterBank,
    channels: Vec<RingBuffer<f32>>,
    total_input_frames: i64,
    next_source_pos: f64,
    output_frames_emitted: i64,
    step: f64,
    special_ratio: SpecialRatio,
    scratch: Vec<f32>,
    finished: bool,
}

#[derive(Debug, Clone)]
struct CoreI16 {
    config: ResamplerConfig,
    backend: SelectedBackend,
    filter: FilterBank,
    channels: Vec<RingBuffer<i16>>,
    total_input_frames: i64,
    next_source_pos: f64,
    output_frames_emitted: i64,
    step: f64,
    special_ratio: SpecialRatio,
    scratch: Vec<i16>,
    finished: bool,
}

struct CommonInit<T: Copy + Default> {
    backend: SelectedBackend,
    filter: FilterBank,
    channels: Vec<RingBuffer<T>>,
    step: f64,
    special_ratio: SpecialRatio,
    taps: usize,
}

impl CoreF32 {
    fn new(config: ResamplerConfig) -> Result<Self> {
        let init = init_common::<f32>(config)?;
        Ok(Self {
            config,
            backend: init.backend,
            filter: init.filter,
            channels: init.channels,
            total_input_frames: 0,
            next_source_pos: 0.0,
            output_frames_emitted: 0,
            step: init.step,
            special_ratio: init.special_ratio,
            scratch: vec![0.0; init.taps],
            finished: false,
        })
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<ProcessStats> {
        if self.finished {
            return Ok(self.stats(0, 0));
        }
        let input_frames = self.append_input(input)?;
        let before = output.len();
        self.render_available(output, false);
        self.discard_consumed();
        Ok(self.stats(input_frames, (output.len() - before) / self.config.channels))
    }

    fn finish(&mut self, output: &mut Vec<f32>) -> Result<ProcessStats> {
        let before = output.len();
        self.render_available(output, true);
        self.finished = true;
        Ok(self.stats(0, (output.len() - before) / self.config.channels))
    }

    fn append_input(&mut self, input: &[f32]) -> Result<usize> {
        if !input.len().is_multiple_of(self.config.channels) {
            return Err(frame_alignment_error(input.len(), self.config.channels));
        }
        let frames = input.len() / self.config.channels;
        ensure_history_capacity(&mut self.channels, frames, self.filter.taps());
        for frame in input.chunks_exact(self.config.channels) {
            for (channel, &sample) in frame.iter().enumerate() {
                self.channels[channel].push(sample);
            }
        }
        self.total_input_frames += frames as i64;
        Ok(frames)
    }

    fn render_available(&mut self, output: &mut Vec<f32>, flush: bool) {
        let out_frames = self.output_frames_for_flush(flush);
        output.reserve(self.frames_ready_to_render(flush, out_frames) * self.config.channels);
        while self.should_render(flush, out_frames) {
            self.render_frame_into(output);
            self.advance_output();
        }
    }

    fn render_frame_into(&mut self, output: &mut Vec<f32>) {
        let center = self.next_source_pos as i64;
        let phase = self.phase_index(center);
        let coeffs = self.filter.coeffs_for_phase(phase);
        let first_tap_frame = center - self.filter.half_taps() as i64 + 1;
        let taps = self.filter.taps();
        for channel in 0..self.config.channels {
            for tap in 0..taps {
                self.scratch[tap] = self.sample_at(channel, first_tap_frame + tap as i64);
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
        self.channels[channel].get(absolute_frame).unwrap_or(0.0)
    }

    fn discard_consumed(&mut self) {
        let keep_from = (self.next_source_pos as i64 - self.filter.half_taps() as i64 - 2)
            .max(self.channels[0].start_frame());
        for channel in &mut self.channels {
            channel.discard_before(keep_from);
        }
    }

    #[inline]
    fn output_frames_for_flush(&self, flush: bool) -> i64 {
        if flush {
            expected_output_frames(
                self.total_input_frames,
                self.config.input_rate,
                self.config.output_rate,
            )
        } else {
            i64::MAX
        }
    }

    #[inline]
    fn should_render(&self, flush: bool, out_frames: i64) -> bool {
        if flush {
            self.output_frames_emitted < out_frames
        } else {
            has_future_samples(
                self.next_source_pos,
                self.filter.half_taps(),
                self.total_input_frames,
            )
        }
    }

    #[inline]
    fn frames_ready_to_render(&self, flush: bool, out_frames: i64) -> usize {
        frames_ready_to_render(
            flush,
            out_frames,
            self.output_frames_emitted,
            self.total_input_frames,
            self.filter.half_taps(),
            self.next_source_pos,
            self.step,
        )
    }

    #[inline]
    fn phase_index(&self, center: i64) -> usize {
        phase_index(
            self.special_ratio,
            self.output_frames_emitted,
            self.next_source_pos,
            center,
            self.filter.phase_count(),
        )
    }

    #[inline]
    fn advance_output(&mut self) {
        self.next_source_pos += self.step;
        self.output_frames_emitted += 1;
    }

    #[inline]
    fn stats(&self, input_frames: usize, output_frames: usize) -> ProcessStats {
        ProcessStats {
            input_frames,
            output_frames,
            backend: self.backend,
        }
    }
}

impl CoreI16 {
    fn new(config: ResamplerConfig) -> Result<Self> {
        let init = init_common::<i16>(config)?;
        Ok(Self {
            config,
            backend: init.backend,
            filter: init.filter,
            channels: init.channels,
            total_input_frames: 0,
            next_source_pos: 0.0,
            output_frames_emitted: 0,
            step: init.step,
            special_ratio: init.special_ratio,
            scratch: vec![0; init.taps],
            finished: false,
        })
    }

    fn process(&mut self, input: &[i16], output: &mut Vec<i16>) -> Result<ProcessStats> {
        if self.finished {
            return Ok(self.stats(0, 0));
        }
        let input_frames = self.append_input(input)?;
        let before = output.len();
        self.render_available(output, false);
        self.discard_consumed();
        Ok(self.stats(input_frames, (output.len() - before) / self.config.channels))
    }

    fn finish(&mut self, output: &mut Vec<i16>) -> Result<ProcessStats> {
        let before = output.len();
        self.render_available(output, true);
        self.finished = true;
        Ok(self.stats(0, (output.len() - before) / self.config.channels))
    }

    fn append_input(&mut self, input: &[i16]) -> Result<usize> {
        if !input.len().is_multiple_of(self.config.channels) {
            return Err(frame_alignment_error(input.len(), self.config.channels));
        }
        let frames = input.len() / self.config.channels;
        ensure_history_capacity(&mut self.channels, frames, self.filter.taps());
        for frame in input.chunks_exact(self.config.channels) {
            for (channel, &sample) in frame.iter().enumerate() {
                self.channels[channel].push(sample);
            }
        }
        self.total_input_frames += frames as i64;
        Ok(frames)
    }

    fn render_available(&mut self, output: &mut Vec<i16>, flush: bool) {
        let out_frames = self.output_frames_for_flush(flush);
        output.reserve(self.frames_ready_to_render(flush, out_frames) * self.config.channels);
        while self.should_render(flush, out_frames) {
            self.render_frame_into(output);
            self.advance_output();
        }
    }

    fn render_frame_into(&mut self, output: &mut Vec<i16>) {
        let center = self.next_source_pos as i64;
        let phase = self.phase_index(center);
        let coeffs = self.filter.coeffs_q15_for_phase(phase);
        let first_tap_frame = center - self.filter.half_taps() as i64 + 1;
        let taps = self.filter.taps();
        for channel in 0..self.config.channels {
            for tap in 0..taps {
                self.scratch[tap] = self.sample_at(channel, first_tap_frame + tap as i64);
            }
            output.push(backend::dot_i16_q15(
                self.backend,
                &self.scratch[..taps],
                coeffs,
            ));
        }
    }

    #[inline(always)]
    fn sample_at(&self, channel: usize, absolute_frame: i64) -> i16 {
        self.channels[channel].get(absolute_frame).unwrap_or(0)
    }

    fn discard_consumed(&mut self) {
        let keep_from = (self.next_source_pos as i64 - self.filter.half_taps() as i64 - 2)
            .max(self.channels[0].start_frame());
        for channel in &mut self.channels {
            channel.discard_before(keep_from);
        }
    }

    #[inline]
    fn output_frames_for_flush(&self, flush: bool) -> i64 {
        if flush {
            expected_output_frames(
                self.total_input_frames,
                self.config.input_rate,
                self.config.output_rate,
            )
        } else {
            i64::MAX
        }
    }

    #[inline]
    fn should_render(&self, flush: bool, out_frames: i64) -> bool {
        if flush {
            self.output_frames_emitted < out_frames
        } else {
            has_future_samples(
                self.next_source_pos,
                self.filter.half_taps(),
                self.total_input_frames,
            )
        }
    }

    #[inline]
    fn frames_ready_to_render(&self, flush: bool, out_frames: i64) -> usize {
        frames_ready_to_render(
            flush,
            out_frames,
            self.output_frames_emitted,
            self.total_input_frames,
            self.filter.half_taps(),
            self.next_source_pos,
            self.step,
        )
    }

    #[inline]
    fn phase_index(&self, center: i64) -> usize {
        phase_index(
            self.special_ratio,
            self.output_frames_emitted,
            self.next_source_pos,
            center,
            self.filter.phase_count(),
        )
    }

    #[inline]
    fn advance_output(&mut self) {
        self.next_source_pos += self.step;
        self.output_frames_emitted += 1;
    }

    #[inline]
    fn stats(&self, input_frames: usize, output_frames: usize) -> ProcessStats {
        ProcessStats {
            input_frames,
            output_frames,
            backend: self.backend,
        }
    }
}

fn init_common<T: Copy + Default>(config: ResamplerConfig) -> Result<CommonInit<T>> {
    config.validate()?;
    let backend = config.backend.select()?;
    let filter = FilterBank::new(config.input_rate, config.output_rate, config.quality);
    let taps = filter.taps();
    let ring_capacity = (taps * 4).max(taps + 8);
    let channels = (0..config.channels)
        .map(|_| RingBuffer::with_capacity(ring_capacity))
        .collect();
    let step = config.input_rate as f64 / config.output_rate as f64;
    let special_ratio = match (config.input_rate, config.output_rate) {
        (8_000, 16_000) => SpecialRatio::Up2,
        (16_000, 8_000) => SpecialRatio::Down2,
        _ => SpecialRatio::General,
    };
    Ok(CommonInit {
        backend,
        filter,
        channels,
        step,
        special_ratio,
        taps,
    })
}

fn ensure_history_capacity<T: Copy + Default>(
    channels: &mut [RingBuffer<T>],
    incoming_frames: usize,
    taps: usize,
) {
    for channel in channels {
        let live = (channel.end_frame() - channel.start_frame()).max(0) as usize;
        channel.ensure_capacity((taps * 4).max(live + incoming_frames + taps + 8));
    }
}

#[inline]
fn has_future_samples(source_pos: f64, half_taps: usize, total_input_frames: i64) -> bool {
    let center = source_pos as i64;
    center + (half_taps as i64) < total_input_frames
}

#[inline]
fn frames_ready_to_render(
    flush: bool,
    out_frames: i64,
    output_frames_emitted: i64,
    total_input_frames: i64,
    half_taps: usize,
    next_source_pos: f64,
    step: f64,
) -> usize {
    if flush {
        return out_frames.saturating_sub(output_frames_emitted).max(0) as usize;
    }

    let max_center = total_input_frames - half_taps as i64 - 1;
    let current_center = next_source_pos as i64;
    if current_center > max_center {
        return 0;
    }
    (((max_center as f64 - next_source_pos) / step) as usize).saturating_add(1)
}

#[inline]
fn phase_index(
    special_ratio: SpecialRatio,
    output_frames_emitted: i64,
    source_pos: f64,
    center: i64,
    phases: usize,
) -> usize {
    match special_ratio {
        SpecialRatio::Up2 => {
            if output_frames_emitted & 1 == 0 {
                0
            } else {
                phases / 2
            }
        }
        SpecialRatio::Down2 => 0,
        SpecialRatio::General => {
            let fraction = source_pos - center as f64;
            ((fraction * phases as f64) as usize).min(phases - 1)
        }
    }
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
    use crate::{Backend, Error, Quality};

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
        for chunk in input.chunks(2) {
            chunked.process(chunk, &mut chunked_out).unwrap();
        }
        chunked.finish(&mut chunked_out).unwrap();

        assert_eq!(one_out.len(), chunked_out.len());
        for (a, b) in one_out.iter().zip(chunked_out.iter()) {
            assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
        }
    }

    #[test]
    fn large_chunk_forces_ring_growth_and_matches_one_shot() {
        let input: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.013).sin()).collect();
        let mut one = Resampler::<f32>::new(cfg(48_000, 44_100, 1)).unwrap();
        let mut one_out = Vec::new();
        one.process(&input, &mut one_out).unwrap();
        one.finish(&mut one_out).unwrap();

        let mut chunked = Resampler::<f32>::new(cfg(48_000, 44_100, 1)).unwrap();
        let mut chunked_out = Vec::new();
        chunked.process(&input[..3000], &mut chunked_out).unwrap();
        chunked.process(&input[3000..], &mut chunked_out).unwrap();
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
    fn special_ratio_chunked_paths_match_one_shot_for_i16_stereo() {
        let input: Vec<i16> = (0..160)
            .flat_map(|i| {
                let s = (((i as f32) * 0.2).sin() * 16_000.0) as i16;
                [s, -s]
            })
            .collect();
        let mut one = Resampler::<i16>::new(cfg(8_000, 16_000, 2)).unwrap();
        let mut one_out = Vec::new();
        one.process(&input, &mut one_out).unwrap();
        one.finish(&mut one_out).unwrap();
        assert_eq!(one_out.len(), 640);

        let mut chunked = Resampler::<i16>::new(cfg(8_000, 16_000, 2)).unwrap();
        let mut chunked_out = Vec::new();
        for chunk in input.chunks(10) {
            chunked.process(chunk, &mut chunked_out).unwrap();
        }
        chunked.finish(&mut chunked_out).unwrap();
        assert_eq!(one_out, chunked_out);
    }

    #[test]
    fn downsampling_filters_instead_of_dropping_every_other_sample() {
        let input: Vec<f32> = (0..320)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut resampler = Resampler::<f32>::new(cfg(16_000, 8_000, 1)).unwrap();
        let mut output = Vec::new();
        resampler.process(&input, &mut output).unwrap();
        resampler.finish(&mut output).unwrap();
        assert_eq!(output.len(), 160);
        assert!(
            output.iter().any(|&sample| sample < 0.5),
            "output looks like naive even-sample decimation"
        );
    }

    #[test]
    fn i16_path_clamps_and_outputs_expected_length() {
        let input = vec![i16::MAX; 441];
        let mut resampler = Resampler::<i16>::new(cfg(44_100, 48_000, 1)).unwrap();
        let mut output = Vec::new();
        resampler.process(&input, &mut output).unwrap();
        resampler.finish(&mut output).unwrap();
        assert_eq!(output.len(), 480);
        assert!(output.iter().any(|&sample| sample > 0));
    }

    #[test]
    fn i16_silence_remains_silence_and_impulse_is_bounded() {
        let mut silence = Resampler::<i16>::new(cfg(44_100, 48_000, 1)).unwrap();
        let mut silence_out = Vec::new();
        silence.process(&vec![0; 441], &mut silence_out).unwrap();
        silence.finish(&mut silence_out).unwrap();
        assert!(silence_out.iter().all(|&sample| sample == 0));

        let mut impulse_input = vec![0i16; 441];
        impulse_input[100] = 24_000;
        let mut impulse = Resampler::<i16>::new(cfg(44_100, 48_000, 1)).unwrap();
        let mut impulse_out = Vec::new();
        impulse.process(&impulse_input, &mut impulse_out).unwrap();
        impulse.finish(&mut impulse_out).unwrap();
        assert!(impulse_out.iter().any(|&sample| sample != 0));
        assert!(
            impulse_out
                .iter()
                .all(|&sample| (sample as i32).abs() <= 24_000)
        );
    }

    #[test]
    fn i16_scalar_and_avx2_are_close_when_available() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            let input: Vec<i16> = (0..512)
                .map(|i| (((i as f32) * 0.09).sin() * 12_000.0) as i16)
                .collect();
            let mut scalar = Resampler::<i16>::new(cfg(44_100, 48_000, 1)).unwrap();
            let mut scalar_out = Vec::new();
            scalar.process(&input, &mut scalar_out).unwrap();
            scalar.finish(&mut scalar_out).unwrap();

            let mut avx_cfg = cfg(44_100, 48_000, 1);
            avx_cfg.backend = Backend::Avx2;
            let mut avx = Resampler::<i16>::new(avx_cfg).unwrap();
            let mut avx_out = Vec::new();
            avx.process(&input, &mut avx_out).unwrap();
            avx.finish(&mut avx_out).unwrap();

            assert_eq!(scalar_out.len(), avx_out.len());
            for (a, b) in scalar_out.iter().zip(avx_out.iter()) {
                assert!((*a as i32 - *b as i32).abs() <= 1, "{a} != {b}");
            }
        }
    }
}
