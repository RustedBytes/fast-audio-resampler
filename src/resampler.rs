use std::marker::PhantomData;

use crate::backend::{self, SelectedBackend};
use crate::error::frame_alignment_error;
use crate::filter::FilterBank;
use crate::iir::PolyphaseIir2x;
use crate::ring::RingBuffer;
use crate::{Error, Quality, ResamplerConfig, Result};

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

    #[inline]
    pub fn required_output_capacity(&self, input_frames: usize) -> usize {
        match &self.inner {
            Inner::F32(core) => core.required_output_capacity(input_frames),
            Inner::I16(core) => core.required_output_capacity(input_frames),
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

    #[inline]
    pub fn flush(&mut self, output: &mut Vec<f32>) -> Result<ProcessStats> {
        self.finish(output)
    }

    pub fn process_into_slice(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<ProcessStats> {
        if !input.len().is_multiple_of(self.config_channels()) {
            return Err(frame_alignment_error(input.len(), self.config_channels()));
        }
        let required = self.required_output_capacity(input.len() / self.config_channels());
        if output.len() < required {
            return Err(Error::OutputTooSmall {
                required,
                available: output.len(),
            });
        }
        let mut tmp = Vec::with_capacity(required);
        let stats = self.process(input, &mut tmp)?;
        output[..tmp.len()].copy_from_slice(&tmp);
        Ok(stats)
    }

    #[inline]
    pub fn reset(&mut self) {
        match &mut self.inner {
            Inner::F32(core) => core.reset(),
            Inner::I16(_) => unreachable!("f32 resampler cannot hold i16 core"),
        }
    }

    #[inline]
    fn config_channels(&self) -> usize {
        match &self.inner {
            Inner::F32(core) => core.config.channels,
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

    #[inline]
    pub fn flush(&mut self, output: &mut Vec<i16>) -> Result<ProcessStats> {
        self.finish(output)
    }

    pub fn process_into_slice(
        &mut self,
        input: &[i16],
        output: &mut [i16],
    ) -> Result<ProcessStats> {
        if !input.len().is_multiple_of(self.config_channels()) {
            return Err(frame_alignment_error(input.len(), self.config_channels()));
        }
        let required = self.required_output_capacity(input.len() / self.config_channels());
        if output.len() < required {
            return Err(Error::OutputTooSmall {
                required,
                available: output.len(),
            });
        }
        let mut tmp = Vec::with_capacity(required);
        let stats = self.process(input, &mut tmp)?;
        output[..tmp.len()].copy_from_slice(&tmp);
        Ok(stats)
    }

    #[inline]
    pub fn reset(&mut self) {
        match &mut self.inner {
            Inner::I16(core) => core.reset(),
            Inner::F32(_) => unreachable!("i16 resampler cannot hold f32 core"),
        }
    }

    #[inline]
    fn config_channels(&self) -> usize {
        match &self.inner {
            Inner::I16(core) => core.config.channels,
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
    iir: Option<PolyphaseIir2x>,
    iir_input_frames_consumed: i64,
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
    iir: Option<PolyphaseIir2x>,
    iir_input_frames_consumed: i64,
    scratch: Vec<i16>,
    finished: bool,
}

struct CommonInit<T: Copy + Default> {
    backend: SelectedBackend,
    filter: FilterBank,
    channels: Vec<RingBuffer<T>>,
    step: f64,
    special_ratio: SpecialRatio,
    iir: Option<PolyphaseIir2x>,
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
            iir: init.iir,
            iir_input_frames_consumed: 0,
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
        if self.iir.is_some() {
            self.render_iir_available(output, false);
            self.discard_iir_consumed();
        } else {
            self.render_available(output, false);
            self.discard_consumed();
        }
        Ok(self.stats(input_frames, (output.len() - before) / self.config.channels))
    }

    fn finish(&mut self, output: &mut Vec<f32>) -> Result<ProcessStats> {
        let before = output.len();
        if self.iir.is_some() {
            self.render_iir_available(output, true);
        } else {
            self.render_available(output, true);
        }
        self.finished = true;
        Ok(self.stats(0, (output.len() - before) / self.config.channels))
    }

    #[inline]
    fn required_output_capacity(&self, input_frames: usize) -> usize {
        required_output_capacity(
            input_frames,
            self.config.input_rate,
            self.config.output_rate,
            self.config.channels,
            self.filter.half_taps(),
        )
    }

    fn reset(&mut self) {
        let config = self.config;
        if let Ok(new_core) = Self::new(config) {
            *self = new_core;
        }
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
        match self.special_ratio {
            SpecialRatio::Up2 => {
                self.render_up2_frame_into(output);
                return;
            }
            SpecialRatio::Down2 => {
                self.render_down2_frame_into(output);
                return;
            }
            SpecialRatio::General => {}
        }

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

    fn render_iir_available(&mut self, output: &mut Vec<f32>, flush: bool) {
        if self.iir.as_ref().is_some_and(PolyphaseIir2x::is_up) {
            let ready = self
                .total_input_frames
                .saturating_sub(self.iir_input_frames_consumed)
                .max(0) as usize;
            output.reserve(ready * 2 * self.config.channels);
            while self.iir_input_frames_consumed < self.total_input_frames {
                let frame = self.iir_input_frames_consumed;
                for channel in 0..self.config.channels {
                    let sample = self.sample_at(channel, frame);
                    let pair = self.iir.as_mut().unwrap().process_up(channel, sample);
                    output.extend_from_slice(&pair);
                }
                self.iir_input_frames_consumed += 1;
                self.output_frames_emitted += 2;
            }
            return;
        }

        let complete_pairs = (self.total_input_frames - self.iir_input_frames_consumed) / 2;
        let flush_tail =
            usize::from(flush && self.iir_input_frames_consumed < self.total_input_frames);
        output.reserve((complete_pairs as usize + flush_tail) * self.config.channels);
        while self.iir_input_frames_consumed + 1 < self.total_input_frames {
            self.render_iir_down_pair(
                output,
                self.iir_input_frames_consumed,
                self.iir_input_frames_consumed + 1,
            );
            self.iir_input_frames_consumed += 2;
            self.output_frames_emitted += 1;
        }
        if flush && self.iir_input_frames_consumed < self.total_input_frames {
            self.render_iir_down_pair(output, self.iir_input_frames_consumed, -1);
            self.iir_input_frames_consumed += 1;
            self.output_frames_emitted += 1;
        }
    }

    fn render_iir_down_pair(&mut self, output: &mut Vec<f32>, even_frame: i64, odd_frame: i64) {
        for channel in 0..self.config.channels {
            let even = self.sample_at(channel, even_frame);
            let odd = if odd_frame >= 0 {
                self.sample_at(channel, odd_frame)
            } else {
                0.0
            };
            output.push(self.iir.as_mut().unwrap().process_down(channel, even, odd));
        }
    }

    fn render_up2_frame_into(&self, output: &mut Vec<f32>) {
        let source_frame = self.output_frames_emitted / 2;
        let half_band = self.filter.half_band();
        if self.output_frames_emitted & 1 == 0 {
            for channel in 0..self.config.channels {
                output.push(self.sample_at(channel, source_frame));
            }
            return;
        }

        let offsets = half_band.side_input_offsets_for_up();
        let coeffs = half_band.side_coeffs_up();
        for channel in 0..self.config.channels {
            let mut acc = 0.0f32;
            for i in 0..coeffs.len() {
                acc += self.sample_at(channel, source_frame + offsets[i]) * coeffs[i];
            }
            output.push(acc);
        }
    }

    fn render_down2_frame_into(&self, output: &mut Vec<f32>) {
        let center = self.output_frames_emitted * 2;
        let half_band = self.filter.half_band();
        let offsets = half_band.side_offsets();
        let coeffs = half_band.side_coeffs();
        for channel in 0..self.config.channels {
            let mut acc = self.sample_at(channel, center) * half_band.center_coeff();
            for i in 0..coeffs.len() {
                acc += self.sample_at(channel, center - offsets[i]) * coeffs[i];
            }
            output.push(acc);
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

    fn discard_iir_consumed(&mut self) {
        let keep_from = self
            .iir_input_frames_consumed
            .saturating_sub(2)
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
            iir: init.iir,
            iir_input_frames_consumed: 0,
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
        if self.iir.is_some() {
            self.render_iir_available(output, false);
            self.discard_iir_consumed();
        } else {
            self.render_available(output, false);
            self.discard_consumed();
        }
        Ok(self.stats(input_frames, (output.len() - before) / self.config.channels))
    }

    fn finish(&mut self, output: &mut Vec<i16>) -> Result<ProcessStats> {
        let before = output.len();
        if self.iir.is_some() {
            self.render_iir_available(output, true);
        } else {
            self.render_available(output, true);
        }
        self.finished = true;
        Ok(self.stats(0, (output.len() - before) / self.config.channels))
    }

    #[inline]
    fn required_output_capacity(&self, input_frames: usize) -> usize {
        required_output_capacity(
            input_frames,
            self.config.input_rate,
            self.config.output_rate,
            self.config.channels,
            self.filter.half_taps(),
        )
    }

    fn reset(&mut self) {
        let config = self.config;
        if let Ok(new_core) = Self::new(config) {
            *self = new_core;
        }
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
        match self.special_ratio {
            SpecialRatio::Up2 => {
                self.render_up2_frame_into(output);
                return;
            }
            SpecialRatio::Down2 => {
                self.render_down2_frame_into(output);
                return;
            }
            SpecialRatio::General => {}
        }

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

    fn render_iir_available(&mut self, output: &mut Vec<i16>, flush: bool) {
        if self.iir.as_ref().is_some_and(PolyphaseIir2x::is_up) {
            let ready = self
                .total_input_frames
                .saturating_sub(self.iir_input_frames_consumed)
                .max(0) as usize;
            output.reserve(ready * 2 * self.config.channels);
            while self.iir_input_frames_consumed < self.total_input_frames {
                let frame = self.iir_input_frames_consumed;
                for channel in 0..self.config.channels {
                    let sample = self.sample_at(channel, frame) as f32;
                    let pair = self.iir.as_mut().unwrap().process_up(channel, sample);
                    output.push(f32_to_i16(pair[0]));
                    output.push(f32_to_i16(pair[1]));
                }
                self.iir_input_frames_consumed += 1;
                self.output_frames_emitted += 2;
            }
            return;
        }

        let complete_pairs = (self.total_input_frames - self.iir_input_frames_consumed) / 2;
        let flush_tail =
            usize::from(flush && self.iir_input_frames_consumed < self.total_input_frames);
        output.reserve((complete_pairs as usize + flush_tail) * self.config.channels);
        while self.iir_input_frames_consumed + 1 < self.total_input_frames {
            self.render_iir_down_pair(
                output,
                self.iir_input_frames_consumed,
                self.iir_input_frames_consumed + 1,
            );
            self.iir_input_frames_consumed += 2;
            self.output_frames_emitted += 1;
        }
        if flush && self.iir_input_frames_consumed < self.total_input_frames {
            self.render_iir_down_pair(output, self.iir_input_frames_consumed, -1);
            self.iir_input_frames_consumed += 1;
            self.output_frames_emitted += 1;
        }
    }

    fn render_iir_down_pair(&mut self, output: &mut Vec<i16>, even_frame: i64, odd_frame: i64) {
        for channel in 0..self.config.channels {
            let even = self.sample_at(channel, even_frame) as f32;
            let odd = if odd_frame >= 0 {
                self.sample_at(channel, odd_frame) as f32
            } else {
                0.0
            };
            let sample = self.iir.as_mut().unwrap().process_down(channel, even, odd);
            output.push(f32_to_i16(sample));
        }
    }

    fn render_up2_frame_into(&self, output: &mut Vec<i16>) {
        let source_frame = self.output_frames_emitted / 2;
        let half_band = self.filter.half_band();
        if self.output_frames_emitted & 1 == 0 {
            for channel in 0..self.config.channels {
                output.push(self.sample_at(channel, source_frame));
            }
            return;
        }

        let offsets = half_band.side_input_offsets_for_up();
        let coeffs = half_band.side_coeffs_up_q15();
        for channel in 0..self.config.channels {
            let mut acc = 0i64;
            for i in 0..coeffs.len() {
                acc += self.sample_at(channel, source_frame + offsets[i]) as i64 * coeffs[i] as i64;
            }
            output.push(q15_acc_to_i16(acc));
        }
    }

    fn render_down2_frame_into(&self, output: &mut Vec<i16>) {
        let center = self.output_frames_emitted * 2;
        let half_band = self.filter.half_band();
        let offsets = half_band.side_offsets();
        let coeffs = half_band.side_coeffs_q15();
        for channel in 0..self.config.channels {
            let mut acc =
                self.sample_at(channel, center) as i64 * half_band.center_coeff_q15() as i64;
            for i in 0..coeffs.len() {
                acc += self.sample_at(channel, center - offsets[i]) as i64 * coeffs[i] as i64;
            }
            output.push(q15_acc_to_i16(acc));
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

    fn discard_iir_consumed(&mut self) {
        let keep_from = self
            .iir_input_frames_consumed
            .saturating_sub(2)
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
    let ring_capacity = (taps * 4).max(
        config
            .max_input_frames_per_chunk
            .unwrap_or(0)
            .saturating_add(taps)
            .saturating_add(8),
    );
    let channels = (0..config.channels)
        .map(|_| RingBuffer::with_capacity(ring_capacity))
        .collect();
    let step = config.input_rate as f64 / config.output_rate as f64;
    let special_ratio = match (config.input_rate, config.output_rate) {
        (8_000, 16_000) => SpecialRatio::Up2,
        (16_000, 8_000) => SpecialRatio::Down2,
        _ => SpecialRatio::General,
    };
    let iir = match (config.quality, special_ratio) {
        (Quality::Fast, SpecialRatio::Up2) => Some(PolyphaseIir2x::up(config.channels)),
        (Quality::Fast, SpecialRatio::Down2) => Some(PolyphaseIir2x::down(config.channels)),
        _ => None,
    };
    Ok(CommonInit {
        backend,
        filter,
        channels,
        step,
        special_ratio,
        iir,
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

#[inline]
fn required_output_capacity(
    input_frames: usize,
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    half_taps: usize,
) -> usize {
    let frames = expected_output_frames(
        input_frames as i64 + half_taps as i64 + 2,
        input_rate,
        output_rate,
    )
    .max(0) as usize;
    frames.saturating_mul(channels)
}

#[inline(always)]
fn q15_acc_to_i16(acc: i64) -> i16 {
    let rounded = (acc + (1 << 14)) >> 15;
    rounded.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

#[inline(always)]
fn f32_to_i16(sample: f32) -> i16 {
    sample.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, Error, Quality};

    fn cfg(input_rate: u32, output_rate: u32, channels: usize) -> ResamplerConfig {
        cfg_with_quality(input_rate, output_rate, channels, Quality::Fast)
    }

    fn cfg_with_quality(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        quality: Quality,
    ) -> ResamplerConfig {
        ResamplerConfig {
            input_rate,
            output_rate,
            channels,
            quality,
            backend: Backend::Scalar,
            max_input_frames_per_chunk: None,
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
    fn f32_half_band_special_ratios_match_chunked_and_one_shot() {
        for (input_rate, output_rate, frames) in [(8_000, 16_000, 320), (16_000, 8_000, 640)] {
            let input: Vec<f32> = (0..frames)
                .flat_map(|i| {
                    let s = ((i as f32) * 0.047).sin() * 0.5;
                    [s, -s]
                })
                .collect();
            let mut one = Resampler::<f32>::new(cfg_with_quality(
                input_rate,
                output_rate,
                2,
                Quality::Balanced,
            ))
            .unwrap();
            let mut one_out = Vec::new();
            one.process(&input, &mut one_out).unwrap();
            one.finish(&mut one_out).unwrap();

            let mut chunked = Resampler::<f32>::new(cfg_with_quality(
                input_rate,
                output_rate,
                2,
                Quality::Balanced,
            ))
            .unwrap();
            let mut chunked_out = Vec::new();
            for chunk in input.chunks(14) {
                chunked.process(chunk, &mut chunked_out).unwrap();
            }
            chunked.finish(&mut chunked_out).unwrap();

            assert_eq!(
                one_out.len(),
                expected_output_frames(frames, input_rate, output_rate) as usize * 2
            );
            assert_eq!(one_out.len(), chunked_out.len());
            for (a, b) in one_out.iter().zip(chunked_out.iter()) {
                assert!((a - b).abs() < 1.0e-6, "{a} != {b}");
            }
        }
    }

    #[test]
    fn f32_half_band_up2_preserves_original_samples_on_even_outputs() {
        let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.13).sin()).collect();
        let mut resampler =
            Resampler::<f32>::new(cfg_with_quality(8_000, 16_000, 1, Quality::Balanced)).unwrap();
        let mut output = Vec::new();
        resampler.process(&input, &mut output).unwrap();
        resampler.finish(&mut output).unwrap();

        for (source, rendered) in input.iter().zip(output.iter().step_by(2)) {
            assert!((source - rendered).abs() < 1.0e-6, "{source} != {rendered}");
        }
    }

    #[test]
    fn f32_half_band_roundtrip_preserves_low_frequency_shape() {
        let input: Vec<f32> = (0..320).map(|i| ((i as f32) * 0.08).sin() * 0.5).collect();
        let mut up =
            Resampler::<f32>::new(cfg_with_quality(8_000, 16_000, 1, Quality::Balanced)).unwrap();
        let mut up_out = Vec::new();
        up.process(&input, &mut up_out).unwrap();
        up.finish(&mut up_out).unwrap();

        let mut down =
            Resampler::<f32>::new(cfg_with_quality(16_000, 8_000, 1, Quality::Balanced)).unwrap();
        let mut down_out = Vec::new();
        down.process(&up_out, &mut down_out).unwrap();
        down.finish(&mut down_out).unwrap();

        assert_eq!(input.len(), down_out.len());
        let mean_abs_error = input
            .iter()
            .zip(down_out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / input.len() as f32;
        assert!(mean_abs_error < 0.02, "mean abs error {mean_abs_error}");
    }

    #[test]
    fn f32_polyphase_iir_roundtrip_preserves_low_frequency_shape_after_latency() {
        let input: Vec<f32> = (0..320).map(|i| ((i as f32) * 0.08).sin() * 0.5).collect();
        let mut up = Resampler::<f32>::new(cfg(8_000, 16_000, 1)).unwrap();
        let mut up_out = Vec::new();
        up.process(&input, &mut up_out).unwrap();
        up.finish(&mut up_out).unwrap();

        let mut down = Resampler::<f32>::new(cfg(16_000, 8_000, 1)).unwrap();
        let mut down_out = Vec::new();
        down.process(&up_out, &mut down_out).unwrap();
        down.finish(&mut down_out).unwrap();

        assert_eq!(input.len(), down_out.len());
        let delay = 2;
        let mean_abs_error = input[..input.len() - delay]
            .iter()
            .zip(down_out[delay..].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (input.len() - delay) as f32;
        assert!(mean_abs_error < 0.02, "mean abs error {mean_abs_error}");
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
    fn i16_half_band_down2_chunked_matches_one_shot() {
        let input: Vec<i16> = (0..640)
            .flat_map(|i| {
                let s = (((i as f32) * 0.041).sin() * 12_000.0) as i16;
                [s, -s]
            })
            .collect();
        let mut one = Resampler::<i16>::new(cfg(16_000, 8_000, 2)).unwrap();
        let mut one_out = Vec::new();
        one.process(&input, &mut one_out).unwrap();
        one.finish(&mut one_out).unwrap();
        assert_eq!(one_out.len(), 640);

        let mut chunked = Resampler::<i16>::new(cfg(16_000, 8_000, 2)).unwrap();
        let mut chunked_out = Vec::new();
        for chunk in input.chunks(22) {
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
    fn half_band_special_ratios_keep_silence_silent_and_impulses_bounded() {
        for (input_rate, output_rate, frames) in [(8_000, 16_000, 160), (16_000, 8_000, 320)] {
            let mut silence = Resampler::<i16>::new(cfg(input_rate, output_rate, 1)).unwrap();
            let mut silence_out = Vec::new();
            silence.process(&vec![0; frames], &mut silence_out).unwrap();
            silence.finish(&mut silence_out).unwrap();
            assert!(silence_out.iter().all(|&sample| sample == 0));

            let mut impulse_input = vec![0i16; frames];
            impulse_input[frames / 2] = 24_000;
            let mut impulse = Resampler::<i16>::new(cfg(input_rate, output_rate, 1)).unwrap();
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
    }

    #[test]
    fn i16_half_band_downsampling_filters_instead_of_decimating() {
        let input: Vec<i16> = (0..320)
            .map(|i| if i % 2 == 0 { 16_000 } else { -16_000 })
            .collect();
        let mut resampler = Resampler::<i16>::new(cfg(16_000, 8_000, 1)).unwrap();
        let mut output = Vec::new();
        resampler.process(&input, &mut output).unwrap();
        resampler.finish(&mut output).unwrap();
        assert_eq!(output.len(), 160);
        assert!(
            output.iter().any(|&sample| sample < 8_000),
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

    #[test]
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn f32_scalar_and_rvv_match_for_special_and_general_ratios() {
        for (input_rate, output_rate, input_frames) in [(8_000, 16_000, 160), (44_100, 48_000, 441)]
        {
            let input: Vec<f32> = (0..input_frames)
                .map(|i| ((i as f32) * 0.09).sin())
                .collect();
            let mut scalar = Resampler::<f32>::new(cfg(input_rate, output_rate, 1)).unwrap();
            let mut scalar_out = Vec::new();
            scalar.process(&input, &mut scalar_out).unwrap();
            scalar.finish(&mut scalar_out).unwrap();

            let mut rvv_cfg = cfg(input_rate, output_rate, 1);
            rvv_cfg.backend = Backend::Rvv;
            let mut rvv = Resampler::<f32>::new(rvv_cfg).unwrap();
            let mut rvv_out = Vec::new();
            rvv.process(&input, &mut rvv_out).unwrap();
            rvv.finish(&mut rvv_out).unwrap();

            assert_eq!(scalar_out.len(), rvv_out.len());
            for (a, b) in scalar_out.iter().zip(rvv_out.iter()) {
                assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn i16_scalar_and_rvv_are_close_for_special_and_general_ratios() {
        for (input_rate, output_rate, input_frames) in [(8_000, 16_000, 160), (44_100, 48_000, 441)]
        {
            let input: Vec<i16> = (0..input_frames)
                .map(|i| (((i as f32) * 0.09).sin() * 12_000.0) as i16)
                .collect();
            let mut scalar = Resampler::<i16>::new(cfg(input_rate, output_rate, 1)).unwrap();
            let mut scalar_out = Vec::new();
            scalar.process(&input, &mut scalar_out).unwrap();
            scalar.finish(&mut scalar_out).unwrap();

            let mut rvv_cfg = cfg(input_rate, output_rate, 1);
            rvv_cfg.backend = Backend::Rvv;
            let mut rvv = Resampler::<i16>::new(rvv_cfg).unwrap();
            let mut rvv_out = Vec::new();
            rvv.process(&input, &mut rvv_out).unwrap();
            rvv.finish(&mut rvv_out).unwrap();

            assert_eq!(scalar_out.len(), rvv_out.len());
            for (a, b) in scalar_out.iter().zip(rvv_out.iter()) {
                assert!((*a as i32 - *b as i32).abs() <= 1, "{a} != {b}");
            }
        }
    }

    #[test]
    fn reset_reuses_resampler_for_new_stream() {
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.03).sin()).collect();
        let mut first = Resampler::<f32>::new(cfg(44_100, 48_000, 1)).unwrap();
        let mut expected = Vec::new();
        first.process(&input, &mut expected).unwrap();
        first.finish(&mut expected).unwrap();

        let mut reusable = Resampler::<f32>::new(cfg(44_100, 48_000, 1)).unwrap();
        let mut ignored = Vec::new();
        reusable.process(&input[..128], &mut ignored).unwrap();
        reusable.reset();
        let mut actual = Vec::new();
        reusable.process(&input, &mut actual).unwrap();
        reusable.flush(&mut actual).unwrap();

        assert_eq!(expected.len(), actual.len());
        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
        }
    }

    #[test]
    fn process_into_slice_reports_capacity_and_writes_samples() {
        let input: Vec<i16> = (0..320)
            .map(|i| (((i as f32) * 0.07).sin() * 10_000.0) as i16)
            .collect();
        let mut resampler = Resampler::<i16>::new(cfg(16_000, 8_000, 1)).unwrap();
        let required = resampler.required_output_capacity(input.len());
        let mut too_small = vec![0i16; required.saturating_sub(1)];
        let err = resampler
            .process_into_slice(&input, &mut too_small)
            .unwrap_err();
        assert_eq!(
            err,
            Error::OutputTooSmall {
                required,
                available: required - 1
            }
        );

        let mut output = vec![0i16; required];
        let stats = resampler.process_into_slice(&input, &mut output).unwrap();
        assert_eq!(stats.input_frames, input.len());
        assert!(stats.output_frames > 0);
        assert!(
            output[..stats.output_frames]
                .iter()
                .any(|&sample| sample != 0)
        );
    }
}
