use crate::{Error, FirBackend, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Resampling quality preset.
///
/// Higher quality settings use more FIR taps and phases, trading CPU time and
/// memory for stronger filtering.
pub enum Quality {
    /// Lowest CPU cost with the shortest filters.
    Fast,
    /// Default quality/cost tradeoff for general use.
    #[default]
    Balanced,
    /// Highest quality preset with the longest filters.
    Best,
}

impl Quality {
    #[inline]
    pub(crate) const fn taps(self) -> usize {
        match self {
            Self::Fast => 24,
            Self::Balanced => 48,
            Self::Best => 96,
        }
    }

    #[inline]
    pub(crate) const fn phases(self) -> usize {
        match self {
            Self::Fast => 256,
            Self::Balanced => 512,
            Self::Best => 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Configuration used to construct a [`crate::Resampler`].
pub struct ResamplerConfig {
    /// Input sample rate in Hz. Must be greater than zero.
    pub input_rate: u32,
    /// Output sample rate in Hz. Must be greater than zero.
    pub output_rate: u32,
    /// Number of interleaved channels. Must be greater than zero.
    pub channels: usize,
    /// Filter quality preset.
    pub quality: Quality,
    /// FIR backend preference.
    pub backend: FirBackend,
    /// Optional expected maximum input chunk size, in frames.
    ///
    /// Setting this for predictable streaming packet sizes lets the resampler
    /// pre-size its internal history buffers.
    pub max_input_frames_per_chunk: Option<usize>,
}

impl ResamplerConfig {
    /// Creates a configuration with balanced quality, automatic backend
    /// selection, and no fixed chunk-size hint.
    #[inline]
    pub fn new(input_rate: u32, output_rate: u32, channels: usize) -> Self {
        Self {
            input_rate,
            output_rate,
            channels,
            quality: Quality::default(),
            backend: FirBackend::Auto,
            max_input_frames_per_chunk: None,
        }
    }

    pub(crate) fn validate(self) -> Result<()> {
        if self.input_rate == 0 || self.output_rate == 0 {
            return Err(Error::InvalidSampleRate);
        }
        if self.channels == 0 {
            return Err(Error::InvalidChannels);
        }
        Ok(())
    }
}
