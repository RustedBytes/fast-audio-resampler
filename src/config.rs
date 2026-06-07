use crate::{Error, FirBackend, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Quality {
    Fast,
    #[default]
    Balanced,
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
pub struct ResamplerConfig {
    pub input_rate: u32,
    pub output_rate: u32,
    pub channels: usize,
    pub quality: Quality,
    pub backend: FirBackend,
    pub max_input_frames_per_chunk: Option<usize>,
}

impl ResamplerConfig {
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
