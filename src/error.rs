use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error returned by resampler construction and processing operations.
pub enum Error {
    /// Input or output sample rate was zero.
    InvalidSampleRate,
    /// Channel count was zero.
    InvalidChannels,
    /// Input sample count was not divisible by the configured channel count.
    InputNotFrameAligned { samples: usize, channels: usize },
    /// Fixed output slice did not have enough sample capacity.
    OutputTooSmall { required: usize, available: usize },
    /// Requested backend is unavailable on the current CPU or build target.
    UnsupportedBackend(&'static str),
    /// Command-line interface error.
    Cli(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleRate => write!(f, "sample rates must be greater than zero"),
            Self::InvalidChannels => write!(f, "channel count must be greater than zero"),
            Self::InputNotFrameAligned { samples, channels } => write!(
                f,
                "input has {samples} samples, which is not divisible by {channels} channels"
            ),
            Self::OutputTooSmall {
                required,
                available,
            } => write!(
                f,
                "output buffer has {available} samples, but {required} samples are required"
            ),
            Self::UnsupportedBackend(backend) => {
                write!(f, "requested backend is not available: {backend}")
            }
            Self::Cli(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

#[cold]
pub(crate) fn frame_alignment_error(samples: usize, channels: usize) -> Error {
    Error::InputNotFrameAligned { samples, channels }
}
