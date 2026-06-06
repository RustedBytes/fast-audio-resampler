use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidSampleRate,
    InvalidChannels,
    InputNotFrameAligned { samples: usize, channels: usize },
    OutputTooSmall { required: usize, available: usize },
    UnsupportedBackend(&'static str),
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
