//! Fast streaming audio resampling for Intel and AMD x86 CPUs.
//!
//! The crate uses a windowed-sinc polyphase FIR design with runtime CPU feature
//! dispatch. AVX/FMA kernels are isolated behind backend modules; the scalar
//! path remains the correctness reference and the portable fallback.

mod backend;
mod config;
mod error;
mod filter;
mod resampler;

pub use backend::{Backend, SelectedBackend};
pub use config::{Quality, ResamplerConfig};
pub use error::{Error, Result};
pub use resampler::{ProcessStats, Resampler};
