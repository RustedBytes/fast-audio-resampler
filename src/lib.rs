//! Fast streaming audio resampling for x86 and AArch64 ARM CPUs.
//!
//! The crate uses a windowed-sinc polyphase FIR design with runtime CPU feature
//! dispatch. AVX/FMA and NEON kernels are isolated behind backend modules; the
//! scalar path remains the correctness reference and the portable fallback.

mod aligned;
mod backend;
mod config;
mod error;
mod filter;
mod resampler;
mod ring;

pub use backend::{Backend, SelectedBackend};
pub use config::{Quality, ResamplerConfig};
pub use error::{Error, Result};
pub use resampler::{ProcessStats, Resampler};
