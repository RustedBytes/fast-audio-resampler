//! Fast streaming audio resampling for x86, AArch64 ARM, and RISC-V CPUs.
//!
//! The crate uses a windowed-sinc polyphase FIR design with runtime CPU feature
//! dispatch where stable detection is available. AVX/FMA, NEON, and RVV kernels
//! are isolated behind backend modules; the scalar path remains the correctness
//! reference and the portable fallback.

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
