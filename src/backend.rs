#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Scalar,
    Avx2,
    Avx512,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedBackend {
    Scalar,
    Avx2,
    Avx512,
}

impl SelectedBackend {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2+fma",
            Self::Avx512 => "avx512f",
        }
    }
}

impl Backend {
    pub(crate) fn select(self) -> crate::Result<SelectedBackend> {
        match self {
            Self::Auto => Ok(auto_select()),
            Self::Scalar => Ok(SelectedBackend::Scalar),
            Self::Avx2 if avx2_available() => Ok(SelectedBackend::Avx2),
            Self::Avx512 if avx512_available() => Ok(SelectedBackend::Avx512),
            Self::Avx2 => Err(crate::Error::UnsupportedBackend("avx2+fma")),
            Self::Avx512 => Err(crate::Error::UnsupportedBackend("avx512f")),
        }
    }
}

#[inline]
pub(crate) fn dot_f32(backend: SelectedBackend, samples: &[f32], coeffs: &[f32]) -> f32 {
    debug_assert_eq!(samples.len(), coeffs.len());
    match backend {
        SelectedBackend::Scalar => scalar::dot_f32(samples, coeffs),
        SelectedBackend::Avx2 => x86::dot_f32_avx2(samples, coeffs),
        SelectedBackend::Avx512 => x86::dot_f32_avx512(samples, coeffs),
    }
}

#[inline]
pub(crate) fn dot_i16_q15(backend: SelectedBackend, samples: &[i16], coeffs: &[i16]) -> i16 {
    debug_assert_eq!(samples.len(), coeffs.len());
    let acc = match backend {
        SelectedBackend::Scalar => scalar::dot_i16_q15(samples, coeffs),
        SelectedBackend::Avx2 => x86::dot_i16_q15_avx2(samples, coeffs),
        SelectedBackend::Avx512 if avx2_available() => x86::dot_i16_q15_avx2(samples, coeffs),
        SelectedBackend::Avx512 => scalar::dot_i16_q15(samples, coeffs),
    };
    q15_acc_to_i16(acc)
}

#[inline(always)]
fn q15_acc_to_i16(acc: i64) -> i16 {
    let rounded = (acc + (1 << 14)) >> 15;
    rounded.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

#[inline]
fn auto_select() -> SelectedBackend {
    if avx512_available() {
        SelectedBackend::Avx512
    } else if avx2_available() {
        SelectedBackend::Avx2
    } else {
        SelectedBackend::Scalar
    }
}

#[inline]
fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn avx512_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

mod scalar {
    #[inline(always)]
    pub(crate) fn dot_f32(samples: &[f32], coeffs: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..samples.len() {
            // Avoid `mul_add` here: without FMA target features LLVM lowers it
            // to a libm `fmaf` call, which is much slower than scalar mul/add.
            unsafe {
                acc += *samples.get_unchecked(i) * *coeffs.get_unchecked(i);
            }
        }
        acc
    }

    #[inline(always)]
    pub(crate) fn dot_i16_q15(samples: &[i16], coeffs: &[i16]) -> i64 {
        let mut acc = 0i64;
        for i in 0..samples.len() {
            unsafe {
                acc += (*samples.get_unchecked(i) as i64) * (*coeffs.get_unchecked(i) as i64);
            }
        }
        acc
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    pub(crate) fn dot_f32_avx2(samples: &[f32], coeffs: &[f32]) -> f32 {
        unsafe { dot_f32_avx2_inner(samples, coeffs) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_f32_avx2_inner(samples: &[f32], coeffs: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let chunks = samples.len() / 8;
        for chunk in 0..chunks {
            let i = chunk * 8;
            let s = unsafe { _mm256_loadu_ps(samples.as_ptr().add(i)) };
            let c = unsafe { _mm256_loadu_ps(coeffs.as_ptr().add(i)) };
            acc = _mm256_fmadd_ps(s, c, acc);
        }
        let mut lanes = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut total = lanes.iter().copied().sum::<f32>();
        for i in chunks * 8..samples.len() {
            let sample = unsafe { *samples.get_unchecked(i) };
            let coeff = unsafe { *coeffs.get_unchecked(i) };
            total = sample.mul_add(coeff, total);
        }
        total
    }

    #[inline]
    pub(crate) fn dot_f32_avx512(samples: &[f32], coeffs: &[f32]) -> f32 {
        unsafe { dot_f32_avx512_inner(samples, coeffs) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn dot_f32_avx512_inner(samples: &[f32], coeffs: &[f32]) -> f32 {
        let mut acc = _mm512_setzero_ps();
        let chunks = samples.len() / 16;
        for chunk in 0..chunks {
            let i = chunk * 16;
            let s = unsafe { _mm512_loadu_ps(samples.as_ptr().add(i)) };
            let c = unsafe { _mm512_loadu_ps(coeffs.as_ptr().add(i)) };
            acc = _mm512_fmadd_ps(s, c, acc);
        }
        let mut lanes = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut total = lanes.iter().copied().sum::<f32>();
        for i in chunks * 16..samples.len() {
            let sample = unsafe { *samples.get_unchecked(i) };
            let coeff = unsafe { *coeffs.get_unchecked(i) };
            total = sample.mul_add(coeff, total);
        }
        total
    }

    #[inline]
    pub(crate) fn dot_i16_q15_avx2(samples: &[i16], coeffs: &[i16]) -> i64 {
        unsafe { dot_i16_q15_avx2_inner(samples, coeffs) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn dot_i16_q15_avx2_inner(samples: &[i16], coeffs: &[i16]) -> i64 {
        let mut acc = _mm256_setzero_si256();
        let chunks = samples.len() / 16;
        for chunk in 0..chunks {
            let i = chunk * 16;
            let s = unsafe { _mm256_loadu_si256(samples.as_ptr().add(i).cast::<__m256i>()) };
            let c = unsafe { _mm256_loadu_si256(coeffs.as_ptr().add(i).cast::<__m256i>()) };
            let products = _mm256_madd_epi16(s, c);
            acc = _mm256_add_epi32(acc, products);
        }

        let mut lanes = [0i32; 8];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc) };
        let mut total = lanes.iter().map(|&lane| lane as i64).sum::<i64>();
        for i in chunks * 16..samples.len() {
            let sample = unsafe { *samples.get_unchecked(i) as i64 };
            let coeff = unsafe { *coeffs.get_unchecked(i) as i64 };
            total += sample * coeff;
        }
        total
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    #[inline]
    pub(crate) fn dot_f32_avx2(samples: &[f32], coeffs: &[f32]) -> f32 {
        super::scalar::dot_f32(samples, coeffs)
    }

    #[inline]
    pub(crate) fn dot_f32_avx512(samples: &[f32], coeffs: &[f32]) -> f32 {
        super::scalar::dot_f32(samples, coeffs)
    }

    #[inline]
    pub(crate) fn dot_i16_q15_avx2(samples: &[i16], coeffs: &[i16]) -> i64 {
        super::scalar::dot_i16_q15(samples, coeffs)
    }
}
