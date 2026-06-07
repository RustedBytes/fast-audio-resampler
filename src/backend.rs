#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Rvv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedBackend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Rvv,
}

impl SelectedBackend {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2+fma",
            Self::Avx512 => "avx512f",
            Self::Neon => "neon",
            Self::Rvv => "rvv",
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
            Self::Neon if neon_available() => Ok(SelectedBackend::Neon),
            Self::Rvv if rvv_available() => Ok(SelectedBackend::Rvv),
            Self::Avx2 => Err(crate::Error::UnsupportedBackend("avx2+fma")),
            Self::Avx512 => Err(crate::Error::UnsupportedBackend("avx512f")),
            Self::Neon => Err(crate::Error::UnsupportedBackend("neon")),
            Self::Rvv => Err(crate::Error::UnsupportedBackend("rvv")),
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
        SelectedBackend::Neon => aarch64::dot_f32_neon(samples, coeffs),
        SelectedBackend::Rvv => riscv64::dot_f32_rvv(samples, coeffs),
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
        SelectedBackend::Neon => aarch64::dot_i16_q15_neon(samples, coeffs),
        SelectedBackend::Rvv => riscv64::dot_i16_q15_rvv(samples, coeffs),
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
    } else if neon_available() {
        SelectedBackend::Neon
    } else if rvv_available() {
        SelectedBackend::Rvv
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

#[inline]
fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[inline]
fn rvv_available() -> bool {
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        true
    }
    #[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
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

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    #[inline]
    pub(crate) fn dot_f32_neon(samples: &[f32], coeffs: &[f32]) -> f32 {
        unsafe { dot_f32_neon_inner(samples, coeffs) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn dot_f32_neon_inner(samples: &[f32], coeffs: &[f32]) -> f32 {
        let mut acc = vdupq_n_f32(0.0);
        let chunks = samples.len() / 4;
        for chunk in 0..chunks {
            let i = chunk * 4;
            let s = unsafe { vld1q_f32(samples.as_ptr().add(i)) };
            let c = unsafe { vld1q_f32(coeffs.as_ptr().add(i)) };
            acc = vfmaq_f32(acc, s, c);
        }

        let mut total = vaddvq_f32(acc);
        for i in chunks * 4..samples.len() {
            let sample = unsafe { *samples.get_unchecked(i) };
            let coeff = unsafe { *coeffs.get_unchecked(i) };
            total = sample.mul_add(coeff, total);
        }
        total
    }

    #[inline]
    pub(crate) fn dot_i16_q15_neon(samples: &[i16], coeffs: &[i16]) -> i64 {
        unsafe { dot_i16_q15_neon_inner(samples, coeffs) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn dot_i16_q15_neon_inner(samples: &[i16], coeffs: &[i16]) -> i64 {
        let mut acc = vdupq_n_s64(0);
        let chunks = samples.len() / 8;
        for chunk in 0..chunks {
            let i = chunk * 8;
            let s = unsafe { vld1q_s16(samples.as_ptr().add(i)) };
            let c = unsafe { vld1q_s16(coeffs.as_ptr().add(i)) };
            let products_low = vmull_s16(vget_low_s16(s), vget_low_s16(c));
            let products_high = vmull_s16(vget_high_s16(s), vget_high_s16(c));
            acc = vaddq_s64(acc, vpaddlq_s32(products_low));
            acc = vaddq_s64(acc, vpaddlq_s32(products_high));
        }

        let mut total = vaddvq_s64(acc);
        for i in chunks * 8..samples.len() {
            let sample = unsafe { *samples.get_unchecked(i) as i64 };
            let coeff = unsafe { *coeffs.get_unchecked(i) as i64 };
            total += sample * coeff;
        }
        total
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod aarch64 {
    #[inline]
    pub(crate) fn dot_f32_neon(samples: &[f32], coeffs: &[f32]) -> f32 {
        super::scalar::dot_f32(samples, coeffs)
    }

    #[inline]
    pub(crate) fn dot_i16_q15_neon(samples: &[i16], coeffs: &[i16]) -> i64 {
        super::scalar::dot_i16_q15(samples, coeffs)
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

#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
mod riscv64 {
    use core::arch::asm;

    #[inline]
    pub(crate) fn dot_f32_rvv(samples: &[f32], coeffs: &[f32]) -> f32 {
        unsafe { dot_f32_rvv_inner(samples, coeffs) }
    }

    unsafe fn dot_f32_rvv_inner(samples: &[f32], coeffs: &[f32]) -> f32 {
        let mut samples_ptr = samples.as_ptr();
        let mut coeffs_ptr = coeffs.as_ptr();
        let mut remaining = samples.len();
        let mut total: f32;

        unsafe {
            asm!(
                "vsetvli t0, zero, e32, m1, ta, ma",
                "vmv.v.i v0, 0",
                "2:",
                "beqz {remaining}, 3f",
                "vsetvli t0, {remaining}, e32, m1, tu, ma",
                "vle32.v v8, ({samples_ptr})",
                "vle32.v v9, ({coeffs_ptr})",
                "vfmacc.vv v0, v8, v9",
                "slli t1, t0, 2",
                "add {samples_ptr}, {samples_ptr}, t1",
                "add {coeffs_ptr}, {coeffs_ptr}, t1",
                "sub {remaining}, {remaining}, t0",
                "j 2b",
                "3:",
                "vsetvli t0, zero, e32, m1, ta, ma",
                "vmv.v.i v10, 0",
                "vfredusum.vs v11, v0, v10",
                "vfmv.f.s {total}, v11",
                samples_ptr = inout(reg) samples_ptr,
                coeffs_ptr = inout(reg) coeffs_ptr,
                remaining = inout(reg) remaining,
                total = lateout(freg) total,
                out("t0") _,
                out("t1") _,
                out("v0") _,
                out("v8") _,
                out("v9") _,
                out("v10") _,
                out("v11") _,
            );
        }

        total
    }

    #[inline]
    pub(crate) fn dot_i16_q15_rvv(samples: &[i16], coeffs: &[i16]) -> i64 {
        if samples.len() > 128 {
            return super::scalar::dot_i16_q15(samples, coeffs);
        }
        unsafe { dot_i16_q15_rvv_inner(samples, coeffs) }
    }

    unsafe fn dot_i16_q15_rvv_inner(samples: &[i16], coeffs: &[i16]) -> i64 {
        let mut products = [0i32; 128];
        let mut samples_ptr = samples.as_ptr();
        let mut coeffs_ptr = coeffs.as_ptr();
        let mut remaining = samples.len();
        let mut total = 0i64;

        while remaining > 0 {
            let mut vl = remaining;
            unsafe {
                asm!(
                    "vsetvli {vl}, {vl}, e16, m1, ta, ma",
                    "vle16.v v8, ({samples_ptr})",
                    "vle16.v v9, ({coeffs_ptr})",
                    "vsetvli t0, {vl}, e32, m2, ta, ma",
                    "vmv.v.i v0, 0",
                    "vsetvli zero, {vl}, e16, m1, ta, ma",
                    "vwmacc.vv v0, v8, v9",
                    "vsetvli zero, {vl}, e32, m2, ta, ma",
                    "vse32.v v0, ({products_ptr})",
                    vl = inout(reg) vl,
                    samples_ptr = in(reg) samples_ptr,
                    coeffs_ptr = in(reg) coeffs_ptr,
                    products_ptr = in(reg) products.as_mut_ptr(),
                    out("t0") _,
                    out("v0") _,
                    out("v1") _,
                    out("v8") _,
                    out("v9") _,
                );
            }

            for &product in &products[..vl] {
                total += product as i64;
            }
            samples_ptr = unsafe { samples_ptr.add(vl) };
            coeffs_ptr = unsafe { coeffs_ptr.add(vl) };
            remaining -= vl;
        }

        total
    }
}

#[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
mod riscv64 {
    #[inline]
    pub(crate) fn dot_f32_rvv(samples: &[f32], coeffs: &[f32]) -> f32 {
        super::scalar::dot_f32(samples, coeffs)
    }

    #[inline]
    pub(crate) fn dot_i16_q15_rvv(samples: &[i16], coeffs: &[i16]) -> i64 {
        super::scalar::dot_i16_q15(samples, coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn explicit_neon_is_unsupported_off_aarch64() {
        assert_eq!(
            Backend::Neon.select(),
            Err(Error::UnsupportedBackend("neon"))
        );
    }

    #[test]
    #[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
    fn explicit_rvv_is_unsupported_without_rvv_target_feature() {
        assert_eq!(Backend::Rvv.select(), Err(Error::UnsupportedBackend("rvv")));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn auto_selects_neon_on_aarch64() {
        assert_eq!(Backend::Auto.select().unwrap(), SelectedBackend::Neon);
        assert_eq!(Backend::Neon.select().unwrap(), SelectedBackend::Neon);
    }

    #[test]
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn auto_selects_rvv_on_riscv64_with_v() {
        assert_eq!(Backend::Auto.select().unwrap(), SelectedBackend::Rvv);
        assert_eq!(Backend::Rvv.select().unwrap(), SelectedBackend::Rvv);
    }
}
