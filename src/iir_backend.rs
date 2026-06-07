use crate::backend::SelectedFirBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectedIirBackend {
    Scalar,
    #[allow(dead_code)]
    Sse2,
    Neon,
    Rvv,
}

impl SelectedIirBackend {
    #[inline]
    pub(crate) fn auto_select() -> Self {
        if neon_available() {
            Self::Neon
        } else if rvv_available() {
            Self::Rvv
        } else {
            Self::Scalar
        }
    }

    #[inline]
    pub(crate) fn from_fir_backend(backend: SelectedFirBackend) -> Self {
        match backend {
            SelectedFirBackend::Scalar => Self::Scalar,
            SelectedFirBackend::Neon if neon_available() => Self::Neon,
            SelectedFirBackend::Rvv if rvv_available() => Self::Rvv,
            _ => Self::auto_select(),
        }
    }
}

#[inline(always)]
pub(crate) fn allpass_pair(
    backend: SelectedIirBackend,
    signal: [f32; 2],
    next_state: [f32; 2],
    coeffs: [f32; 2],
    state: [f32; 2],
) -> [f32; 2] {
    match backend {
        SelectedIirBackend::Scalar
        | SelectedIirBackend::Sse2
        | SelectedIirBackend::Neon
        | SelectedIirBackend::Rvv => scalar::allpass_pair(signal, next_state, coeffs, state),
    }
}

#[inline(always)]
pub(crate) fn allpass_pair_stereo(
    backend: SelectedIirBackend,
    signal: [[f32; 2]; 2],
    next_state: [[f32; 2]; 2],
    coeffs: [f32; 2],
    state: [[f32; 2]; 2],
) -> [[f32; 2]; 2] {
    match backend {
        SelectedIirBackend::Scalar => {
            scalar::allpass_pair_stereo(signal, next_state, coeffs, state)
        }
        SelectedIirBackend::Sse2 => {
            x86::allpass_pair_stereo_sse2(signal, next_state, coeffs, state)
        }
        SelectedIirBackend::Neon => {
            aarch64::allpass_pair_stereo_neon(signal, next_state, coeffs, state)
        }
        SelectedIirBackend::Rvv => {
            riscv64::allpass_pair_stereo_rvv(signal, next_state, coeffs, state)
        }
    }
}

#[inline]
#[allow(dead_code)]
fn sse2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            true
        }
        #[cfg(target_arch = "x86")]
        {
            std::is_x86_feature_detected!("sse2")
        }
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
    pub(crate) fn allpass_pair(
        signal: [f32; 2],
        next_state: [f32; 2],
        coeffs: [f32; 2],
        state: [f32; 2],
    ) -> [f32; 2] {
        [
            (signal[0] - next_state[0]) * coeffs[0] + state[0],
            (signal[1] - next_state[1]) * coeffs[1] + state[1],
        ]
    }

    #[inline(always)]
    pub(crate) fn allpass_pair_stereo(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        [
            allpass_stereo_lane(signal[0], next_state[0], coeffs[0], state[0]),
            allpass_stereo_lane(signal[1], next_state[1], coeffs[1], state[1]),
        ]
    }

    #[inline(always)]
    fn allpass_stereo_lane(
        signal: [f32; 2],
        next_state: [f32; 2],
        coeff: f32,
        state: [f32; 2],
    ) -> [f32; 2] {
        [
            (signal[0] - next_state[0]) * coeff + state[0],
            (signal[1] - next_state[1]) * coeff + state[1],
        ]
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    pub(crate) fn allpass_pair_stereo_sse2(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        unsafe { allpass_pair_stereo_sse2_inner(signal, next_state, coeffs, state) }
    }

    #[target_feature(enable = "sse,sse2")]
    unsafe fn allpass_pair_stereo_sse2_inner(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        [
            unsafe { allpass_lane(signal[0], next_state[0], coeffs[0], state[0]) },
            unsafe { allpass_lane(signal[1], next_state[1], coeffs[1], state[1]) },
        ]
    }

    #[target_feature(enable = "sse,sse2")]
    unsafe fn allpass_lane(
        signal: [f32; 2],
        next_state: [f32; 2],
        coeff: f32,
        state: [f32; 2],
    ) -> [f32; 2] {
        let signal = _mm_setr_ps(signal[0], signal[1], 0.0, 0.0);
        let next_state = _mm_setr_ps(next_state[0], next_state[1], 0.0, 0.0);
        let coeff = _mm_set1_ps(coeff);
        let state = _mm_setr_ps(state[0], state[1], 0.0, 0.0);
        let out = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(signal, next_state), coeff), state);
        let mut lanes = [0.0f32; 4];
        unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), out) };
        [lanes[0], lanes[1]]
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    #[inline]
    pub(crate) fn allpass_pair_stereo_sse2(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        super::scalar::allpass_pair_stereo(signal, next_state, coeffs, state)
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    #[inline]
    pub(crate) fn allpass_pair_stereo_neon(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        unsafe { allpass_pair_stereo_neon_inner(signal, next_state, coeffs, state) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn allpass_pair_stereo_neon_inner(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        [
            allpass_lane(signal[0], next_state[0], coeffs[0], state[0]),
            allpass_lane(signal[1], next_state[1], coeffs[1], state[1]),
        ]
    }

    #[inline(always)]
    fn allpass_lane(
        signal: [f32; 2],
        next_state: [f32; 2],
        coeff: f32,
        state: [f32; 2],
    ) -> [f32; 2] {
        let signal = unsafe { vld1_f32(signal.as_ptr()) };
        let next_state = unsafe { vld1_f32(next_state.as_ptr()) };
        let state = unsafe { vld1_f32(state.as_ptr()) };
        let out = vmla_f32(state, vsub_f32(signal, next_state), vdup_n_f32(coeff));
        let mut lanes = [0.0f32; 2];
        unsafe { vst1_f32(lanes.as_mut_ptr(), out) };
        lanes
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod aarch64 {
    #[inline]
    pub(crate) fn allpass_pair_stereo_neon(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        super::scalar::allpass_pair_stereo(signal, next_state, coeffs, state)
    }
}

#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
mod riscv64 {
    use core::arch::asm;

    #[inline]
    pub(crate) fn allpass_pair_stereo_rvv(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        [
            unsafe { allpass_lane(signal[0], next_state[0], coeffs[0], state[0]) },
            unsafe { allpass_lane(signal[1], next_state[1], coeffs[1], state[1]) },
        ]
    }

    #[inline]
    unsafe fn allpass_lane(
        signal: [f32; 2],
        next_state: [f32; 2],
        coeff: f32,
        state: [f32; 2],
    ) -> [f32; 2] {
        let mut out = [0.0f32; 2];
        unsafe {
            asm!(
                "vsetivli zero, 2, e32, m1, ta, ma",
                "vle32.v v8, ({signal_ptr})",
                "vle32.v v9, ({next_state_ptr})",
                "vfsub.vv v8, v8, v9",
                "vfmul.vf v8, v8, {coeff}",
                "vle32.v v9, ({state_ptr})",
                "vfadd.vv v8, v8, v9",
                "vse32.v v8, ({out_ptr})",
                signal_ptr = in(reg) signal.as_ptr(),
                next_state_ptr = in(reg) next_state.as_ptr(),
                state_ptr = in(reg) state.as_ptr(),
                out_ptr = in(reg) out.as_mut_ptr(),
                coeff = in(freg) coeff,
                out("v8") _,
                out("v9") _,
            );
        }
        out
    }
}

#[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
mod riscv64 {
    #[inline]
    pub(crate) fn allpass_pair_stereo_rvv(
        signal: [[f32; 2]; 2],
        next_state: [[f32; 2]; 2],
        coeffs: [f32; 2],
        state: [[f32; 2]; 2],
    ) -> [[f32; 2]; 2] {
        super::scalar::allpass_pair_stereo(signal, next_state, coeffs, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs() -> ([[f32; 2]; 2], [[f32; 2]; 2], [f32; 2], [[f32; 2]; 2]) {
        (
            [[0.25, -0.5], [0.75, 0.125]],
            [[-0.125, 0.375], [0.5, -0.25]],
            [0.29505825, 0.7137337],
            [[0.01, -0.02], [0.03, -0.04]],
        )
    }

    #[test]
    fn stereo_auto_iir_backend_matches_scalar() {
        let (signal, next_state, coeffs, state) = sample_inputs();
        let scalar = allpass_pair_stereo(
            SelectedIirBackend::Scalar,
            signal,
            next_state,
            coeffs,
            state,
        );
        let auto = allpass_pair_stereo(
            SelectedIirBackend::auto_select(),
            signal,
            next_state,
            coeffs,
            state,
        );
        assert_eq!(scalar, auto);
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn stereo_sse2_iir_backend_matches_scalar() {
        let (signal, next_state, coeffs, state) = sample_inputs();
        let scalar = allpass_pair_stereo(
            SelectedIirBackend::Scalar,
            signal,
            next_state,
            coeffs,
            state,
        );
        let sse2 = allpass_pair_stereo(SelectedIirBackend::Sse2, signal, next_state, coeffs, state);
        assert_eq!(scalar, sse2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn stereo_neon_iir_backend_matches_scalar() {
        let (signal, next_state, coeffs, state) = sample_inputs();
        let scalar = allpass_pair_stereo(
            SelectedIirBackend::Scalar,
            signal,
            next_state,
            coeffs,
            state,
        );
        let neon = allpass_pair_stereo(SelectedIirBackend::Neon, signal, next_state, coeffs, state);
        assert_eq!(scalar, neon);
    }

    #[test]
    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    fn stereo_rvv_iir_backend_matches_scalar() {
        let (signal, next_state, coeffs, state) = sample_inputs();
        let scalar = allpass_pair_stereo(
            SelectedIirBackend::Scalar,
            signal,
            next_state,
            coeffs,
            state,
        );
        let rvv = allpass_pair_stereo(SelectedIirBackend::Rvv, signal, next_state, coeffs, state);
        assert_eq!(scalar, rvv);
    }
}
