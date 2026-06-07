#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectedIirBackend {
    Scalar,
}

impl SelectedIirBackend {
    #[inline]
    pub(crate) fn auto_select() -> Self {
        Self::Scalar
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
        SelectedIirBackend::Scalar => scalar::allpass_pair(signal, next_state, coeffs, state),
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
