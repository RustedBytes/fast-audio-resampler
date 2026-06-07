use crate::iir_backend::{self, SelectedIirBackend};

const FAST_IIR_COEFFS: [f32; 6] = [
    0.086928910174,
    0.295058247590,
    0.524893965468,
    0.713733698049,
    0.850801377617,
    0.953334485124,
];

#[derive(Debug, Clone)]
pub(crate) enum PolyphaseIir2x {
    Up(Vec<Iir2xFilter>),
    Down(Vec<Iir2xFilter>),
    UpStereo(Iir2xStereoFilter),
    DownStereo(Iir2xStereoFilter),
}

#[derive(Debug, Clone)]
pub(crate) struct Iir2xFilter {
    backend: SelectedIirBackend,
    coeffs: Vec<[f32; 2]>,
    states: Vec<[f32; 2]>,
    last_state: [f32; 2],
}

#[derive(Debug, Clone)]
pub(crate) struct Iir2xStereoFilter {
    backend: SelectedIirBackend,
    coeffs: Vec<[f32; 2]>,
    states: Vec<[[f32; 2]; 2]>,
    last_state: [[f32; 2]; 2],
}

impl PolyphaseIir2x {
    pub(crate) fn up(channels: usize, backend: SelectedIirBackend) -> Self {
        if channels == 2 {
            Self::UpStereo(Iir2xStereoFilter::fast(backend))
        } else {
            Self::Up((0..channels).map(|_| Iir2xFilter::fast(backend)).collect())
        }
    }

    pub(crate) fn down(channels: usize, backend: SelectedIirBackend) -> Self {
        if channels == 2 {
            Self::DownStereo(Iir2xStereoFilter::fast(backend))
        } else {
            Self::Down((0..channels).map(|_| Iir2xFilter::fast(backend)).collect())
        }
    }

    #[inline]
    pub(crate) fn is_up(&self) -> bool {
        matches!(self, Self::Up(_) | Self::UpStereo(_))
    }

    #[inline]
    pub(crate) fn is_stereo(&self) -> bool {
        matches!(self, Self::UpStereo(_) | Self::DownStereo(_))
    }

    #[inline]
    pub(crate) fn process_up(&mut self, channel: usize, sample: f32) -> [f32; 2] {
        match self {
            Self::Up(filters) => filters[channel].process_pair(sample, sample),
            Self::UpStereo(_) => unreachable!("stereo IIR must process both channels together"),
            Self::Down(_) => unreachable!("downsampling IIR cannot process upsampling samples"),
            Self::DownStereo(_) => {
                unreachable!("downsampling IIR cannot process upsampling samples")
            }
        }
    }

    #[inline]
    pub(crate) fn process_up_stereo(&mut self, left: f32, right: f32) -> [[f32; 2]; 2] {
        match self {
            Self::UpStereo(filter) => {
                let [even, odd] = filter.process_pair([left, right], [left, right]);
                [[even[0], even[1]], [odd[0], odd[1]]]
            }
            Self::Up(_) => unreachable!("mono IIR cannot process stereo lanes together"),
            Self::Down(_) | Self::DownStereo(_) => {
                unreachable!("downsampling IIR cannot process upsampling samples")
            }
        }
    }

    #[inline]
    pub(crate) fn process_down(&mut self, channel: usize, even: f32, odd: f32) -> f32 {
        match self {
            Self::Down(filters) => {
                let [a, b] = filters[channel].process_pair(even, odd);
                0.5 * (a + b)
            }
            Self::DownStereo(_) => unreachable!("stereo IIR must process both channels together"),
            Self::Up(_) => unreachable!("upsampling IIR cannot process downsampling pairs"),
            Self::UpStereo(_) => unreachable!("upsampling IIR cannot process downsampling pairs"),
        }
    }

    #[inline]
    pub(crate) fn process_down_stereo(
        &mut self,
        even_left: f32,
        even_right: f32,
        odd_left: f32,
        odd_right: f32,
    ) -> [f32; 2] {
        match self {
            Self::DownStereo(filter) => {
                let [a, b] = filter.process_pair([even_left, even_right], [odd_left, odd_right]);
                [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
            }
            Self::Down(_) => unreachable!("mono IIR cannot process stereo lanes together"),
            Self::Up(_) | Self::UpStereo(_) => {
                unreachable!("upsampling IIR cannot process downsampling pairs")
            }
        }
    }
}

impl Iir2xFilter {
    fn fast(backend: SelectedIirBackend) -> Self {
        Self::new(&FAST_IIR_COEFFS, backend)
    }

    fn new(coeff_arr: &[f32], backend: SelectedIirBackend) -> Self {
        debug_assert!(coeff_arr.len().is_multiple_of(2));
        let coeffs: Vec<[f32; 2]> = coeff_arr
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();
        let states = vec![[0.0; 2]; coeffs.len()];
        Self {
            backend,
            coeffs,
            states,
            last_state: [0.0; 2],
        }
    }

    #[inline]
    fn process_pair(&mut self, s0: f32, s1: f32) -> [f32; 2] {
        if self.backend == SelectedIirBackend::Scalar {
            return self.process_pair_scalar(s0, s1);
        }

        let mut signal = [s1, s0];
        let last = self.coeffs.len() - 1;

        for i in 0..last {
            let next_state = self.states[i + 1];
            let tmp = iir_backend::allpass_pair(
                self.backend,
                signal,
                next_state,
                self.coeffs[i],
                self.states[i],
            );
            self.states[i] = signal;
            signal = tmp;
        }

        let tmp = iir_backend::allpass_pair(
            self.backend,
            signal,
            self.last_state,
            self.coeffs[last],
            self.states[last],
        );
        self.states[last] = signal;
        self.last_state = tmp;
        tmp
    }

    #[inline]
    fn process_pair_scalar(&mut self, s0: f32, s1: f32) -> [f32; 2] {
        let mut signal = [s1, s0];
        let last = self.coeffs.len() - 1;

        for i in 0..last {
            let next_state = self.states[i + 1];
            let tmp = [
                (signal[0] - next_state[0]) * self.coeffs[i][0] + self.states[i][0],
                (signal[1] - next_state[1]) * self.coeffs[i][1] + self.states[i][1],
            ];
            self.states[i] = signal;
            signal = tmp;
        }

        let tmp = [
            (signal[0] - self.last_state[0]) * self.coeffs[last][0] + self.states[last][0],
            (signal[1] - self.last_state[1]) * self.coeffs[last][1] + self.states[last][1],
        ];
        self.states[last] = signal;
        self.last_state = tmp;
        tmp
    }
}

impl Iir2xStereoFilter {
    fn fast(backend: SelectedIirBackend) -> Self {
        Self::new(&FAST_IIR_COEFFS, backend)
    }

    fn new(coeff_arr: &[f32], backend: SelectedIirBackend) -> Self {
        debug_assert!(coeff_arr.len().is_multiple_of(2));
        let coeffs: Vec<[f32; 2]> = coeff_arr
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();
        let states = vec![[[0.0; 2]; 2]; coeffs.len()];
        Self {
            backend,
            coeffs,
            states,
            last_state: [[0.0; 2]; 2],
        }
    }

    #[inline]
    fn process_pair(&mut self, s0: [f32; 2], s1: [f32; 2]) -> [[f32; 2]; 2] {
        if self.backend == SelectedIirBackend::Scalar {
            return self.process_pair_scalar(s0, s1);
        }

        let mut signal = [s1, s0];
        let last = self.coeffs.len() - 1;

        for i in 0..last {
            let next_state = self.states[i + 1];
            let tmp = iir_backend::allpass_pair_stereo(
                self.backend,
                signal,
                next_state,
                self.coeffs[i],
                self.states[i],
            );
            self.states[i] = signal;
            signal = tmp;
        }

        let tmp = iir_backend::allpass_pair_stereo(
            self.backend,
            signal,
            self.last_state,
            self.coeffs[last],
            self.states[last],
        );
        self.states[last] = signal;
        self.last_state = tmp;
        tmp
    }

    #[inline]
    fn process_pair_scalar(&mut self, s0: [f32; 2], s1: [f32; 2]) -> [[f32; 2]; 2] {
        let mut signal = [s1, s0];
        let last = self.coeffs.len() - 1;

        for i in 0..last {
            let next_state = self.states[i + 1];
            let tmp = [
                [
                    (signal[0][0] - next_state[0][0]) * self.coeffs[i][0] + self.states[i][0][0],
                    (signal[0][1] - next_state[0][1]) * self.coeffs[i][0] + self.states[i][0][1],
                ],
                [
                    (signal[1][0] - next_state[1][0]) * self.coeffs[i][1] + self.states[i][1][0],
                    (signal[1][1] - next_state[1][1]) * self.coeffs[i][1] + self.states[i][1][1],
                ],
            ];
            self.states[i] = signal;
            signal = tmp;
        }

        let tmp = [
            [
                (signal[0][0] - self.last_state[0][0]) * self.coeffs[last][0]
                    + self.states[last][0][0],
                (signal[0][1] - self.last_state[0][1]) * self.coeffs[last][0]
                    + self.states[last][0][1],
            ],
            [
                (signal[1][0] - self.last_state[1][0]) * self.coeffs[last][1]
                    + self.states[last][1][0],
                (signal[1][1] - self.last_state[1][1]) * self.coeffs[last][1]
                    + self.states[last][1][1],
            ],
        ];
        self.states[last] = signal;
        self.last_state = tmp;
        tmp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyphase_iir_filters_silence_to_silence() {
        let mut up = PolyphaseIir2x::up(1, SelectedIirBackend::Scalar);
        for _ in 0..16 {
            assert_eq!(up.process_up(0, 0.0), [0.0, 0.0]);
        }

        let mut down = PolyphaseIir2x::down(1, SelectedIirBackend::Scalar);
        for _ in 0..16 {
            assert_eq!(down.process_down(0, 0.0, 0.0), 0.0);
        }
    }

    #[test]
    fn polyphase_iir_down_filters_alternating_high_frequency() {
        let mut down = PolyphaseIir2x::down(1, SelectedIirBackend::Scalar);
        let output: Vec<f32> = (0..64).map(|_| down.process_down(0, 1.0, -1.0)).collect();
        let tail_mean = output[32..].iter().map(|sample| sample.abs()).sum::<f32>() / 32.0;
        assert!(tail_mean < 0.1, "tail mean {tail_mean}");
    }

    #[test]
    fn stereo_iir_matches_two_independent_mono_upsamplers() {
        let mut stereo = PolyphaseIir2x::up(2, SelectedIirBackend::auto_select());
        assert!(stereo.is_stereo());
        let mut left = PolyphaseIir2x::up(1, SelectedIirBackend::Scalar);
        let mut right = PolyphaseIir2x::up(1, SelectedIirBackend::Scalar);

        for i in 0..64 {
            let l = ((i as f32) * 0.03).sin();
            let r = ((i as f32) * 0.05).cos();
            let stereo_pair = stereo.process_up_stereo(l, r);
            let left_pair = left.process_up(0, l);
            let right_pair = right.process_up(0, r);
            assert_eq!(stereo_pair[0], [left_pair[0], right_pair[0]]);
            assert_eq!(stereo_pair[1], [left_pair[1], right_pair[1]]);
        }
    }

    #[test]
    fn stereo_iir_matches_two_independent_mono_downsamplers() {
        let mut stereo = PolyphaseIir2x::down(2, SelectedIirBackend::auto_select());
        assert!(stereo.is_stereo());
        let mut left = PolyphaseIir2x::down(1, SelectedIirBackend::Scalar);
        let mut right = PolyphaseIir2x::down(1, SelectedIirBackend::Scalar);

        for i in 0..64 {
            let even_l = ((i as f32) * 0.03).sin();
            let odd_l = ((i as f32) * 0.07).sin();
            let even_r = ((i as f32) * 0.05).cos();
            let odd_r = ((i as f32) * 0.11).cos();
            let stereo_sample = stereo.process_down_stereo(even_l, even_r, odd_l, odd_r);
            let left_sample = left.process_down(0, even_l, odd_l);
            let right_sample = right.process_down(0, even_r, odd_r);
            assert_eq!(stereo_sample, [left_sample, right_sample]);
        }
    }
}
