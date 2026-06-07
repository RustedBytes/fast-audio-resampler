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
}

#[derive(Debug, Clone)]
pub(crate) struct Iir2xFilter {
    coeffs: Vec<[f32; 2]>,
    states: Vec<[f32; 2]>,
    last_state: [f32; 2],
}

impl PolyphaseIir2x {
    pub(crate) fn up(channels: usize) -> Self {
        Self::Up((0..channels).map(|_| Iir2xFilter::fast()).collect())
    }

    pub(crate) fn down(channels: usize) -> Self {
        Self::Down((0..channels).map(|_| Iir2xFilter::fast()).collect())
    }

    #[inline]
    pub(crate) fn is_up(&self) -> bool {
        matches!(self, Self::Up(_))
    }

    #[inline]
    pub(crate) fn process_up(&mut self, channel: usize, sample: f32) -> [f32; 2] {
        match self {
            Self::Up(filters) => filters[channel].process_pair(sample, sample),
            Self::Down(_) => unreachable!("downsampling IIR cannot process upsampling samples"),
        }
    }

    #[inline]
    pub(crate) fn process_down(&mut self, channel: usize, even: f32, odd: f32) -> f32 {
        match self {
            Self::Down(filters) => {
                let [a, b] = filters[channel].process_pair(even, odd);
                0.5 * (a + b)
            }
            Self::Up(_) => unreachable!("upsampling IIR cannot process downsampling pairs"),
        }
    }
}

impl Iir2xFilter {
    fn fast() -> Self {
        Self::new(&FAST_IIR_COEFFS)
    }

    fn new(coeff_arr: &[f32]) -> Self {
        debug_assert!(coeff_arr.len().is_multiple_of(2));
        let coeffs: Vec<[f32; 2]> = coeff_arr
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();
        let states = vec![[0.0; 2]; coeffs.len()];
        Self {
            coeffs,
            states,
            last_state: [0.0; 2],
        }
    }

    #[inline]
    fn process_pair(&mut self, s0: f32, s1: f32) -> [f32; 2] {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyphase_iir_filters_silence_to_silence() {
        let mut up = PolyphaseIir2x::up(1);
        for _ in 0..16 {
            assert_eq!(up.process_up(0, 0.0), [0.0, 0.0]);
        }

        let mut down = PolyphaseIir2x::down(1);
        for _ in 0..16 {
            assert_eq!(down.process_down(0, 0.0, 0.0), 0.0);
        }
    }

    #[test]
    fn polyphase_iir_down_filters_alternating_high_frequency() {
        let mut down = PolyphaseIir2x::down(1);
        let output: Vec<f32> = (0..64).map(|_| down.process_down(0, 1.0, -1.0)).collect();
        let tail_mean = output[32..].iter().map(|sample| sample.abs()).sum::<f32>() / 32.0;
        assert!(tail_mean < 0.1, "tail mean {tail_mean}");
    }
}
