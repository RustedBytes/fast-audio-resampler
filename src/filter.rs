use std::f64::consts::PI;

use crate::Quality;
use crate::aligned::AlignedVec;

#[derive(Debug, Clone)]
pub(crate) struct FilterBank {
    taps: usize,
    phases: usize,
    coeffs: AlignedVec<f32, 64>,
    coeffs_q15: AlignedVec<i16, 64>,
    half_band: Option<HalfBandFilter>,
    third_band: Option<ThirdBandFilter>,
}

#[derive(Debug, Clone)]
pub(crate) struct HalfBandFilter {
    half_taps: usize,
    side_offsets: Vec<i64>,
    side_input_offsets_for_up: Vec<i64>,
    side_coeffs: AlignedVec<f32, 64>,
    side_coeffs_up: AlignedVec<f32, 64>,
    side_coeffs_q15: AlignedVec<i16, 64>,
    side_coeffs_up_q15: AlignedVec<i16, 64>,
    center_coeff: f32,
    center_coeff_q15: i16,
}

#[derive(Debug, Clone)]
pub(crate) struct ThirdBandFilter {
    half_taps: usize,
    side_offsets: Vec<i64>,
    side_coeffs: AlignedVec<f32, 64>,
    side_coeffs_q15: AlignedVec<i16, 64>,
    center_coeff: f32,
    center_coeff_q15: i16,
}

impl FilterBank {
    pub(crate) fn new(input_rate: u32, output_rate: u32, quality: Quality) -> Self {
        if matches!((input_rate, output_rate), (8_000, 16_000) | (16_000, 8_000)) {
            let half_band = HalfBandFilter::new(quality);
            return Self {
                taps: half_band.taps(),
                phases: quality.phases(),
                coeffs: AlignedVec::from_slice(&[]),
                coeffs_q15: AlignedVec::from_slice(&[]),
                half_band: Some(half_band),
                third_band: None,
            };
        }

        if matches!((input_rate, output_rate), (24_000, 8_000)) {
            let third_band = ThirdBandFilter::new(quality);
            return Self {
                taps: third_band.taps(),
                phases: quality.phases(),
                coeffs: AlignedVec::from_slice(&[]),
                coeffs_q15: AlignedVec::from_slice(&[]),
                half_band: None,
                third_band: Some(third_band),
            };
        }

        let taps = quality.taps();
        let phases = quality.phases();
        let cutoff = if output_rate < input_rate {
            output_rate as f64 / input_rate as f64
        } else {
            1.0
        };
        let cutoff = (0.95 * cutoff.min(1.0)).min(0.98);
        let beta = match quality {
            Quality::Fast => 5.0,
            Quality::Balanced => 7.5,
            Quality::Best => 10.0,
        };
        let denom = modified_bessel0(beta);
        let mut coeffs = vec![0.0f32; taps * phases];

        for phase in 0..phases {
            let frac = phase as f64 / phases as f64;
            let mut sum = 0.0;
            for tap in 0..taps {
                let centered = tap as f64 - (taps as f64 - 1.0) * 0.5 - frac;
                let x = centered * cutoff;
                let sinc = if x.abs() < 1.0e-12 {
                    cutoff
                } else {
                    cutoff * (PI * x).sin() / (PI * x)
                };
                let window_pos = (2.0 * tap as f64) / (taps as f64 - 1.0) - 1.0;
                let window =
                    modified_bessel0(beta * (1.0 - window_pos * window_pos).sqrt()) / denom;
                let value = sinc * window;
                coeffs[phase * taps + tap] = value as f32;
                sum += value;
            }
            if sum != 0.0 {
                for tap in 0..taps {
                    coeffs[phase * taps + tap] /= sum as f32;
                }
            }
        }

        let coeffs_q15: Vec<i16> = coeffs
            .iter()
            .map(|&coeff| {
                let scaled = (coeff * 32767.0)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32);
                scaled as i16
            })
            .collect();

        Self {
            taps,
            phases,
            coeffs: AlignedVec::from_slice(&coeffs),
            coeffs_q15: AlignedVec::from_slice(&coeffs_q15),
            half_band: None,
            third_band: None,
        }
    }

    #[inline(always)]
    pub(crate) fn taps(&self) -> usize {
        self.taps
    }

    #[inline(always)]
    pub(crate) fn half_taps(&self) -> usize {
        self.taps / 2
    }

    #[inline(always)]
    pub(crate) fn coeffs_for_phase(&self, phase: usize) -> &[f32] {
        let start = phase * self.taps;
        &self.coeffs[start..start + self.taps]
    }

    #[inline(always)]
    pub(crate) fn coeffs_q15_for_phase(&self, phase: usize) -> &[i16] {
        let start = phase * self.taps;
        &self.coeffs_q15[start..start + self.taps]
    }

    pub(crate) fn phase_count(&self) -> usize {
        self.phases
    }

    #[inline(always)]
    pub(crate) fn half_band(&self) -> &HalfBandFilter {
        self.half_band
            .as_ref()
            .expect("half-band filter is only available for exact 8 kHz <-> 16 kHz ratios")
    }

    #[inline(always)]
    pub(crate) fn third_band(&self) -> &ThirdBandFilter {
        self.third_band
            .as_ref()
            .expect("third-band filter is only available for exact 24 kHz -> 8 kHz ratios")
    }
}

impl HalfBandFilter {
    fn new(quality: Quality) -> Self {
        let half_taps = quality.taps() / 2;
        let taps = half_taps * 2 + 1;
        let center = half_taps as i64;
        let beta = match quality {
            Quality::Fast => 5.0,
            Quality::Balanced => 7.5,
            Quality::Best => 10.0,
        };
        let denom = modified_bessel0(beta);
        let mut coeffs = vec![0.0f64; taps];

        for (tap, coeff) in coeffs.iter_mut().enumerate() {
            let offset = tap as i64 - center;
            if offset == 0 {
                *coeff = 0.5;
            } else if offset & 1 != 0 {
                let x = offset as f64;
                let sinc = (0.5 * PI * x).sin() / (PI * x);
                let window_pos = (2.0 * tap as f64) / (taps as f64 - 1.0) - 1.0;
                let window =
                    modified_bessel0(beta * (1.0 - window_pos * window_pos).sqrt()) / denom;
                *coeff = sinc * window;
            }
        }

        let sum = coeffs.iter().sum::<f64>();
        if sum != 0.0 {
            for coeff in &mut coeffs {
                *coeff /= sum;
            }
        }

        let mut side_offsets = Vec::new();
        let mut side_input_offsets_for_up = Vec::new();
        let mut side_coeffs = Vec::new();
        for (tap, &coeff) in coeffs.iter().enumerate() {
            let offset = tap as i64 - center;
            if offset != 0 && offset & 1 != 0 {
                side_offsets.push(offset);
                side_input_offsets_for_up.push((1 - offset) / 2);
                side_coeffs.push(coeff as f32);
            }
        }

        let side_coeffs_up: Vec<f32> = side_coeffs.iter().map(|&coeff| coeff * 2.0).collect();
        let side_coeffs_q15: Vec<i16> = side_coeffs.iter().map(|&coeff| q15(coeff)).collect();
        let side_coeffs_up_q15: Vec<i16> = side_coeffs_up.iter().map(|&coeff| q15(coeff)).collect();
        let center_coeff = coeffs[half_taps] as f32;
        let center_coeff_q15 = q15(center_coeff);

        Self {
            half_taps,
            side_offsets,
            side_input_offsets_for_up,
            side_coeffs: AlignedVec::from_slice(&side_coeffs),
            side_coeffs_up: AlignedVec::from_slice(&side_coeffs_up),
            side_coeffs_q15: AlignedVec::from_slice(&side_coeffs_q15),
            side_coeffs_up_q15: AlignedVec::from_slice(&side_coeffs_up_q15),
            center_coeff,
            center_coeff_q15,
        }
    }

    #[inline(always)]
    fn taps(&self) -> usize {
        self.half_taps * 2 + 1
    }

    #[inline(always)]
    pub(crate) fn side_offsets(&self) -> &[i64] {
        &self.side_offsets
    }

    #[inline(always)]
    pub(crate) fn side_input_offsets_for_up(&self) -> &[i64] {
        &self.side_input_offsets_for_up
    }

    #[inline(always)]
    pub(crate) fn side_coeffs(&self) -> &[f32] {
        &self.side_coeffs
    }

    #[inline(always)]
    pub(crate) fn side_coeffs_up(&self) -> &[f32] {
        &self.side_coeffs_up
    }

    #[inline(always)]
    pub(crate) fn side_coeffs_q15(&self) -> &[i16] {
        &self.side_coeffs_q15
    }

    #[inline(always)]
    pub(crate) fn side_coeffs_up_q15(&self) -> &[i16] {
        &self.side_coeffs_up_q15
    }

    #[inline(always)]
    pub(crate) fn center_coeff(&self) -> f32 {
        self.center_coeff
    }

    #[inline(always)]
    pub(crate) fn center_coeff_q15(&self) -> i16 {
        self.center_coeff_q15
    }
}

impl ThirdBandFilter {
    fn new(quality: Quality) -> Self {
        let half_taps = quality.taps() / 2;
        let taps = half_taps * 2 + 1;
        let center = half_taps as i64;
        let cutoff = 1.0 / 3.0;
        let beta = match quality {
            Quality::Fast => 5.0,
            Quality::Balanced => 7.5,
            Quality::Best => 10.0,
        };
        let denom = modified_bessel0(beta);
        let mut coeffs = vec![0.0f64; taps];

        for (tap, coeff) in coeffs.iter_mut().enumerate() {
            let offset = tap as i64 - center;
            if offset == 0 {
                *coeff = cutoff;
            } else if offset % 3 != 0 {
                let x = offset as f64 * cutoff;
                let sinc = cutoff * (PI * x).sin() / (PI * x);
                let window_pos = (2.0 * tap as f64) / (taps as f64 - 1.0) - 1.0;
                let window =
                    modified_bessel0(beta * (1.0 - window_pos * window_pos).sqrt()) / denom;
                *coeff = sinc * window;
            }
        }

        let sum = coeffs.iter().sum::<f64>();
        if sum != 0.0 {
            for coeff in &mut coeffs {
                *coeff /= sum;
            }
        }

        let mut side_offsets = Vec::new();
        let mut side_coeffs = Vec::new();
        for (tap, &coeff) in coeffs.iter().enumerate() {
            let offset = tap as i64 - center;
            if offset != 0 && offset % 3 != 0 {
                side_offsets.push(offset);
                side_coeffs.push(coeff as f32);
            }
        }

        let side_coeffs_q15: Vec<i16> = side_coeffs.iter().map(|&coeff| q15(coeff)).collect();
        let center_coeff = coeffs[half_taps] as f32;
        let center_coeff_q15 = q15(center_coeff);

        Self {
            half_taps,
            side_offsets,
            side_coeffs: AlignedVec::from_slice(&side_coeffs),
            side_coeffs_q15: AlignedVec::from_slice(&side_coeffs_q15),
            center_coeff,
            center_coeff_q15,
        }
    }

    #[inline(always)]
    fn taps(&self) -> usize {
        self.half_taps * 2 + 1
    }

    #[inline(always)]
    pub(crate) fn side_offsets(&self) -> &[i64] {
        &self.side_offsets
    }

    #[inline(always)]
    pub(crate) fn side_coeffs(&self) -> &[f32] {
        &self.side_coeffs
    }

    #[inline(always)]
    pub(crate) fn side_coeffs_q15(&self) -> &[i16] {
        &self.side_coeffs_q15
    }

    #[inline(always)]
    pub(crate) fn center_coeff(&self) -> f32 {
        self.center_coeff
    }

    #[inline(always)]
    pub(crate) fn center_coeff_q15(&self) -> i16 {
        self.center_coeff_q15
    }
}

fn modified_bessel0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x * 0.5;
    for k in 1..=24 {
        let k = k as f64;
        term *= (half * half) / (k * k);
        sum += term;
        if term < 1.0e-14 * sum {
            break;
        }
    }
    sum
}

fn q15(coeff: f32) -> i16 {
    let scaled = (coeff * 32767.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32);
    scaled as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_phase_major_coefficients() {
        let bank = FilterBank::new(48_000, 44_100, Quality::Balanced);
        assert_eq!(bank.taps(), 48);
        assert_eq!(bank.phase_count(), 512);
        let coeffs = bank.coeffs_for_phase(128);
        assert_eq!(coeffs.len(), 48);
        let sum = coeffs.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn creates_compact_half_band_for_exact_8k_16k_ratios() {
        let bank = FilterBank::new(8_000, 16_000, Quality::Fast);
        let half_band = bank.half_band();
        assert_eq!(bank.taps(), Quality::Fast.taps() + 1);
        assert_eq!(bank.half_taps(), Quality::Fast.taps() / 2);
        assert_eq!(half_band.side_offsets().len(), Quality::Fast.taps() / 2);
        assert!((half_band.center_coeff() - 0.5).abs() < 0.01);
    }

    #[test]
    fn creates_compact_third_band_for_exact_24k_to_8k_ratio() {
        let bank = FilterBank::new(24_000, 8_000, Quality::Fast);
        let third_band = bank.third_band();
        assert_eq!(bank.taps(), Quality::Fast.taps() + 1);
        assert_eq!(bank.half_taps(), Quality::Fast.taps() / 2);
        assert_eq!(third_band.side_offsets().len(), 16);
        assert!(
            third_band
                .side_offsets()
                .iter()
                .all(|offset| offset % 3 != 0)
        );
        let sum = third_band.center_coeff() + third_band.side_coeffs().iter().copied().sum::<f32>();
        assert!((sum - 1.0).abs() < 0.001);
        assert!((third_band.center_coeff() - 1.0 / 3.0).abs() < 0.01);
    }
}
