use std::f64::consts::PI;

use crate::Quality;
use crate::aligned::AlignedVec;

#[derive(Debug, Clone)]
pub(crate) struct FilterBank {
    taps: usize,
    phases: usize,
    coeffs: AlignedVec<f32, 64>,
    coeffs_q15: AlignedVec<i16, 64>,
}

impl FilterBank {
    pub(crate) fn new(input_rate: u32, output_rate: u32, quality: Quality) -> Self {
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
}
