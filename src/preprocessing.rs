//! 16 kHz speech preprocessing with nalgebra matrices throughout.
//!
//! The DFT, mel bank, orthonormal DCT, and Savitzky–Golay derivative
//! operators match features.py. Nine frames supply centered derivatives;
//! a prediction has 50 ms of lookahead. No Python runs in the live app.
use nalgebra::{DMatrix, DVector};
use serde::Serialize;
use std::collections::VecDeque;
use std::f64::consts::PI;

pub const SAMPLE_RATE: usize = 16000;
pub const HOP: usize = 160;
const WINDOW: usize = 320;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MELS: usize = 64;

struct Frame {
    mfcc: DVector<f64>,
    spectral: [f64; 4],
    rms: f32,
    index: usize,
}

#[derive(Debug, Serialize)]
pub struct FeatureFrame {
    pub features: Vec<f32>,
    pub rms: f32,
    pub sample_index: usize,
}

pub struct Preprocessor {
    real_dft: DMatrix<f64>,
    imaginary_dft: DMatrix<f64>,
    mel_bank: DMatrix<f64>,
    dct: DMatrix<f64>,
    first_derivative: DVector<f64>,
    second_derivative: DVector<f64>,
    audio: VecDeque<f64>,
    frames: VecDeque<Frame>,
    frame_count: usize,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor {
    pub fn new() -> Self {
        let real_dft = DMatrix::from_fn(BINS, WINDOW, |k, n| {
            let hamming = 0.54 - 0.46 * (2.0 * PI * n as f64 / WINDOW as f64).cos();
            hamming * (2.0 * PI * k as f64 * n as f64 / FFT as f64).cos()
        });
        let imaginary_dft = DMatrix::from_fn(BINS, WINDOW, |k, n| {
            let hamming = 0.54 - 0.46 * (2.0 * PI * n as f64 / WINDOW as f64).cos();
            -hamming * (2.0 * PI * k as f64 * n as f64 / FFT as f64).sin()
        });
        let mel_max = 2595.0 * (1.0 + SAMPLE_RATE as f64 / 1400.0).log10();
        let hz: Vec<f64> = (0..MELS + 2)
            .map(|i| 700.0 * (10f64.powf(mel_max * i as f64 / (MELS + 1) as f64 / 2595.0) - 1.0))
            .collect();
        let mel_bank = DMatrix::from_fn(MELS, BINS, |m, k| {
            let frequency = k as f64 * SAMPLE_RATE as f64 / FFT as f64;
            let lower = (frequency - hz[m]) / (hz[m + 1] - hz[m]);
            let upper = (hz[m + 2] - frequency) / (hz[m + 2] - hz[m + 1]);
            lower.min(upper).max(0.0) * 2.0 / (hz[m + 2] - hz[m])
        });
        let dct = DMatrix::from_fn(40, MELS, |k, n| {
            let scale = if k == 0 {
                (1.0 / MELS as f64).sqrt()
            } else {
                (2.0 / MELS as f64).sqrt()
            };
            scale * (PI * k as f64 * (n as f64 + 0.5) / MELS as f64).cos()
        });
        let polynomial = DMatrix::from_fn(9, 3, |row, col| (row as f64 - 4.0).powi(col as i32));
        let fit = (polynomial.transpose() * &polynomial)
            .try_inverse()
            .unwrap()
            * polynomial.transpose();
        let first_derivative = fit.row(1).transpose().into_owned();
        let second_derivative = fit.row(2).transpose().into_owned() * 2.0;
        Self {
            real_dft,
            imaginary_dft,
            mel_bank,
            dct,
            first_derivative,
            second_derivative,
            audio: VecDeque::from(vec![0.0; WINDOW / 2]),
            frames: VecDeque::new(),
            frame_count: 0,
        }
    }

    fn analyze(&self, samples: &[f64], index: usize) -> Frame {
        let x = DVector::from_column_slice(samples);
        let real = &self.real_dft * &x;
        let imaginary = &self.imaginary_dft * &x;
        let power = real.component_mul(&real) + imaginary.component_mul(&imaginary);
        let magnitude = power.map(|v| v.sqrt());
        let total = magnitude.sum();
        let log_mel = (&self.mel_bank * power).map(|v| 10.0 * v.max(1e-12).log10());
        let mfcc = &self.dct * log_mel;
        let frequencies = DVector::from_fn(BINS, |k, _| k as f64 * SAMPLE_RATE as f64 / FFT as f64);
        let centroid = magnitude.dot(&frequencies) / total.max(1e-12);
        let variance = frequencies.map(|f| (f - centroid).powi(2));
        let bandwidth = (magnitude.dot(&variance) / total.max(1e-12)).sqrt();
        let mut cumulative = 0.0;
        let mut rolloff = 0.0;
        for (k, mag) in magnitude.iter().enumerate() {
            cumulative += mag;
            if cumulative >= 0.85 * total {
                rolloff = frequencies[k];
                break;
            }
        }
        let crossings = samples
            .windows(2)
            .filter(|w| w[0].is_sign_negative() != w[1].is_sign_negative())
            .count();
        let zcr = crossings as f64 / (WINDOW - 1) as f64;
        Frame {
            mfcc,
            spectral: [bandwidth, centroid, rolloff, zcr],
            rms: (x.dot(&x) / WINDOW as f64).sqrt() as f32,
            index,
        }
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<FeatureFrame> {
        self.audio.extend(
            samples
                .iter()
                .map(|&v| if v.is_finite() { v as f64 } else { 0.0 }),
        );
        let mut output = Vec::new();
        while self.audio.len() >= WINDOW {
            let window: Vec<f64> = self.audio.iter().take(WINDOW).copied().collect();
            self.frames
                .push_back(self.analyze(&window, self.frame_count));
            self.frame_count += 1;
            self.audio.drain(..HOP);
            if self.frames.len() == 9 {
                let mfcc = DMatrix::from_fn(40, 9, |coefficient, frame| {
                    self.frames[frame].mfcc[coefficient]
                });
                let first = &mfcc * &self.first_derivative;
                let second = &mfcc * &self.second_derivative;
                let center = &self.frames[4];
                let mut features = Vec::with_capacity(62);
                features.extend(center.mfcc.iter().map(|&v| v as f32));
                features.extend(first.iter().take(12).map(|&v| v as f32));
                features.extend(second.iter().take(6).map(|&v| v as f32));
                features.extend(center.spectral.iter().map(|&v| v as f32));
                output.push(FeatureFrame {
                    features,
                    rms: center.rms,
                    sample_index: center.index * HOP,
                });
                self.frames.pop_front();
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_finite_and_derivatives_vanish() {
        let frames = Preprocessor::new().push(&vec![0.0; 3200]);
        assert_eq!(frames.len(), 12);
        assert_eq!(frames[0].sample_index, 640);
        for frame in frames {
            assert_eq!(frame.features.len(), 62);
            assert!(frame.features.iter().all(|x| x.is_finite()));
            assert!((frame.features[0] + 960.0).abs() < 1e-4);
            assert!(frame.features[40..58].iter().all(|v| v.abs() < 1e-5));
            assert_eq!(frame.rms, 0.0);
        }
    }

    #[test]
    fn arbitrary_capture_chunk_sizes_produce_identical_frames() {
        let audio: Vec<f32> = (0..4000).map(|n| (n as f32 * 0.17).sin() * 0.1).collect();
        let expected = Preprocessor::new().push(&audio);
        let mut state = Preprocessor::new();
        let actual: Vec<_> = audio
            .chunks(37)
            .flat_map(|chunk| state.push(chunk))
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.features, b.features);
            assert_eq!(a.sample_index, b.sample_index);
        }
    }
}
