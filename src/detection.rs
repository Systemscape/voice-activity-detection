//! Probability smoothing and a short hangover avoid flickering between syllables.
pub struct Detection {
    pub probability: f32,
    pub speech: bool,
    pub threshold: f32,
    pub level_db: f32,
    hangover: usize,
}

impl Default for Detection {
    fn default() -> Self {
        Self {
            probability: 0.0,
            speech: false,
            threshold: 0.9,
            level_db: -100.0,
            hangover: 0,
        }
    }
}

impl Detection {
    pub fn update(&mut self, probability: f32, rms: f32) {
        self.level_db = 20.0 * rms.max(1e-5).log10();
        let probability = if self.level_db < -55.0 {
            0.0
        } else {
            probability.clamp(0.0, 1.0)
        };
        self.probability = 0.8 * self.probability + 0.2 * probability;
        if self.probability >= self.threshold {
            self.speech = true;
            self.hangover = 30;
        } else if self.speech {
            if self.probability >= (self.threshold - 0.15).max(0.0) {
                self.hangover = 30;
            } else {
                self.hangover = self.hangover.saturating_sub(1);
                if self.hangover == 0 {
                    self.speech = false;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_hangover_and_silence_gate() {
        let mut state = Detection::default();
        for _ in 0..20 {
            state.update(1.0, 0.1);
        }
        assert!(state.speech);
        for _ in 0..10 {
            state.update(0.0, 0.1);
        }
        assert!(state.speech);
        for _ in 0..50 {
            state.update(1.0, 0.0);
        }
        assert!(!state.speech);
    }
}
