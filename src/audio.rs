//! Linux capture through alsa-utils. Device names are passed as arguments, never a shell.
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use voice_activity_detection::{model::Model, preprocessing::Preprocessor};

#[derive(Clone)]
pub struct Microphone {
    pub id: String,
    pub label: String,
}

pub fn microphones() -> Result<Vec<Microphone>, String> {
    let output = Command::new("arecord")
        .arg("-l")
        .output()
        .map_err(|e| format!("Cannot run arecord. Install alsa-utils: {e}"))?;
    let mut devices = vec![Microphone {
        id: "default".into(),
        label: "System default microphone".into(),
    }];
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(card) = line
            .strip_prefix("card ")
            .and_then(|s| s.split(':').next())
            .and_then(|s| s.trim().parse::<u32>().ok())
        else {
            continue;
        };
        let Some(device) = line
            .split("device ")
            .nth(1)
            .and_then(|s| s.split(':').next())
            .and_then(|s| s.trim().parse::<u32>().ok())
        else {
            continue;
        };
        devices.push(Microphone {
            id: format!("plughw:{card},{device}"),
            label: line.trim().to_owned(),
        });
    }
    Ok(devices)
}

pub enum AudioUpdate {
    Frame { probability: f32, rms: f32 },
    Error(String),
}

pub struct Capture {
    pub updates: mpsc::Receiver<AudioUpdate>,
    stopped: Arc<AtomicBool>,
    child: Arc<Mutex<Child>>,
    worker: Option<JoinHandle<()>>,
}

impl Capture {
    pub fn start(device: &str, model_path: &Path) -> Result<Self, String> {
        let model = Model::load(model_path.to_str().ok_or("Invalid model path")?)?;
        let mut child = Command::new("arecord")
            .args([
                "-q",
                "-D",
                device,
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "--buffer-time=100000",
                "--period-time=20000",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Could not start microphone capture: {e}"))?;
        let mut pcm = child.stdout.take().ok_or("Missing capture output")?;
        let mut stderr = child.stderr.take().ok_or("Missing capture errors")?;
        let capture_errors = Arc::new(Mutex::new(String::new()));
        let errors = capture_errors.clone();
        thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.by_ref().take(8192).read_to_string(&mut text);
            if let Ok(mut stored) = errors.lock() {
                *stored = text;
            }
        });
        let child = Arc::new(Mutex::new(child));
        let stopped = Arc::new(AtomicBool::new(false));
        let stop = stopped.clone();
        let (sender, updates) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut preprocessing = Preprocessor::new();
            let mut bytes = [0u8; 320];
            while !stop.load(Ordering::Acquire) {
                if let Err(error) = pcm.read_exact(&mut bytes) {
                    if !stop.load(Ordering::Acquire) {
                        let detail = capture_errors
                            .lock()
                            .map(|s| s.trim().to_owned())
                            .unwrap_or_default();
                        let message = if detail.is_empty() {
                            format!(
                                "Microphone capture stopped: {error}. Try another input device."
                            )
                        } else {
                            detail
                        };
                        let _ = sender.send(AudioUpdate::Error(message));
                    }
                    break;
                }
                let samples: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                let frames = preprocessing.push(&samples);
                for frame in frames {
                    match model.predict(vec![frame.features]) {
                        Ok(probability) => {
                            if sender
                                .send(AudioUpdate::Frame {
                                    probability: probability[0],
                                    rms: frame.rms,
                                })
                                .is_err()
                            {
                                return;
                            }
                        }
                        Err(error) => {
                            let _ = sender.send(AudioUpdate::Error(error));
                            return;
                        }
                    }
                }
            }
        });
        Ok(Self {
            updates,
            stopped,
            child,
            worker: Some(worker),
        })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
