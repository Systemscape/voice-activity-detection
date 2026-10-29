mod audio;
mod detection;

use audio::{AudioUpdate, Capture, Microphone};
use detection::Detection;
use eframe::egui::{self, Color32, RichText};
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct SpeechApp {
    devices: Vec<Microphone>,
    selected: usize,
    capture: Option<Capture>,
    detection: Detection,
    error: Option<String>,
    model_path: PathBuf,
    preview: bool,
    started: Instant,
}

impl SpeechApp {
    fn new(model_path: PathBuf, preview: bool) -> Self {
        let (devices, error) = match audio::microphones() {
            Ok(devices) => (devices, None),
            Err(error) => (vec![], Some(error)),
        };
        Self {
            devices,
            selected: 0,
            capture: None,
            detection: Detection::default(),
            error,
            model_path,
            preview,
            started: Instant::now(),
        }
    }

    fn toggle(&mut self) {
        if self.capture.is_some() {
            self.capture = None;
            let threshold = self.detection.threshold;
            self.detection = Detection::default();
            self.detection.threshold = threshold;
            return;
        }
        if let Some(device) = self.devices.get(self.selected) {
            match Capture::start(&device.id, &self.model_path) {
                Ok(capture) => {
                    self.capture = Some(capture);
                    self.error = None;
                }
                Err(error) => self.error = Some(error),
            }
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui) {
        let updates: Vec<_> = self
            .capture
            .as_ref()
            .map(|c| c.updates.try_iter().collect())
            .unwrap_or_default();
        for update in updates {
            match update {
                AudioUpdate::Frame { probability, rms } => self.detection.update(probability, rms),
                AudioUpdate::Error(error) => {
                    self.capture = None;
                    self.error = Some(error);
                    self.detection.speech = false;
                }
            }
        }
        if self.preview {
            // Explicit visual preview; never starts or reads a microphone.
            self.detection.speech = true;
            self.detection.probability = 0.92;
            self.detection.level_db = -24.0;
            if self.started.elapsed() > Duration::from_secs(4) {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        let active = self.capture.is_some() || self.preview;
        ui.add_space(16.0);
        ui.heading(RichText::new("Speech detector").size(28.0));
        ui.label("Choose a microphone, then start listening.");
        ui.add_space(20.0);
        ui.label(RichText::new("MICROPHONE").small().strong());
        ui.add_enabled_ui(!active, |ui| {
            ui.horizontal(|ui| {
                let label = self
                    .devices
                    .get(self.selected)
                    .map(|d| d.label.as_str())
                    .unwrap_or("No input devices");
                egui::ComboBox::from_id_salt("microphone")
                    .selected_text(label)
                    .width(390.0)
                    .show_ui(ui, |ui| {
                        for (i, device) in self.devices.iter().enumerate() {
                            ui.selectable_value(&mut self.selected, i, &device.label);
                        }
                    });
                if ui.button("Refresh").clicked() {
                    match audio::microphones() {
                        Ok(devices) => {
                            self.devices = devices;
                            self.selected = 0;
                            self.error = None;
                        }
                        Err(error) => self.error = Some(error),
                    }
                }
            });
        });
        ui.add_space(24.0);
        let (text, color) = if !active {
            ("START", Color32::from_rgb(60, 76, 99))
        } else if self.detection.speech {
            ("SPEECH", Color32::from_rgb(25, 146, 92))
        } else {
            ("LISTENING", Color32::from_rgb(46, 82, 128))
        };
        let button = egui::Button::new(
            RichText::new(text)
                .size(48.0)
                .strong()
                .color(Color32::WHITE),
        )
        .fill(color)
        .corner_radius(18.0)
        .min_size(egui::vec2(ui.available_width(), 220.0));
        let button = if self.preview {
            button.sense(egui::Sense::hover())
        } else {
            button
        };
        if ui.add_enabled(!self.devices.is_empty(), button).clicked() && !self.preview {
            self.toggle();
        }
        ui.add_space(8.0);
        ui.label(if self.preview {
            "Visual preview — microphone is off"
        } else if active {
            "Click the large button to stop."
        } else {
            "Click the large button to start."
        });
        ui.add_space(18.0);
        ui.horizontal(|ui| {
            ui.label("Speech confidence");
            ui.label(RichText::new(format!("{:.0}%", self.detection.probability * 100.0)).strong());
        });
        ui.add(
            egui::ProgressBar::new(self.detection.probability).fill(Color32::from_rgb(25, 146, 92)),
        );
        ui.add_space(8.0);
        ui.label(format!("Input level: {:.0} dBFS", self.detection.level_db));
        ui.add(egui::ProgressBar::new(
            ((self.detection.level_db + 60.0) / 60.0).clamp(0.0, 1.0),
        ));
        ui.add_space(12.0);
        ui.add(
            egui::Slider::new(&mut self.detection.threshold, 0.3..=0.99).text("Speech threshold"),
        )
        .on_hover_text("Increase this if background noise triggers the indicator.");
        if let Some(error) = &self.error {
            ui.add_space(8.0);
            ui.colored_label(Color32::from_rgb(255, 140, 140), error);
        }
        if active {
            ui.ctx().request_repaint_after(Duration::from_millis(30));
        }
    }
}

impl eframe::App for SpeechApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.draw(ui));
        });
    }
}

const DEFAULT_MODEL: &str = "model/speech_detector.json";

fn default_model() -> PathBuf {
    // Binary next to a model directory, or inside target/<profile>/ of this repo.
    if let Ok(executable) = std::env::current_exe()
        && let Some(bin) = executable.parent()
    {
        for base in [bin.to_path_buf(), bin.join("../..")] {
            let candidate = base.join(DEFAULT_MODEL);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_MODEL)
}

fn main() -> eframe::Result {
    let mut model_path = default_model();
    let mut preview = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model_path = PathBuf::from(args.next().expect("--model needs a JSON path"))
            }
            "--preview" => preview = true,
            "--list-microphones" => {
                match audio::microphones() {
                    Ok(devices) => {
                        for device in devices {
                            println!("{}\t{}", device.id, device.label);
                        }
                    }
                    Err(error) => eprintln!("{error}"),
                }
                return Ok(());
            }
            "--help" | "-h" => {
                println!(
                    "Speech detector\n  --model PATH         Exported feedforward JSON\n  --list-microphones   List capture devices\n  --preview           Four-second UI preview; microphone stays off"
                );
                return Ok(());
            }
            _ => {
                eprintln!("Unknown option: {arg}");
                return Ok(());
            }
        }
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([610.0, 720.0])
            .with_min_inner_size([540.0, 610.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Speech detector",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(SpeechApp::new(model_path, preview)))
        }),
    )
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn controls_fit_the_default_window_without_a_display_server() {
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(610.0, 720.0));
        let mut app = SpeechApp {
            devices: vec![Microphone {
                id: "default".into(),
                label: "System default microphone".into(),
            }],
            selected: 0,
            capture: None,
            detection: Detection::default(),
            error: None,
            model_path: PathBuf::new(),
            preview: false,
            started: Instant::now(),
        };
        for preview in [false, true] {
            app.preview = preview;
            let mut used_height = 0.0;
            let result = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(screen),
                    ..Default::default()
                },
                |root| {
                    egui::CentralPanel::default().show(root, |ui| {
                        app.draw(ui);
                        used_height = ui.min_rect().max.y;
                    });
                },
            );
            assert!(!result.shapes.is_empty());
            assert!(
                used_height <= screen.max.y,
                "Controls overflow: {used_height}"
            );
        }
    }
}
