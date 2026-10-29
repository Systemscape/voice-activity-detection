use serde::Deserialize;
use std::io::{self, BufRead, Write};
use voice_activity_detection::model::Model;
use voice_activity_detection::preprocessing::Preprocessor;

#[derive(Deserialize)]
struct Request {
    features: Option<Vec<Vec<f32>>>,
    samples: Option<Vec<f32>>,
    #[serde(default)]
    include_features: bool,
}

fn run() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("Usage: infer MODEL.json")?;
    let model = Model::load(&path)?;
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let answer = serde_json::from_str::<Request>(&line)
            .map_err(|e| e.to_string())
            .and_then(|request| {
                let features = match (request.features, request.samples) {
                    (Some(features), None) => features,
                    (None, Some(samples)) => Preprocessor::new()
                        .push(&samples)
                        .into_iter()
                        .map(|f| f.features)
                        .collect(),
                    _ => return Err("Supply either features or 16 kHz mono samples".into()),
                };
                let probabilities = model.predict(features.clone())?;
                Ok((probabilities, request.include_features.then_some(features)))
            });
        let value = match answer {
            Ok((probabilities, features)) => {
                serde_json::json!({ "probabilities": probabilities, "features": features })
            }
            Err(error) => serde_json::json!({ "error": error }),
        };
        serde_json::to_writer(&mut output, &value).map_err(|e| e.to_string())?;
        writeln!(output).map_err(|e| e.to_string())?;
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("VAD inference: {error}");
        std::process::exit(1);
    }
}
