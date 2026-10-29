# voice-activity-detection

Native Rust desktop app that listens to a microphone and shows, in real time,
whether somebody is speaking. [egui](https://docs.rs/eframe) draws the window,
[nalgebra](https://nalgebra.rs) computes the speech features and
[Burn](https://burn.dev) runs a small feedforward model on the CPU.

## Run

```sh
cargo run --release
```

Or with [just](https://just.systems): `just run` (`just r`), `just build` (`just b`),
`just` lists all recipes.

Pick a microphone and click **START**. The button turns blue while listening and
green when speech is detected. Input level and speech confidence are shown below;
raise the threshold slider if background noise triggers the indicator.

```sh
cargo run --release -- --list-microphones   # show capture devices
cargo run --release -- --model other.json   # use another exported model
cargo run --release -- --preview            # 4 s UI preview, microphone stays off
```

## Requirements

- Rust 1.95 or newer
- Linux desktop session (X11 or Wayland) with OpenGL
- `arecord` from `alsa-utils` for audio capture (16 kHz mono PCM, processed in
  memory, nothing is written to disk)

## How it works

Audio is cut into frames and turned into 62 features: windowed DFT, mel filter
bank, DCT cepstrum, spectral statistics and polynomial derivatives over nine
frames (about 50 ms lookahead). The model in `model/speech_detector.json` is a
feedforward network with BatchNorm and ReLU layers exported from PyTorch; feature
normalisation is part of the export. Probabilities are smoothed and a 300 ms
hangover keeps the indicator stable between syllables.

The bundled model was trained on a small augmented dataset with automatically
generated GMM VAD labels (87.7 % held-out frame accuracy). It is a demonstration
model, not a production detector.

## Headless inference

The `infer` binary reads newline-delimited JSON on stdin and answers with
per-frame probabilities, useful for checks without a display or microphone:

```sh
cargo build --release
python3 -c 'import json; print(json.dumps({"samples": [0.0] * 16000}))' | \
  target/release/infer model/speech_detector.json
```

Send `{"samples": [...]}` with at least 1440 samples of 16 kHz mono audio, or
`{"features": [[...62 numbers...]]}` to skip preprocessing.

## Tests

```sh
cargo test --release
```

Includes a headless layout check of the window and streaming tests that verify
chunk-size independence of the feature pipeline.

## License

MIT, see [LICENSE](LICENSE).
