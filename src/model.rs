//! A local newline-delimited JSON worker: feature batches in, VAD probabilities out.
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData, activation};
use serde::Deserialize;

type Backend = NdArray<f32>;
type Matrix = Tensor<Backend, 2>;

#[derive(Deserialize)]
struct ModelFile {
    schema_version: u32,
    feature_count: usize,
    layers: Vec<LayerFile>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LayerFile {
    Affine {
        scale: Vec<f32>,
        bias: Vec<f32>,
    },
    Linear {
        input: usize,
        output: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    },
    Relu,
}

enum Layer {
    Affine(Matrix, Matrix),
    Linear(Matrix, Matrix),
    Relu,
}

pub struct Model {
    features: usize,
    layers: Vec<Layer>,
}

fn matrix(values: Vec<f32>, rows: usize, columns: usize) -> Matrix {
    Matrix::from_data(
        TensorData::new(values, [rows, columns]),
        &Default::default(),
    )
}

impl Model {
    pub fn load(path: &str) -> Result<Self, String> {
        let content = std::fs::read(path).map_err(|e| e.to_string())?;
        let definition: ModelFile = serde_json::from_slice(&content).map_err(|e| e.to_string())?;
        if definition.schema_version != 1 || definition.feature_count != 62 {
            return Err("Expected schema 1 and 62 speech features".into());
        }
        let mut width = definition.feature_count;
        let mut layers = Vec::new();
        for definition in definition.layers {
            match definition {
                LayerFile::Affine { scale, bias } => {
                    if scale.len() != width
                        || bias.len() != width
                        || !scale.iter().chain(&bias).all(|x| x.is_finite())
                    {
                        return Err("Invalid affine layer dimensions or values".into());
                    }
                    layers.push(Layer::Affine(
                        matrix(scale, 1, width),
                        matrix(bias, 1, width),
                    ));
                }
                LayerFile::Linear {
                    input,
                    output,
                    weight,
                    bias,
                } => {
                    if input != width
                        || output == 0
                        || weight.len() != input * output
                        || bias.len() != output
                        || !weight.iter().chain(&bias).all(|x| x.is_finite())
                    {
                        return Err("Invalid linear layer dimensions or values".into());
                    }
                    layers.push(Layer::Linear(
                        matrix(weight, input, output),
                        matrix(bias, 1, output),
                    ));
                    width = output;
                }
                LayerFile::Relu => layers.push(Layer::Relu),
            }
        }
        if width != 1 || layers.is_empty() {
            return Err("Model must produce one speech logit".into());
        }
        Ok(Self {
            features: definition.feature_count,
            layers,
        })
    }

    pub fn predict(&self, rows: Vec<Vec<f32>>) -> Result<Vec<f32>, String> {
        if rows.is_empty()
            || rows.len() > 10000
            || rows
                .iter()
                .any(|r| r.len() != self.features || !r.iter().all(|v| v.is_finite()))
        {
            return Err("Expected 1..10000 finite feature rows with 62 columns".into());
        }
        let count = rows.len();
        let mut x = matrix(rows.into_iter().flatten().collect(), count, self.features);
        for layer in &self.layers {
            x = match layer {
                Layer::Affine(scale, bias) => x * scale.clone() + bias.clone(),
                Layer::Linear(weight, bias) => x.matmul(weight.clone()) + bias.clone(),
                Layer::Relu => activation::relu(x),
            };
        }
        activation::sigmoid(x)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burn_executes_affine_relu_and_linear() {
        let model = Model {
            features: 2,
            layers: vec![
                Layer::Affine(matrix(vec![2., 1.], 1, 2), matrix(vec![0., -1.], 1, 2)),
                Layer::Relu,
                Layer::Linear(matrix(vec![1., -1.], 2, 1), matrix(vec![0.], 1, 1)),
            ],
        };
        let result = model.predict(vec![vec![1., 2.], vec![-1., 0.]]).unwrap();
        assert!((result[0] - 1. / (1. + (-1f32).exp())).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!(model.predict(vec![vec![1.]]).is_err());
    }
}
