//! Reference-sized configuration; checkpoints carry their exact per-layer graph.
use super::{HEAD_DIM, HEADS, LAYERS, MULTIPLIERS};
mod budget;
pub(crate) use budget::{MultistreamOptions, valid_multistream_layers};
#[derive(Clone, Debug)]
pub(crate) struct Config {
    version: u32,
    pub layer: usize,
    pub head_dim: usize,
    pub convolution_dilation: u32,
    pub primes: Vec<u32>,
    pub multipliers: Vec<u64>,
    pub legacy: bool,
}
pub(crate) const fn reference_primes(index: usize) -> [u32; 16] {
    let mut values = [0; 16];
    let mut at = 0;
    let mut p = (llama2_tokenizer::MODEL_VOCAB_SIZE * 5) as u32;
    while at < (index + 1) * 16 {
        p += 1;
        let mut d = 2;
        let mut prime = true;
        while d * d <= p {
            if p % d == 0 {
                prime = false;
                break;
            }
            d += 1;
        }
        if prime {
            if at >= index * 16 {
                values[at - index * 16] = p;
            }
            at += 1;
        }
    }
    values
}
fn is_prime(p: u32) -> bool {
    if p < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= p {
        if p % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}
impl Config {
    pub fn reference_multistream(index: usize) -> Self {
        let mut cfg = Self::reference(index);
        cfg.version = 4;
        cfg
    }
    pub fn streams(&self) -> usize {
        if matches!(self.version, 4 | 5) { 4 } else { 1 }
    }
    pub fn key_index(&self, stream: usize) -> usize {
        assert!(stream < self.streams());
        1 + 2 * stream
    }
    pub fn value_index(&self) -> usize {
        1 + 2 * self.streams()
    }
    pub fn projector_indices(&self) -> Vec<usize> {
        (0..=self.streams()).map(|i| 1 + 2 * i).collect()
    }
    pub fn adam_indices(&self) -> Vec<usize> {
        std::iter::once(0)
            .chain((0..=self.streams()).map(|i| 2 + 2 * i))
            .collect()
    }
    pub fn nv_names(&self) -> Vec<String> {
        let mut names = vec!["table".into()];
        for stream in 0..self.streams() {
            let prefix = if self.streams() == 1 {
                "key".into()
            } else {
                format!("key.{stream}")
            };
            names.extend([format!("{prefix}.weight"), format!("{prefix}.bias")]);
        }
        names.extend(["value.weight".into(), "value.bias".into()]);
        names
    }
    pub fn reference(index: usize) -> Self {
        let layer = LAYERS[index];
        let mut multipliers = MULTIPLIERS;
        if index > 0 {
            let mut rng = gpt2_nvfp4::Gpt2Rng::new(10007 * layer as u64);
            let bound = (i64::MAX as u64 / llama2_tokenizer::MODEL_VOCAB_SIZE as u64) / 2;
            for m in &mut multipliers {
                *m = (((rng.next_u32() as u64) << 32 | rng.next_u32() as u64) % bound) * 2 + 1;
            }
        }
        Self {
            version: 3,
            layer,
            head_dim: HEAD_DIM,
            convolution_dilation: 3,
            primes: reference_primes(index).to_vec(),
            multipliers: multipliers.to_vec(),
            legacy: false,
        }
    }
    pub(super) fn initialization_layer(&self) -> usize {
        if self.streams() == 4 && self.layer != LAYERS[0] {
            LAYERS[1]
        } else {
            self.layer
        }
    }

    pub fn memory(&self) -> usize {
        self.primes.len() * self.head_dim
    }
    pub fn max_ngram(&self) -> usize {
        self.multipliers.len()
    }
    pub fn trainable_parameters(&self) -> usize {
        let d = gpt2_nvfp4::GPT2_N_EMBD;
        self.table_len()
            + (self.streams() + 1) * (self.memory() * d + d)
            + self.streams() * d * 7
            + 3 * (self.streams() + 1)
    }
    pub fn gate_rms_epsilon(&self) -> f32 {
        match self.version {
            1 | 2 => 1e-5,
            // The authors' FP32 demo leaves query/key nn.RMSNorm eps unset.
            // Its convolution RMS explicitly remains 1e-5 in every version.
            3 | 4 | 5 => f32::EPSILON,
            _ => unreachable!("Engram graph version must be validated before use"),
        }
    }
    pub fn table_len(&self) -> usize {
        (self.primes.iter().map(|x| *x as usize).sum::<usize>() * self.head_dim).div_ceil(128) * 128
    }
    pub fn prefix(&self) -> String {
        if self.legacy {
            "engram".into()
        } else {
            format!("engram.layers.{}", self.layer)
        }
    }
    pub fn descriptor(&self) -> Vec<f32> {
        [
            self.version,
            HEADS as u32,
            self.head_dim as u32,
            self.memory() as u32,
            self.layer as u32,
            2,
            self.max_ngram() as u32,
            4,
            self.convolution_dilation,
        ]
        .into_iter()
        .chain(self.primes.iter().copied())
        .chain(
            self.multipliers
                .iter()
                .copied()
                .flat_map(|x| (0..4).map(move |i| ((x >> (16 * i)) & 65535) as u32)),
        )
        .map(|x| x as f32)
        .collect()
    }
    pub fn parse(values: &[f32], legacy: bool) -> crate::AppResult<Self> {
        if !matches!(values.len(), 37 | 49)
            || values
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || v.fract() != 0.)
        {
            return Err("invalid Engram architecture descriptor".into());
        }
        let v = values.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let max_ngram = if v[0] == 5 { 4 } else { 3 };
        let dilation = v[8];
        let hash_heads = (max_ngram - 1) * HEADS;
        let coefficient_start = 9 + hash_heads;
        if (legacy && v[0] != 1)
            || (!legacy && !matches!(v[0], 2..=5))
            || (matches!(v[0], 4 | 5) && v[2] != HEAD_DIM as u32)
            || v[1] != HEADS as u32
            || v[4] >= gpt2_nvfp4::GPT2_N_LAYER as u32
            || v[5..8] != [2, max_ngram as u32, 4]
            // Existing version 5 checkpoints recorded dilation three.
            || !(dilation == 3 || (max_ngram == 4 && dilation == 4))
            || !matches!(v[2], 32 | 64)
            || v[3] != hash_heads as u32 * v[2]
            || v.len() != coefficient_start + 4 * max_ngram
        {
            return Err("unsupported Engram architecture".into());
        }
        let primes = v[9..coefficient_start].to_vec();
        if primes
            .iter()
            .any(|p| *p < 128 || *p > 1_000_000 || !is_prime(*p))
            || primes
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != hash_heads
        {
            return Err("invalid Engram prime tables".into());
        }
        let multipliers = v[coefficient_start..]
            .chunks_exact(4)
            .map(|words| {
                words
                    .iter()
                    .enumerate()
                    .fold(0u64, |x, (i, &word)| x | (u64::from(word) << (i * 16)))
            })
            .collect::<Vec<_>>();
        if v[coefficient_start..].iter().any(|v| *v > 65535)
            || multipliers.iter().any(|m| {
                *m % 2 != 1 || *m > i64::MAX as u64 / llama2_tokenizer::MODEL_VOCAB_SIZE as u64
            })
        {
            return Err("invalid Engram hash coefficients".into());
        }
        Ok(Self {
            version: v[0],
            layer: v[4] as usize,
            head_dim: v[2] as usize,
            convolution_dilation: dilation,
            primes,
            multipliers,
            legacy,
        })
    }
}
