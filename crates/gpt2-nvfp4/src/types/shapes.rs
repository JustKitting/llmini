#[path = "shapes/init.rs"]
mod init;

use crate::{
    FixedBytes, GPT2_MLP, GPT2_N_EMBD, GPT2_QK_SCALE_STORAGE, GPT2_QKV, GPT2_VOCAB_SIZE,
    NEXTLAT_HIDDEN, NEXTLAT_INPUT, Nvfp4Shape, Nvfp4Tensor,
};

use super::LinearWeights;

pub(crate) use init::Nvfp4ShapeInit;

pub const fn nvfp4_bytes(rows: usize, cols: usize) -> usize {
    (rows * cols).div_ceil(2)
}

pub const fn nvfp4_scales(rows: usize, cols: usize) -> usize {
    (rows * cols).div_ceil(16)
}

macro_rules! nvfp4_shape {
    ($name:ident, $rows:expr, $cols:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub enum $name {}

        impl Nvfp4Shape for $name {
            const ROWS: usize = $rows;
            const COLS: usize = $cols;
            const BYTE_LEN: usize = nvfp4_bytes($rows, $cols);
            const SCALE_LEN: usize = nvfp4_scales($rows, $cols);

            type Bytes = FixedBytes<{ nvfp4_bytes($rows, $cols) }>;
            type Scales = FixedBytes<{ nvfp4_scales($rows, $cols) }>;

            fn zero_bytes() -> Self::Bytes {
                FixedBytes::zeroed()
            }

            fn zero_scales() -> Self::Scales {
                FixedBytes::zeroed()
            }
        }
    };
}

nvfp4_shape!(TokenEmbeddingShape, GPT2_VOCAB_SIZE, GPT2_N_EMBD);
nvfp4_shape!(HiddenVectorShape, 1, GPT2_N_EMBD);
nvfp4_shape!(QkvWeightShape, GPT2_N_EMBD, GPT2_QKV);
nvfp4_shape!(QkvVectorShape, 1, GPT2_QKV);
nvfp4_shape!(QkScaleShape, 1, GPT2_QK_SCALE_STORAGE);
nvfp4_shape!(ResidualWeightShape, GPT2_N_EMBD, GPT2_N_EMBD);
nvfp4_shape!(MlpUpWeightShape, GPT2_N_EMBD, GPT2_MLP);
nvfp4_shape!(MlpVectorShape, 1, GPT2_MLP);
nvfp4_shape!(MlpDownWeightShape, GPT2_MLP, GPT2_N_EMBD);
nvfp4_shape!(NextLatInputShape, 1, NEXTLAT_INPUT);
nvfp4_shape!(NextLatHiddenShape, 1, NEXTLAT_HIDDEN);
nvfp4_shape!(NextLatProjectionWeightShape, NEXTLAT_INPUT, NEXTLAT_HIDDEN);
nvfp4_shape!(NextLatTransitionWeightShape, NEXTLAT_HIDDEN, NEXTLAT_HIDDEN);
nvfp4_shape!(NextLatOutWeightShape, NEXTLAT_HIDDEN, GPT2_N_EMBD);

pub type TokenEmbedding = Nvfp4Tensor<TokenEmbeddingShape>;
pub type LayerNormTensor = Nvfp4Tensor<HiddenVectorShape>;
pub type QkvLinear = LinearWeights<QkvWeightShape, QkvVectorShape>;
pub type QkScaleTensor = Nvfp4Tensor<QkScaleShape>;
pub type ResidualLinear = LinearWeights<ResidualWeightShape, HiddenVectorShape>;
pub type MlpUpLinear = LinearWeights<MlpUpWeightShape, MlpVectorShape>;
pub type MlpDownLinear = LinearWeights<MlpDownWeightShape, HiddenVectorShape>;
