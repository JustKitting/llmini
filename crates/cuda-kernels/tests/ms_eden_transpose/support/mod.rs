mod fixtures;
mod quant;
mod source;

pub(super) use self::fixtures::{cpu_amax, input_matrix, padded_rows};
pub(super) use self::quant::QuantScratch;
pub(super) use self::source::{RowwiseSourceScratch, SourceScratch};

pub(super) const ROWS: usize = 33;
pub(super) const COLS: usize = 128;
pub(super) const SIGN_SEED: u32 = 0x1c69_b3f5;
pub(super) const SCALE_SEED: u32 = 0x4a7c_15d3;
pub(super) const SCALE_OVERRIDE: f32 = 0.25;
