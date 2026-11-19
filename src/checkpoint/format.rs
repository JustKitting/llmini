mod reader;
mod writer;

pub use reader::{CheckpointReader, CheckpointTensor};
pub use writer::CheckpointWriter;

pub(super) const MAGIC: &[u8] = b"GPT2_NVFP4_CHECKPOINT\n";
pub(super) const VERSION: u32 = 11;
