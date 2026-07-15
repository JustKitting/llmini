use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_LAYER, Gpt2ForwardSaved, Gpt2ForwardTape, HiddenState};

use super::device_buffer::block_array;
use super::tape_block::BlockTapeBuffers;
use super::tape_leaf::{LayerNormTapeBuffers, RowwiseTapeBuffers};

pub struct ForwardTapeBuffers {
    blocks: [BlockTapeBuffers; GPT2_N_LAYER],
    final_norm: LayerNormTapeBuffers,
    lm_head_input: RowwiseTapeBuffers,
}

impl ForwardTapeBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            blocks: block_array(|index| BlockTapeBuffers::new(stream, index))?,
            final_norm: LayerNormTapeBuffers::new(stream)?,
            lm_head_input: RowwiseTapeBuffers::gpt2_rows(stream, HiddenState::LEN)?,
        })
    }

    pub fn saved<'a>(
        &'a self,
        tokens: &'a DeviceBuffer<u32>,
        batch_size: u32,
        seq_len: u32,
        row_count: u32,
        logits: &'a DeviceBuffer<f32>,
    ) -> Gpt2ForwardSaved<'a> {
        Gpt2ForwardSaved {
            tokens,
            batch_size,
            seq_len,
            row_count,
            blocks: std::array::from_fn(|i| self.blocks[i].saved(batch_size, seq_len, row_count)),
            final_norm: self.final_norm.saved(row_count),
            lm_head_input_nvfp4: self.lm_head_input.saved(),
            logits,
        }
    }

    pub fn tape(&mut self) -> Gpt2ForwardTape<'_> {
        let blocks = self.blocks.as_mut_ptr();
        Gpt2ForwardTape {
            blocks: std::array::from_fn(|i| unsafe { (&mut *blocks.add(i)).tape() }),
            final_norm: self.final_norm.tape(),
            lm_head_input_nvfp4: self.lm_head_input.tape(),
        }
    }

    pub fn block_qkv(&self, index: usize) -> &DeviceBuffer<u16> {
        self.blocks[index].qkv()
    }
}
