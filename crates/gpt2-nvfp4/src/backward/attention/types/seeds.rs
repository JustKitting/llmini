use crate::Gpt2Rng;

#[derive(Clone, Copy)]
pub struct AttentionBackwardSeeds {
    pub(crate) sign: u32,
    pub(crate) scale: u32,
}

impl AttentionBackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            sign: rng.next_u32(),
            scale: rng.next_u32(),
        }
    }

    pub(crate) fn attention_mask_seed(self) -> u32 {
        self.sign ^ self.scale.rotate_left(13) ^ 0xa511_e9b3
    }
}
