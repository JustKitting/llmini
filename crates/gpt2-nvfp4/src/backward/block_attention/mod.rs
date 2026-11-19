mod run;
mod types;

pub use run::attention_side_backward;
pub(crate) use run::attention_side_backward_at_depth;
pub use types::{
    BlockAttentionBackwardArgs, BlockAttentionBackwardModules, BlockAttentionBackwardScratch,
    BlockAttentionBackwardSeeds,
    ValueRmsBackwardScratch,
};
