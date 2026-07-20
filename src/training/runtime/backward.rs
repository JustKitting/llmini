use gpt2_nvfp4::{
    AttentionBackwardModules, BlockAttentionBackwardModules, BlockMlpBackwardModules,
    FinalHeadBackwardModules, Gpt2BackwardModules, MlpBackwardModules,
};

use super::Runtime;

impl Runtime {
    pub fn backward_modules(&self) -> Gpt2BackwardModules<'_> {
        let linear = AttentionBackwardModules {
            transpose: &self.transpose,
            decode: &self.decode,
            linear: &self.linear,
            quant: &self.quant,
        };
        Gpt2BackwardModules {
            residual: &self.residual,
            final_head: FinalHeadBackwardModules {
                loss: &self.loss,
                transpose: &self.transpose,
                decode: &self.decode,
                linear: &self.linear,
                quant: &self.quant,
            },
            final_norm: &self.layer_norm_backward,
            attention: BlockAttentionBackwardModules {
                residual: &self.residual,
                layer_norm: &self.layer_norm_backward,
                attention: &self.attention,
                canon: &self.canon,
                f16_tc: &self.f16_tc_matmul,
                linear,
            },
            mlp: BlockMlpBackwardModules {
                residual: &self.residual,
                layer_norm: &self.layer_norm_backward,
                canon: &self.canon,
                mlp: MlpBackwardModules {
                    transpose: &self.transpose,
                    decode: &self.decode,
                    linear: &self.linear,
                    quant: &self.quant,
                    mlp: &self.mlp,
                },
            },
        }
    }
}
