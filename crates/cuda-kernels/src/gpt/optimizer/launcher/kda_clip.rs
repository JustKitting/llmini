use cuda_core::DriverError;

use super::super::args::KdaMuonClipArgs;
use super::super::kda_clip::KDA_CLIP_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::grid_x_config;

impl OptimizerModule {
    pub fn apply_kda_muon_clip(&self, args: KdaMuonClipArgs<'_>) -> Result<(), DriverError> {
        let len = args.input_dim * args.qkv_dim;
        assert_eq!(len as usize % 16, 0);
        assert!(args.qkv.len() >= args.row_count as usize * args.qkv_dim as usize);
        assert!(args.z_master.len() >= len as usize);
        assert!(args.x_master.len() >= len as usize);
        assert!(args.momentum.len() >= len as usize);
        assert!(args.scores.len() >= args.head_count as usize);
        assert!(args.bytes.len() >= len as usize / 2);
        assert!(args.scales.len() >= len as usize / 16);

        self.apply.kda_clip.kda_muon_qk_clip_kernel(
            args.stream,
            grid_x_config(args.head_count, KDA_CLIP_THREADS_PER_BLOCK),
            args.qkv,
            args.z_master,
            args.x_master,
            args.momentum,
            args.scores,
            args.row_count,
            args.qkv_dim,
            args.input_dim,
            args.embedding_dim,
            args.head_count,
            args.head_dim,
            args.tau,
            args.silu_qk,
        )?;

        self.requantize(
            args.stream,
            args.bytes,
            args.scales,
            args.global_scale,
            &*args.x_master,
            args.amax,
            args.chunk_amax,
            len,
        )
    }
}
