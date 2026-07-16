use cuda_core::DriverError;

use super::super::args::{KdaMuonClipArgs, KdaMuonClipFactorArgs};
use super::super::kda_clip::KDA_CLIP_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::grid_x_config;

impl OptimizerModule {
    pub fn prepare_kda_muon_clip_factor(
        &self,
        args: KdaMuonClipFactorArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.qkv.len() >= args.row_count as usize * args.qkv_dim as usize);
        assert!(args.qk_norm_max.len() >= (args.norm_offset + 2 * args.head_count) as usize);
        assert!(args.scores.len() >= args.head_count as usize);
        assert!(args.factors.len() >= (args.factor_offset + args.head_count) as usize);
        self.apply.kda_clip.kda_muon_qk_clip_factor_kernel(
            args.stream,
            grid_x_config(args.head_count, KDA_CLIP_THREADS_PER_BLOCK),
            args.qkv,
            args.qk_norm_max,
            args.scores,
            args.factors,
            args.row_count,
            args.qkv_dim,
            args.embedding_dim,
            args.head_count,
            args.head_dim,
            args.tau,
            args.silu_qk,
            args.norm_offset,
            args.factor_offset,
            args.precomputed_qk_norms,
        )
    }

    pub fn apply_kda_muon_clip(&self, mut args: KdaMuonClipArgs<'_>) -> Result<(), DriverError> {
        let len = self.apply_kda_muon_clip_master(&mut args)?;
        self.materialize_master(
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

    pub fn apply_kda_muon_clip_deferred_quantization(
        &self,
        mut args: KdaMuonClipArgs<'_>,
    ) -> Result<(), DriverError> {
        self.apply_kda_muon_clip_master(&mut args).map(|_| ())
    }

    fn apply_kda_muon_clip_master(
        &self,
        args: &mut KdaMuonClipArgs<'_>,
    ) -> Result<u32, DriverError> {
        let len = args.input_dim * args.qkv_dim;
        assert_eq!(len as usize % 16, 0);
        assert!(args.qkv.len() >= args.row_count as usize * args.qkv_dim as usize);
        assert!(args.qk_norm_max.len() >= (args.norm_offset + 2 * args.head_count) as usize);
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
            args.qk_norm_max,
            &mut *args.z_master,
            &mut *args.x_master,
            &mut *args.momentum,
            &mut *args.scores,
            args.row_count,
            args.qkv_dim,
            args.input_dim,
            args.embedding_dim,
            args.head_count,
            args.head_dim,
            args.tau,
            args.silu_qk,
            args.norm_offset,
            args.precomputed_qk_norms,
        )?;
        Ok(len)
    }
}
