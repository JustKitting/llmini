use cuda_core::DriverError;

use super::super::args::GradientClipArgs;
use super::super::grad_clip::GRAD_CLIP_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::grid_x_config;

impl OptimizerModule {
    pub fn clip_gradients(&self, args: GradientClipArgs<'_>) -> Result<(), DriverError> {
        assert!(args.chunk_ptrs.len() >= args.chunk_count as usize);
        assert!(args.chunk_lens.len() >= args.chunk_count as usize);
        assert!(args.chunk_sums.len() >= args.chunk_count as usize);
        assert!(!args.scale.is_empty());
        assert!(!args.norm.is_empty());

        self.apply.grad_clip.grad_clip_sumsq_chunks_kernel(
            args.stream,
            grid_x_config(args.chunk_count, GRAD_CLIP_THREADS_PER_BLOCK),
            args.chunk_ptrs,
            args.chunk_lens,
            args.chunk_sums,
            args.chunk_count,
        )?;

        self.apply.grad_clip.grad_clip_scale_kernel(
            args.stream,
            grid_x_config(1, GRAD_CLIP_THREADS_PER_BLOCK),
            args.chunk_sums,
            args.scale,
            args.norm,
            args.chunk_count,
            args.max_norm,
        )?;

        if args.apply {
            self.apply.grad_clip.grad_clip_apply_kernel(
                args.stream,
                grid_x_config(args.chunk_count, GRAD_CLIP_THREADS_PER_BLOCK),
                args.chunk_ptrs,
                args.chunk_lens,
                args.scale,
                args.chunk_count,
            )?;
        }
        Ok(())
    }
}
