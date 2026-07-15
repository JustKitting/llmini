use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::args::{
    F32AddScaledIdentityArgs, F32Linear2Args, F32Linear3Args, F32Linear3SqrtBoundArgs,
    F32Linear3SqrtBoundRowSumsqArgs, F32ScaleInPlaceByAmaxArgs,
};
use super::kernels;
use crate::launch::{grid_x_config, linear_config};

const F32_OPS_THREADS_PER_BLOCK: u32 = 256;

pub struct F32MatrixOpsModule {
    module: kernels::module::LoadedModule,
}

impl F32MatrixOpsModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::module::from_module(module)?,
        })
    }

    pub fn linear2(&self, args: F32Linear2Args<'_, '_>) -> Result<(), DriverError> {
        assert!(args.a.len() >= args.len as usize);
        assert!(args.b.len() >= args.len as usize);
        assert!(args.out.len() >= args.len as usize);

        self.module.f32_linear2_kernel(
            args.stream,
            linear_config(args.len, F32_OPS_THREADS_PER_BLOCK),
            args.a,
            args.b,
            args.out,
            args.len,
            args.a_scale,
            args.b_scale,
        )
    }

    pub fn linear3(&self, args: F32Linear3Args<'_, '_>) -> Result<(), DriverError> {
        assert!(args.a.len() >= args.len as usize);
        assert!(args.b.len() >= args.len as usize);
        assert!(args.c_out.len() >= args.len as usize);

        self.module.f32_linear3_in_place_kernel(
            args.stream,
            linear_config(args.len, F32_OPS_THREADS_PER_BLOCK),
            args.a,
            args.b,
            args.c_out,
            args.len,
            args.a_scale,
            args.b_scale,
            args.c_scale,
        )
    }

    pub fn linear3_sqrt_bound_a(
        &self,
        args: F32Linear3SqrtBoundArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= args.len as usize);
        assert!(args.b.len() >= args.len as usize);
        assert!(args.c_out.len() >= args.len as usize);
        assert!(!args.bound_amax.is_empty());

        self.module.f32_linear3_sqrt_bound_a_in_place_kernel(
            args.stream,
            linear_config(args.len, F32_OPS_THREADS_PER_BLOCK),
            args.a,
            args.b,
            args.c_out,
            args.bound_amax,
            args.len,
            args.a_scale,
            args.b_scale,
            args.c_scale,
        )
    }

    pub fn linear3_sqrt_bound_a_row_sumsq(
        &self,
        args: F32Linear3SqrtBoundRowSumsqArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let len = args.rows as usize * args.cols as usize;
        assert!(args.a.len() >= len);
        assert!(args.b.len() >= len);
        assert!(args.c_out.len() >= len);
        assert!(!args.bound_amax.is_empty());
        assert!(args.row_sumsq.len() >= args.rows as usize);

        self.module
            .f32_linear3_sqrt_bound_a_row_sumsq_in_place_kernel(
                args.stream,
                grid_x_config(args.rows, F32_OPS_THREADS_PER_BLOCK),
                args.a,
                args.b,
                args.c_out,
                args.bound_amax,
                args.row_sumsq,
                args.rows,
                args.cols,
                args.a_scale,
                args.b_scale,
                args.c_scale,
            )
    }

    pub fn add_scaled_identity(
        &self,
        args: F32AddScaledIdentityArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let len = args.dim * args.dim;
        assert!(args.src.len() >= len as usize);
        assert!(args.out.len() >= len as usize);

        self.module.f32_add_scaled_identity_kernel(
            args.stream,
            linear_config(len, F32_OPS_THREADS_PER_BLOCK),
            args.src,
            args.out,
            args.dim,
            args.scale,
        )
    }

    pub fn scale_in_place_by_sqrt_amax_bound(
        &self,
        args: F32ScaleInPlaceByAmaxArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.x.len() >= args.len as usize);
        assert!(!args.amax.is_empty());

        self.module.f32_scale_in_place_by_sqrt_amax_bound_kernel(
            args.stream,
            linear_config(args.len, F32_OPS_THREADS_PER_BLOCK),
            args.x,
            args.amax,
            args.len,
        )
    }

    pub fn scale_in_place_by_amax_bound(
        &self,
        args: F32ScaleInPlaceByAmaxArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.x.len() >= args.len as usize);
        assert!(!args.amax.is_empty());

        self.module.f32_scale_in_place_by_amax_bound_kernel(
            args.stream,
            linear_config(args.len, F32_OPS_THREADS_PER_BLOCK),
            args.x,
            args.amax,
            args.len,
        )
    }
}
