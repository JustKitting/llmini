use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{
    F16ConvertArgs, F16TcMatmulArgs, F16TcMatmulHalfArgs, F16TcMatmulHalfDsArgs,
    F16TcMatmulHalfDsSparseWindowArgs, F16TcMatmulHalfDsWindowArgs, F16TcMatmulHalfRhsArgs,
    F16TcMatmulHalfRhsSparseWindowArgs, F16TcMatmulHalfRhsWindowArgs,
};
use super::cta_tile::{CTA_M, CTA_N, CTA_THREADS, CTA_WIDE_THREADS, CtaMatmulDims};
use super::kernels;
use super::launch_ops::convert;
use super::prepare::prepare_halves;
use crate::launch::launch_config;

pub struct F16TcMatmulModule {
    pub(super) module: kernels::module::LoadedModule,
}

impl F16TcMatmulModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::module::from_module(module)?,
        })
    }

    pub fn fp32_to_f16(&self, args: F16ConvertArgs<'_, '_>) -> Result<(), DriverError> {
        convert(
            &self.module,
            args.stream,
            args.src,
            args.dst,
            args.element_count,
        )
    }

    pub fn batched_matmul(&self, args: F16TcMatmulArgs<'_, '_, '_>) -> Result<(), DriverError> {
        let dims = CtaMatmulDims::new(args.batch_count, args.m, args.n, args.k);
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        let (scratch, k) = prepare_halves(
            &self.module,
            args.stream,
            args.a,
            args.b_t,
            args.scratch,
            dims,
        )?;

        self.module.f16_cta_tc_matmul_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            scratch.a_halves,
            scratch.b_t_halves,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            k,
        )
    }

    pub fn batched_matmul_half_input(
        &self,
        args: F16TcMatmulHalfArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_half_input_lower(
        &self,
        args: F16TcMatmulHalfArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.n);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_lower_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_half_input_lower_ds(
        &self,
        args: F16TcMatmulHalfDsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.n);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.probs.len() >= elements(args.batch_count, args.m, args.n));
        assert!(args.softmax_d.len() >= elements(args.batch_count, args.m, 1));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_lower_ds_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.probs,
            args.softmax_d,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_half_input_windowed_lower_ds(
        &self,
        args: F16TcMatmulHalfDsWindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.n);
        assert_window(args.window, args.m);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.probs.len() >= elements(args.batch_count, args.m, args.n));
        assert!(args.softmax_d.len() >= elements(args.batch_count, args.m, 1));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_windowed_lower_ds_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.probs,
            args.softmax_d,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
            args.window,
        )
    }

    pub fn batched_matmul_half_input_windowed_lower_ds_sparse(
        &self,
        args: F16TcMatmulHalfDsSparseWindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.n);
        assert_window(args.window, args.m);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.probs.len() >= elements(args.batch_count, args.m, args.n));
        assert!(args.softmax_d.len() >= elements(args.batch_count, args.m, 1));
        assert!(args.tile_scales.len() >= sparse_tile_elements(args.batch_count, args.m));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module
            .f16_cta_tc_matmul_windowed_lower_ds_sparse_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.b_t,
                args.probs,
                args.softmax_d,
                args.tile_scales,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
                args.window,
            )
    }

    pub fn batched_matmul_half_rhs_lower_a(
        &self,
        args: F16TcMatmulHalfRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.k);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module.f16_cta_tc_matmul_half_rhs_lower_a_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.rhs,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_half_a_transposed_rhs_lower_a(
        &self,
        args: F16TcMatmulHalfRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.k);
        assert!(args.a.len() >= elements(args.batch_count, args.k, args.m));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module
            .f16_cta_tc_matmul_half_a_transposed_rhs_lower_a_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
            )
    }

    pub fn batched_matmul_half_rhs_windowed_lower_a(
        &self,
        args: F16TcMatmulHalfRhsWindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.k);
        assert_window(args.window, args.k);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module
            .f16_cta_tc_matmul_half_rhs_windowed_lower_a_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
                args.window,
            )
    }

    pub fn batched_matmul_half_a_transposed_rhs_windowed_lower_a(
        &self,
        args: F16TcMatmulHalfRhsWindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.k);
        assert_window(args.window, args.k);
        assert!(args.a.len() >= elements(args.batch_count, args.k, args.m));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module
            .f16_cta_tc_matmul_half_a_transposed_rhs_windowed_lower_a_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
                args.window,
            )
    }

    pub fn batched_matmul_half_rhs_windowed_lower_a_sparse(
        &self,
        args: F16TcMatmulHalfRhsSparseWindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        validate_sparse_half_rhs(&args);
        self.module
            .f16_cta_tc_matmul_half_rhs_windowed_lower_a_sparse_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs,
                args.tile_scales,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
                args.window,
            )
    }

    pub fn batched_matmul_half_a_transposed_rhs_windowed_lower_a_sparse(
        &self,
        args: F16TcMatmulHalfRhsSparseWindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        validate_sparse_half_rhs(&args);
        self.module
            .f16_cta_tc_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs,
                args.tile_scales,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
                args.window,
            )
    }

    pub fn batched_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_scaled(
        &self,
        args: F16TcMatmulHalfRhsSparseWindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        validate_sparse_half_rhs(&args);
        self.module
            .f16_cta_tc_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_scaled_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs,
                args.tile_scales,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
                args.window,
            )
    }
}

fn validate_sparse_half_rhs(args: &F16TcMatmulHalfRhsSparseWindowArgs<'_, '_>) {
    assert_eq!(args.m, args.k);
    assert_window(args.window, args.k);
    assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
    assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
    assert!(args.tile_scales.len() >= sparse_tile_elements(args.batch_count, args.m));
    assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
}

fn sparse_tile_elements(batch_count: u32, seq_len: u32) -> usize {
    let tiles = super::sparse_tile::attention_tile_count(seq_len);
    batch_count as usize * tiles as usize * tiles as usize
}

pub(super) fn cta_config(m: u32, n: u32, batch_count: u32) -> LaunchConfig {
    launch_config(
        (n.div_ceil(CTA_N), m.div_ceil(CTA_M), batch_count),
        CTA_WIDE_THREADS,
    )
}

pub(super) fn cta_narrow_config(m: u32, n: u32, batch_count: u32) -> LaunchConfig {
    launch_config(
        (n.div_ceil(CTA_N), m.div_ceil(CTA_M), batch_count),
        CTA_THREADS,
    )
}

pub(super) fn elements(batch_count: u32, rows: u32, cols: u32) -> usize {
    batch_count as usize * rows as usize * cols as usize
}

pub(super) fn assert_window(window: u32, seq_len: u32) {
    assert!(window > 0 && window <= seq_len);
    assert!(window.is_multiple_of(CTA_M));
}
