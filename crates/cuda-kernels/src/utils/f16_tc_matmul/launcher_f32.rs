use super::args::{
    F16TcMatmulHalfCausalHalfOutArgs, F16TcMatmulKdaInverseArgs, F16TcMatmulKdaWuArgs,
};
use cuda_core::DriverError;

use super::args::{
    F16TcMatmulF32ATransposedHalfRhsArgs, F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args,
    F16TcMatmulF32CausalHalfOutArgs, F16TcMatmulF32HalfArgs, F16TcMatmulF32HalfRhsArgs,
    F16TcMatmulF32RhsArgs, F16TcMatmulF32WindowArgs, F16TcMatmulHalfAF32RhsArgs,
    F16TcMatmulHalfAF32RhsHalfOutArgs, F16TcMatmulHalfATransposedDualRhsArgs,
    F16TcMatmulHalfATransposedF32RhsArgs, F16TcMatmulHalfArgs,
};
use super::launcher::{F16TcMatmulModule, cta_config, cta_narrow_config, elements};

macro_rules! f32_matmul_launcher {
    ($method:ident, $args:ty, $rhs:ident, $kernel:ident, a($a_rows:ident, $a_cols:ident), rhs($rhs_rows:ident, $rhs_cols:ident)) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            assert!(args.a.len() >= elements(args.batch_count, args.$a_rows, args.$a_cols));
            assert!(args.$rhs.len() >= elements(args.batch_count, args.$rhs_rows, args.$rhs_cols));
            assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
            self.module.$kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.$rhs,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
            )
        }
    };
}

impl F16TcMatmulModule {
    pub fn kda_native_inverse(
        &self,
        args: F16TcMatmulKdaInverseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let count = elements(args.batch_count, 64, 64);
        assert!(args.kpos_u.len() >= count && args.kneg.len() >= count && args.out.len() >= count);
        self.module.f16_kda_native_inverse_kernel(
            args.stream,
            cta_narrow_config(64, 64, args.batch_count),
            args.kpos_u,
            args.kneg,
            args.out,
            args.batch_count,
        )
    }

    f32_matmul_launcher!(
        batched_matmul_half_causal_half_out,
        F16TcMatmulHalfCausalHalfOutArgs<'_, '_>,
        b_t,
        f16_cta_tc_matmul_half_causal_half_out_kernel,
        a(m, k),
        rhs(n, k)
    );

    pub fn kda_wu_inplace(&self, args: F16TcMatmulKdaWuArgs<'_, '_>) -> Result<(), DriverError> {
        let count = elements(args.batch_count, 64, 64);
        assert!(
            args.inverse.len() >= count
                && args.vbeta.len() >= count
                && args.kpos_u.len() >= count
                && args.w.len() >= count
        );
        self.module.f16_kda_wu_inplace_kernel(
            args.stream,
            cta_config(64, 64, args.batch_count),
            args.inverse,
            args.vbeta,
            args.kpos_u,
            args.w,
            args.batch_count,
        )
    }

    pub fn batched_kda_fused_inverse(
        &self,
        args: F16TcMatmulF32CausalHalfOutArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!((args.m, args.n, args.k), (64, 64, 64));
        assert!(args.a.len() >= elements(args.batch_count, 64, 64));
        assert!(args.b_t.len() >= elements(args.batch_count, 64, 64));
        assert!(args.out.len() >= elements(args.batch_count, 64, 64));
        self.module.f16_kda_fused_inverse_kernel(
            args.stream,
            cta_narrow_config(64, 64, args.batch_count),
            args.a,
            args.b_t,
            args.out,
            args.batch_count,
            64,
            64,
            64,
        )
    }
    f32_matmul_launcher!(
        batched_matmul_half_a_f32_rhs,
        F16TcMatmulHalfAF32RhsArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_half_a_f32_rhs_kernel,
        a(m, k),
        rhs(k, n)
    );
    f32_matmul_launcher!(
        batched_matmul_half_a_f32_rhs_half_out,
        F16TcMatmulHalfAF32RhsHalfOutArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_half_a_f32_rhs_half_out_kernel,
        a(m, k),
        rhs(k, n)
    );

    f32_matmul_launcher!(
        batched_matmul_f32_causal_half_out,
        F16TcMatmulF32CausalHalfOutArgs<'_, '_>,
        b_t,
        f16_cta_tc_matmul_f32_causal_half_out_kernel,
        a(m, k),
        rhs(n, k)
    );
    f32_matmul_launcher!(
        batched_matmul_half_a_transposed_f32_rhs,
        F16TcMatmulHalfATransposedF32RhsArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_half_a_transposed_f32_rhs_kernel,
        a(k, m),
        rhs(k, n)
    );

    f32_matmul_launcher!(
        batched_matmul_f32_input,
        F16TcMatmulF32Args<'_, '_>,
        b_t,
        f16_cta_tc_matmul_f32_kernel,
        a(m, k),
        rhs(n, k)
    );
    pub fn batched_matmul_f32_input_accumulate(
        &self,
        args: F16TcMatmulF32Args<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module.f16_cta_tc_matmul_f32_accumulate_kernel(
            args.stream,
            cta_narrow_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_half_input_accumulate(
        &self,
        args: F16TcMatmulHalfArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module.f16_cta_tc_matmul_half_accumulate_kernel(
            args.stream,
            cta_narrow_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }
    f32_matmul_launcher!(
        batched_matmul_f32_half_input_causal,
        F16TcMatmulF32HalfArgs<'_, '_>,
        b_t,
        f16_cta_tc_matmul_f32_half_causal_kernel,
        a(m, k),
        rhs(n, k)
    );

    pub fn batched_matmul_f32_input_lower(
        &self,
        args: F16TcMatmulF32Args<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.n);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module.f16_cta_tc_matmul_f32_lower_kernel(
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

    pub fn batched_matmul_f32_input_windowed_lower(
        &self,
        args: F16TcMatmulF32WindowArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.n);
        super::launcher::assert_window(args.window, args.m);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module.f16_cta_tc_matmul_f32_windowed_lower_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
            args.window,
        )
    }

    f32_matmul_launcher!(
        batched_matmul_f32_rhs,
        F16TcMatmulF32RhsArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_f32_rhs_kernel,
        a(m, k),
        rhs(k, n)
    );

    f32_matmul_launcher!(
        batched_matmul_f32_half_rhs,
        F16TcMatmulF32HalfRhsArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_f32_half_rhs_kernel,
        a(m, k),
        rhs(k, n)
    );

    pub fn batched_matmul_f32_half_rhs_lower_a(
        &self,
        args: F16TcMatmulF32HalfRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.k);
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module.f16_cta_tc_matmul_f32_half_rhs_lower_a_kernel(
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

    f32_matmul_launcher!(
        batched_matmul_f32_a_transposed_rhs,
        F16TcMatmulF32ATransposedRhsArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_f32_a_transposed_rhs_kernel,
        a(k, m),
        rhs(k, n)
    );

    pub fn batched_matmul_half_a_transposed_dual_rhs(
        &self,
        args: F16TcMatmulHalfATransposedDualRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.k, args.m));
        assert!(args.rhs0.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.rhs1.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out0.len() >= elements(args.batch_count, args.m, args.n));
        assert!(args.out1.len() >= elements(args.batch_count, args.m, args.n));
        self.module
            .f16_cta_tc_matmul_half_a_transposed_dual_rhs_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs0,
                args.rhs1,
                args.out0,
                args.out1,
                args.batch_count,
                args.m,
                args.n,
                args.k,
            )
    }

    f32_matmul_launcher!(
        batched_matmul_f32_a_transposed_half_rhs,
        F16TcMatmulF32ATransposedHalfRhsArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_f32_a_transposed_half_rhs_kernel,
        a(k, m),
        rhs(k, n)
    );

    pub fn batched_matmul_f32_a_transposed_half_rhs_lower_a(
        &self,
        args: F16TcMatmulF32ATransposedHalfRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.m, args.k);
        assert!(args.a.len() >= elements(args.batch_count, args.k, args.m));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
        self.module
            .f16_cta_tc_matmul_f32_a_transposed_half_rhs_lower_a_kernel(
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
}
