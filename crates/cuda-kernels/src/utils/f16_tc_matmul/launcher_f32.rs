use cuda_core::DriverError;

use super::args::{
    F16TcMatmulF32ATransposedHalfRhsArgs, F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args,
    F16TcMatmulF32HalfRhsArgs, F16TcMatmulF32RhsArgs, F16TcMatmulF32WindowArgs,
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
    f32_matmul_launcher!(
        batched_matmul_f32_input_causal,
        F16TcMatmulF32Args<'_, '_>,
        b_t,
        f16_cta_tc_matmul_f32_causal_kernel,
        a(m, k),
        rhs(n, k)
    );
    f32_matmul_launcher!(
        batched_matmul_f32_input_strict_causal,
        F16TcMatmulF32Args<'_, '_>,
        b_t,
        f16_cta_tc_matmul_f32_strict_causal_kernel,
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
    f32_matmul_launcher!(
        batched_matmul_f32_a_transposed_rhs_strict_neg,
        F16TcMatmulF32ATransposedRhsArgs<'_, '_>,
        rhs,
        f16_cta_tc_matmul_f32_a_transposed_rhs_strict_neg_kernel,
        a(k, m),
        rhs(k, n)
    );
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
