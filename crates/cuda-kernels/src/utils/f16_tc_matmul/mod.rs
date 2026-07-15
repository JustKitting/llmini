#[macro_use]
mod macros;

mod args;
pub(crate) mod convert;
mod cta;
mod cta_add_f32;
mod cta_add_f32_rhs_transposed_base;
mod cta_f32;
mod cta_f32_a_transposed_half_rhs;
mod cta_f32_a_transposed_rhs;
mod cta_f32_accumulate;
mod cta_f32_causal;
mod cta_f32_half_rhs;
mod cta_f32_rhs;
mod cta_half_rhs;
mod cta_lower_ds;
pub(crate) mod cta_stage;
mod cta_stage_f32;
mod cta_stage_f32_transposed;
mod cta_stage_half;
mod cta_store;
mod cta_store_accumulate;
mod cta_store_add;
mod cta_store_causal;
mod cta_sync;
pub(crate) mod cta_tile;
mod kernels;
mod launch_ops;
mod launcher;
mod launcher_add;
mod launcher_f32;
mod pad;
mod prepare;

type CtaATile = cuda_device::SharedArray<u16, { cta_tile::CTA_A_ELEMS }>;
type CtaBTile = cuda_device::SharedArray<u16, { cta_tile::CTA_B_ELEMS }>;

pub use args::{
    F16ConvertArgs, F16TcMatmulAddArgs, F16TcMatmulAddRhsTransposeBaseArgs, F16TcMatmulArgs,
    F16TcMatmulF32ATransposedHalfRhsArgs, F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args,
    F16TcMatmulF32HalfRhsArgs, F16TcMatmulF32RhsArgs, F16TcMatmulHalfArgs, F16TcMatmulHalfDsArgs,
    F16TcMatmulHalfRhsArgs, F16TcMatmulScratch, f16_tc_matmul_elements, f16_tc_matmul_padded_k,
};
pub use launcher::F16TcMatmulModule;
