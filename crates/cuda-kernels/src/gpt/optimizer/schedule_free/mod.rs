mod amax;
mod four_six;
mod value;

use cuda_device::cuda_module;

pub(super) const SCALE_OVERRIDE: f32 = 1.0;

#[cuda_module]
pub(super) mod module {
    use cuda_device::{DisjointSlice, kernel};

    use super::super::symexp_lin::ScalePointers;
    use super::amax::schedule_free_chunk_amax_body;
    use super::four_six::schedule_free_four_six_body;

    #[kernel]
    pub fn schedule_free_chunk_amax_kernel(
        z_master: &[f32],
        x_master: &[f32],
        mut out: DisjointSlice<f32>,
        beta: f32,
        symexp_lin_beta: f32,
        exponential_z: u64,
        exponential_x: u64,
        linear_z: u64,
        linear_x: u64,
        curvature_z: u64,
        curvature_x: u64,
        len: u32,
    ) {
        schedule_free_chunk_amax_body(
            z_master,
            x_master,
            &mut out,
            beta,
            symexp_lin_beta,
            ScalePointers {
                exponential_z,
                exponential_x,
                linear_z,
                linear_x,
                curvature_z,
                curvature_x,
            },
            len,
        );
    }

    #[kernel]
    pub fn schedule_free_four_six_kernel(
        z_master: &[f32],
        x_master: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        beta: f32,
        symexp_lin_beta: f32,
        exponential_z: u64,
        exponential_x: u64,
        linear_z: u64,
        linear_x: u64,
        curvature_z: u64,
        curvature_x: u64,
    ) {
        schedule_free_four_six_body(
            z_master,
            x_master,
            amax,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scale,
            beta,
            symexp_lin_beta,
            ScalePointers {
                exponential_z,
                exponential_x,
                linear_z,
                linear_x,
                curvature_z,
                curvature_x,
            },
        );
    }
}
