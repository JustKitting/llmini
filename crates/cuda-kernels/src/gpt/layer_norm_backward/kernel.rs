use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::float_ptx::{abs_f32, max_f32};
use crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
use crate::layer_norm_utils::f16_column as xhat_f16_column;
use crate::layer_norm_utils::{f16_column, f32_column, nvfp4_vector_column as nvfp4_column};
use crate::warp_reduce::{thread_lane_warp, warp_max_f32, warp_sum_f32};

pub const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / WARP_SIZE;

// Both sums use the same warp/CTA reduction order and share synchronization.
#[inline(always)]
fn block_sum_pair(a: f32, b: f32, storage: *mut u64, lane: u32, warp: u32) -> (f32, f32) {
    let a = warp_sum_f32(a);
    let b = warp_sum_f32(b);
    if lane == 0 {
        unsafe {
            *storage.add(warp as usize) = u64::from(a.to_bits()) | (u64::from(b.to_bits()) << 32);
        }
    }
    thread::sync_threads();
    if warp == 0 {
        let bits = if lane < WARPS_PER_BLOCK {
            unsafe { *storage.add(lane as usize) }
        } else {
            0
        };
        let a = warp_sum_f32(f32::from_bits(bits as u32));
        let b = warp_sum_f32(f32::from_bits((bits >> 32) as u32));
        if lane == 0 {
            unsafe {
                *storage = u64::from(a.to_bits()) | (u64::from(b.to_bits()) << 32);
            }
        }
    }
    thread::sync_threads();
    let bits = unsafe { *storage };
    (
        f32::from_bits(bits as u32),
        f32::from_bits((bits >> 32) as u32),
    )
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(super) mod kernels {
    use super::*;

    macro_rules! maybe_store_input_amax {
        (none; $($ignored:ident),+) => {};
        ($chunk_amax:ident; $local_amax:ident, $row:ident, $lane:ident, $warp_in_block:ident) => {{
            let row_amax = layer_norm_block_reduce!(
                WARP_SUMS,
                WARPS_PER_BLOCK,
                $local_amax,
                $lane,
                $warp_in_block,
                warp_max_f32
            );
            layer_norm_store_row!(&mut $chunk_amax, $row, $lane, $warp_in_block, row_amax);
        }};
    }

    macro_rules! input_xhat {
        (xhat_f16_column,$x:expr,$mean:expr,$inv:expr) => {
            $x
        };
        ($reader:ident,$x:expr,$mean:expr,$inv:expr) => {
            ($x - $mean) * $inv
        };
    }
    macro_rules! residual_quad {
        (xhat_f16_column,$x:ident,$base:expr) => {
            crate::layer_norm_utils::load_f16_quad($x, $base)
        };
        (f16_column,$x:ident,$base:expr) => {
            crate::layer_norm_utils::load_f16_quad($x, $base)
        };
        (f32_column,$x:ident,$base:expr) => {
            crate::layer_norm_utils::load_f32_quad($x, $base)
        };
    }

    macro_rules! layer_norm_backward_input_body {
        (
            $residual_column:ident;
            $residual:ident $d_normalized:ident $mean:ident $inv_std:ident;
            $weight_bytes:ident $weight_scales:ident $weight_global_scale:ident;
            $d_residual:ident $output_scale:ident $row_count:ident $embedding_dim:ident;
            $direct:expr;
            $chunk_amax:ident
        ) => {{
            static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
                SharedArray::UNINIT;
            static mut PAIR_SUMS:SharedArray<u64,{WARPS_PER_BLOCK as usize},8>=SharedArray::UNINIT;

            let row = thread::blockIdx_x();
            let (thread, lane, warp_in_block) = thread_lane_warp();

            if row < $row_count {
                let row_base = row as usize * $embedding_dim as usize;
                let row_mean = $mean[row as usize];
                let row_inv_std = $inv_std[row as usize];
                if $embedding_dim==2048{
                    use crate::layer_norm_utils::{load_f32_quad,store_f32_quad};
                    let base0=row_base+thread as usize*4;let base1=base0+1024;
                    let residual0=residual_quad!($residual_column,$residual,base0);
                    let residual1=residual_quad!($residual_column,$residual,base1);
                    let gradient0=load_f32_quad($d_normalized,base0);let gradient1=load_f32_quad($d_normalized,base1);
                    let weight0=crate::nvfp4::nvfp4_vector_values4($weight_bytes,$weight_scales,$weight_global_scale[0],thread as usize*4);
                    let weight1=crate::nvfp4::nvfp4_vector_values4($weight_bytes,$weight_scales,$weight_global_scale[0],thread as usize*4+1024);
                    let mut dxhat_local=0.0f32;let mut xhat_dxhat_local=0.0f32;
                    let xhat_0=input_xhat!($residual_column,residual0[0],row_mean,row_inv_std);
                    let dxhat_0=gradient0[0]*weight0[0]*$output_scale;
                    dxhat_local+=dxhat_0;xhat_dxhat_local+=xhat_0*dxhat_0;
                    let xhat_1=input_xhat!($residual_column,residual0[1],row_mean,row_inv_std);
                    let dxhat_1=gradient0[1]*weight0[1]*$output_scale;
                    dxhat_local+=dxhat_1;xhat_dxhat_local+=xhat_1*dxhat_1;
                    let xhat_2=input_xhat!($residual_column,residual0[2],row_mean,row_inv_std);
                    let dxhat_2=gradient0[2]*weight0[2]*$output_scale;
                    dxhat_local+=dxhat_2;xhat_dxhat_local+=xhat_2*dxhat_2;
                    let xhat_3=input_xhat!($residual_column,residual0[3],row_mean,row_inv_std);
                    let dxhat_3=gradient0[3]*weight0[3]*$output_scale;
                    dxhat_local+=dxhat_3;xhat_dxhat_local+=xhat_3*dxhat_3;
                    let xhat_4=input_xhat!($residual_column,residual1[0],row_mean,row_inv_std);
                    let dxhat_4=gradient1[0]*weight1[0]*$output_scale;
                    dxhat_local+=dxhat_4;xhat_dxhat_local+=xhat_4*dxhat_4;
                    let xhat_5=input_xhat!($residual_column,residual1[1],row_mean,row_inv_std);
                    let dxhat_5=gradient1[1]*weight1[1]*$output_scale;
                    dxhat_local+=dxhat_5;xhat_dxhat_local+=xhat_5*dxhat_5;
                    let xhat_6=input_xhat!($residual_column,residual1[2],row_mean,row_inv_std);
                    let dxhat_6=gradient1[2]*weight1[2]*$output_scale;
                    dxhat_local+=dxhat_6;xhat_dxhat_local+=xhat_6*dxhat_6;
                    let xhat_7=input_xhat!($residual_column,residual1[3],row_mean,row_inv_std);
                    let dxhat_7=gradient1[3]*weight1[3]*$output_scale;
                    dxhat_local+=dxhat_7;xhat_dxhat_local+=xhat_7*dxhat_7;
                    let pair_storage=unsafe{(&mut *(&raw mut PAIR_SUMS)).as_mut_ptr()};
                    let(dxhat_sum,xhat_dxhat_sum)=block_sum_pair(dxhat_local,xhat_dxhat_local,pair_storage,lane,warp_in_block);
                    let inv_dim=1.0/$embedding_dim as f32;let direct:*const f32=$direct;let mut local_amax=0.0f32;
                    let d0=if direct.is_null(){[0.0;4]}else{unsafe{crate::layer_norm_utils::load_f32_quad_raw(direct,base0)}};
                    let dx_0=(dxhat_0-dxhat_sum*inv_dim-xhat_0*xhat_dxhat_sum*inv_dim)*row_inv_std+d0[0];
                    local_amax=max_f32(local_amax,abs_f32(dx_0));
                    let dx_1=(dxhat_1-dxhat_sum*inv_dim-xhat_1*xhat_dxhat_sum*inv_dim)*row_inv_std+d0[1];
                    local_amax=max_f32(local_amax,abs_f32(dx_1));
                    let dx_2=(dxhat_2-dxhat_sum*inv_dim-xhat_2*xhat_dxhat_sum*inv_dim)*row_inv_std+d0[2];
                    local_amax=max_f32(local_amax,abs_f32(dx_2));
                    let dx_3=(dxhat_3-dxhat_sum*inv_dim-xhat_3*xhat_dxhat_sum*inv_dim)*row_inv_std+d0[3];
                    local_amax=max_f32(local_amax,abs_f32(dx_3));
                    store_f32_quad(&mut $d_residual,base0,[dx_0,dx_1,dx_2,dx_3]);
                    let d1=if direct.is_null(){[0.0;4]}else{unsafe{crate::layer_norm_utils::load_f32_quad_raw(direct,base1)}};
                    let dx_4=(dxhat_4-dxhat_sum*inv_dim-xhat_4*xhat_dxhat_sum*inv_dim)*row_inv_std+d1[0];
                    local_amax=max_f32(local_amax,abs_f32(dx_4));
                    let dx_5=(dxhat_5-dxhat_sum*inv_dim-xhat_5*xhat_dxhat_sum*inv_dim)*row_inv_std+d1[1];
                    local_amax=max_f32(local_amax,abs_f32(dx_5));
                    let dx_6=(dxhat_6-dxhat_sum*inv_dim-xhat_6*xhat_dxhat_sum*inv_dim)*row_inv_std+d1[2];
                    local_amax=max_f32(local_amax,abs_f32(dx_6));
                    let dx_7=(dxhat_7-dxhat_sum*inv_dim-xhat_7*xhat_dxhat_sum*inv_dim)*row_inv_std+d1[3];
                    local_amax=max_f32(local_amax,abs_f32(dx_7));
                    store_f32_quad(&mut $d_residual,base1,[dx_4,dx_5,dx_6,dx_7]);
                    maybe_store_input_amax!($chunk_amax;local_amax,row,lane,warp_in_block);
                    return;
                }
                let mut dxhat_local = 0.0f32;
                let mut xhat_dxhat_local = 0.0f32;
                let mut col = thread;
                while col < $embedding_dim {
                    let xhat = input_xhat!($residual_column,$residual_column(
                        $residual,
                        row_base,
                        col,
                        $embedding_dim,
                    ),row_mean,row_inv_std);
                    let grad = f32_column($d_normalized, row_base, col, $embedding_dim);
                    let weight = nvfp4_column(
                        $weight_bytes,
                        $weight_scales,
                        $weight_global_scale[0],
                        0,
                        col,
                        $embedding_dim,
                    );
                    let dxhat = grad * weight * $output_scale;
                    dxhat_local += dxhat;
                    xhat_dxhat_local += xhat * dxhat;
                    col += THREADS_PER_BLOCK;
                }
                let dxhat_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    dxhat_local,
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );
                let xhat_dxhat_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    xhat_dxhat_local,
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );
                let inv_dim = 1.0 / $embedding_dim as f32;
                let direct: *const f32 = $direct;
                let mut local_amax = 0.0f32;
                let mut col = thread;
                while col < $embedding_dim {
                    let xhat = input_xhat!($residual_column,$residual_column(
                        $residual,
                        row_base,
                        col,
                        $embedding_dim,
                    ),row_mean,row_inv_std);
                    let grad = f32_column($d_normalized, row_base, col, $embedding_dim);
                    let weight = nvfp4_column(
                        $weight_bytes,
                        $weight_scales,
                        $weight_global_scale[0],
                        0,
                        col,
                        $embedding_dim,
                    );
                    let dxhat = grad * weight * $output_scale;
                    let mut dx = (dxhat
                        - dxhat_sum * inv_dim
                        - xhat * xhat_dxhat_sum * inv_dim)
                        * row_inv_std;
                    if !direct.is_null() {
                        dx += unsafe { *direct.add(row_base + col as usize) };
                    }
                    unsafe {
                        *$d_residual.get_unchecked_mut(row_base + col as usize) = dx;
                    }
                    local_amax = max_f32(local_amax, abs_f32(dx));
                    col += THREADS_PER_BLOCK;
                }
                maybe_store_input_amax!(
                    $chunk_amax;
                    local_amax,
                    row,
                    lane,
                    warp_in_block
                );
            }
        }};
    }

    #[kernel]
    pub fn layer_norm_backward_input_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            none
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_amax_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        mut chunk_amax: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            chunk_amax
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_add_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        direct: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            direct.as_ptr();
            none
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_add_amax_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        direct: &[f32],
        mut d_residual: DisjointSlice<f32>,
        mut chunk_amax: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            direct.as_ptr();
            chunk_amax
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_f32_kernel(
        residual: &[f32],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f32_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            none
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_xhat_f16_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            xhat_f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            none
        );
    }
    #[kernel]
    pub fn layer_norm_backward_input_amax_xhat_f16_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        mut chunk_amax: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            xhat_f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            chunk_amax
        );
    }
    #[kernel]
    pub fn layer_norm_backward_input_add_xhat_f16_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        direct: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            xhat_f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            direct.as_ptr();
            none
        );
    }
    #[kernel]
    pub fn layer_norm_backward_input_add_amax_xhat_f16_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        direct: &[f32],
        mut d_residual: DisjointSlice<f32>,
        mut chunk_amax: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            xhat_f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            direct.as_ptr();
            chunk_amax
        );
    }
}
