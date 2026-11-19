use cuda_device::{SharedArray, thread};

use crate::f16_tc_matmul::convert::{load_f16_global_bits_read_only, store_f16x2_shared};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CtaTile};
use crate::kda_tc::{CtaATile, CtaBTile, swizzle_tc_offset};

const M_STRIDE: usize = 66;

#[inline(always)]
pub(crate) fn stage_chain_native<const TRANSPOSED: bool>(
    src: &[u16],
    dst: &mut CtaATile,
    base: usize,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / 16;
        let k = k_base + pair % 16;
        let bits = if TRANSPOSED {
            u32::from(load_f16_global_bits_read_only(
                src.as_ptr(),
                base + (k * 64 + row) as usize,
            )) | (u32::from(load_f16_global_bits_read_only(
                src.as_ptr(),
                base + ((k + 1) * 64 + row) as usize,
            )) << 16)
        } else {
            crate::f16_tc_matmul::convert::load_f16x2_global_bits_read_only(
                src.as_ptr(),
                base + (row * 64 + k) as usize,
            )
        };
        store_f16x2_shared(dst.as_mut_ptr(), swizzle_tc_offset(pair) as usize, bits);
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
pub(crate) fn stage_chain_shared<const TRANSPOSED: bool>(
    src: *const u16,
    dst: &mut CtaATile,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = (pair / 16) as usize;
        let k = (k_base + pair % 16) as usize;
        let bits = unsafe {
            if TRANSPOSED {
                u32::from(*src.add(k * M_STRIDE + row))
                    | (u32::from(*src.add((k + 1) * M_STRIDE + row)) << 16)
            } else {
                *(src.add(row * M_STRIDE + k) as *const u32)
            }
        };
        store_f16x2_shared(dst.as_mut_ptr(), swizzle_tc_offset(pair) as usize, bits);
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
pub(crate) fn stage_chain_vbeta(src: &[f32], dst: &mut CtaBTile, base: usize, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / 16;
        let k = k_base + pair % 16;
        let (lo, hi) = crate::f16_tc_matmul::convert::load_f32x2_global_read_only(
            src.as_ptr(),
            base + (row * 64 + k) as usize,
        );
        store_f16x2_shared(
            dst.as_mut_ptr(),
            swizzle_tc_offset(pair) as usize,
            cuda_device::convert::cvt_f16x2_f32(lo, hi),
        );
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
pub(crate) fn store_chain_half<const STRICT_NEG: bool>(
    acc: [f32; 4],
    dst: *mut u16,
    tile: CtaTile,
    warp_n: u32,
) {
    let mut i = 0;
    while i < 4 {
        let (row, col) = tile.accumulator_coords(warp_n, i);
        // Match the original FP32 strict-negative store before the first
        // backprojection's half rounding, including +0 above the diagonal.
        let lo = if STRICT_NEG {
            if row > col { -acc[i] } else { 0.0 }
        } else {
            acc[i]
        };
        let hi = if STRICT_NEG {
            if row > col + 1 { -acc[i + 1] } else { 0.0 }
        } else {
            acc[i + 1]
        };
        store_f16x2_shared(
            dst,
            row as usize * M_STRIDE + col as usize,
            cuda_device::convert::cvt_f16x2_f32(lo, hi),
        );
        i += 2;
    }
}

#[inline(always)]
pub(crate) fn stage_backprojection<const TRANSPOSED: bool>(
    m: *const u16,
    rhs: &[u16],
    a: &mut CtaATile,
    b: &mut CtaBTile,
    base: usize,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / 16;
        let k = k_base + pair % 16;
        let (lo, hi) = unsafe {
            if TRANSPOSED {
                (
                    *m.add(k as usize * M_STRIDE + row as usize),
                    *m.add((k + 1) as usize * M_STRIDE + row as usize),
                )
            } else {
                (
                    *m.add(row as usize * M_STRIDE + k as usize),
                    *m.add(row as usize * M_STRIDE + k as usize + 1),
                )
            }
        };
        let rhs_lo =
            load_f16_global_bits_read_only(rhs.as_ptr(), base + k as usize * 64 + row as usize);
        let rhs_hi = load_f16_global_bits_read_only(
            rhs.as_ptr(),
            base + (k + 1) as usize * 64 + row as usize,
        );
        let at = swizzle_tc_offset(pair) as usize;
        store_f16x2_shared(a.as_mut_ptr(), at, u32::from(lo) | (u32::from(hi) << 16));
        store_f16x2_shared(
            b.as_mut_ptr(),
            at,
            u32::from(rhs_lo) | (u32::from(rhs_hi) << 16),
        );
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
pub(crate) fn store_backprojection(
    acc: [f32; 4],
    out: &mut SharedArray<f32, 4096>,
    tile: CtaTile,
    warp_n: u32,
) {
    let mut i = 0;
    while i < 4 {
        let (row, col) = tile.accumulator_coords(warp_n, i);
        out[(row * 64 + col) as usize] = acc[i];
        i += 1;
    }
}

// Expand in the kernel MIR so accumulators never acquire an outlined frame.
macro_rules! project_kda_intra_tiles {
    ($inverse:expr, $kneg:expr, $kpos:expr, $vbeta:expr, $du:expr, $dw:expr, $dg:expr, $dkg:expr, $a:expr, $b:expr, $params:expr) => {{
        use $crate::attention::causal_backward_tc::kda::{
            stage_chain_native, stage_chain_shared, stage_chain_vbeta, store_chain_half,
        };
        let (dg, dkg, a, b) = ($dg, $dkg, $a, $b);
        let params = $params;
        let base = ((cuda_device::thread::blockIdx_x() * (params.seq_len / 64)
            + cuda_device::thread::blockIdx_y())
            * 4096) as usize;
        // The same DG bytes own dM, the first inverse product, strict-negative
        // M and finally FP32 dKpos. Every overwrite follows the prior role's
        // complete last read; no global intermediate or extra cast pass.
        let m_half = dg.as_mut_ptr() as *mut u16;
        let tile = $crate::f16_tc_matmul::cta_tile::CtaTile::from_wide_tile(
            cuda_device::thread::threadIdx_x(),
            0,
            0,
            0,
        );
        macro_rules! product {
            ($lo:ident, $hi:ident, $k:ident; $stage_a:block $stage_b:block) => {{
                let mut $k = 0;
                while $k < 64 {
                    $stage_a
                    $stage_b
                    cuda_device::thread::sync_threads();
                    $crate::kda_tc::mma_accumulate_pair(tile, a, b, &mut $lo, &mut $hi);
                    cuda_device::thread::sync_threads();
                    $k += 16;
                }
            }};
        }
        {
            let (mut lo, mut hi) = ([0.0_f32; 4], [0.0_f32; 4]);
            // Retain one FP32 accumulator across DW*Kpos then DU*Vbeta.
            product!(lo, hi, k; {
                stage_chain_native::<false>($dw, a, base, k);
            } {
                stage_chain_native::<false>($kpos, b, base, k);
            });
            product!(lo, hi, k; {
                stage_chain_native::<false>($du, a, base, k);
            } {
                stage_chain_vbeta($vbeta, b, base, k);
            });
            store_chain_half::<false>(lo, m_half, tile, tile.warp_n0);
            store_chain_half::<false>(hi, m_half, tile, tile.warp_n0 + 1);
        }
        cuda_device::thread::sync_threads();
        {
            let (mut lo, mut hi) = ([0.0_f32; 4], [0.0_f32; 4]);
            product!(lo, hi, k; {
                stage_chain_shared::<false>(m_half, a, k);
            } {
                stage_chain_native::<false>($inverse, b, base, k);
            });
            store_chain_half::<false>(lo, m_half, tile, tile.warp_n0);
            store_chain_half::<false>(hi, m_half, tile, tile.warp_n0 + 1);
        }
        cuda_device::thread::sync_threads();
        {
            let (mut lo, mut hi) = ([0.0_f32; 4], [0.0_f32; 4]);
            product!(lo, hi, k; {
                stage_chain_native::<true>($inverse, a, base, k);
            } {
                stage_chain_shared::<true>(m_half, b, k);
            });
            store_chain_half::<true>(lo, m_half, tile, tile.warp_n0);
            store_chain_half::<true>(hi, m_half, tile, tile.warp_n0 + 1);
        }
        cuda_device::thread::sync_threads();
        {
            let (mut acc0, mut acc1) = ([0.0_f32; 4], [0.0_f32; 4]);
            let mut k = 0;
            while k < 64 {
                $crate::attention::causal_backward_tc::kda::stage_backprojection::<false>(
                    m_half, $kneg, a, b, base, k,
                );
                cuda_device::thread::sync_threads();
                $crate::kda_tc::mma_accumulate_pair(tile, a, b, &mut acc0, &mut acc1);
                cuda_device::thread::sync_threads();
                k += 16;
            }
            $crate::attention::causal_backward_tc::kda::store_backprojection(
                acc0,
                dkg,
                tile,
                tile.warp_n0,
            );
            $crate::attention::causal_backward_tc::kda::store_backprojection(
                acc1,
                dkg,
                tile,
                tile.warp_n0 + 1,
            );
        }
        {
            let (mut acc0, mut acc1) = ([0.0_f32; 4], [0.0_f32; 4]);
            let mut k = 0;
            while k < 64 {
                $crate::attention::causal_backward_tc::kda::stage_backprojection::<true>(
                    m_half, $kpos, a, b, base, k,
                );
                cuda_device::thread::sync_threads();
                $crate::kda_tc::mma_accumulate_pair(tile, a, b, &mut acc0, &mut acc1);
                cuda_device::thread::sync_threads();
                k += 16;
            }
            $crate::attention::causal_backward_tc::kda::store_backprojection(
                acc0,
                dg,
                tile,
                tile.warp_n0,
            );
            $crate::attention::causal_backward_tc::kda::store_backprojection(
                acc1,
                dg,
                tile,
                tile.warp_n0 + 1,
            );
        }
        cuda_device::thread::sync_threads();
    }};
}

pub(crate) use project_kda_intra_tiles;
