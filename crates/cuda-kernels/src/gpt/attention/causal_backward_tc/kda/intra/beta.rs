use cuda_device::DisjointSlice;

use super::{KdaIntraCtx, KdaIntraInputs};
use crate::f16_tc_matmul::convert::load_f32_global_read_only;
use crate::float_ptx::fma_f32;
use crate::kda_common::{kda_decay_exp, safe_denom};

pub(super) fn update_beta_grad(
    inputs: KdaIntraInputs<'_>,
    beta_grad: &mut DisjointSlice<f32>,
    ctx: KdaIntraCtx<'_>,
    tid: u32,
) {
    let token_lane = tid;
    if token_lane < ctx.chunk_tokens {
        let token = ctx.start + token_lane;
        let beta_value = load_f32_global_read_only(inputs.beta.as_ptr(), ctx.beta(token));
        let mut db_value = 0.0;

        let mut dim = 0;
        while dim < ctx.head_dim {
            let compact = ctx.compact(token, dim);
            let g_value = load_f32_global_read_only(inputs.g.as_ptr(), compact);
            let g_last = load_f32_global_read_only(inputs.g.as_ptr(), ctx.last_compact(dim));
            let k_value = load_f32_global_read_only(inputs.kg.as_ptr(), compact)
                * kda_decay_exp(g_value - g_last);
            let exp_g = kda_decay_exp(g_value);

            let d_kpos = load_f32_global_read_only(inputs.d_kpos_m.as_ptr(), compact)
                + load_f32_global_read_only(inputs.d_kneg_from_b.as_ptr(), compact);
            db_value = fma_f32(d_kpos, k_value * exp_g, db_value);
            dim += 1;
        }

        let mut v_dim = 0;
        while v_dim < ctx.head_dim {
            let v_compact = ctx.compact(token, v_dim);
            let v_value = load_f32_global_read_only(inputs.vbeta.as_ptr(), v_compact)
                / safe_denom(beta_value);
            let d_vbeta = load_f32_global_read_only(inputs.d_vbeta_m.as_ptr(), v_compact);
            db_value = fma_f32(d_vbeta, v_value, db_value);
            v_dim += 1;
        }

        unsafe {
            *beta_grad.get_unchecked_mut(ctx.beta(token)) = db_value;
        }
    }
}
