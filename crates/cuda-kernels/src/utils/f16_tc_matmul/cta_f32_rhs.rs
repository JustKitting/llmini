cta_rhs_matmul_body_fn!(
    cta_matmul_f32_rhs_body,
    rhs: f32,
    super::cta_stage_f32::stage_tiles_f32_rhs_transposed
);

cta_rhs_matmul_body_fn!(@typed cta_matmul_half_a_f32_rhs_body,u16,rhs:f32,f32,super::cta_stage_f32::stage_tiles_half_a_f32_rhs,super::cta_store::store);
cta_rhs_matmul_body_fn!(@typed cta_matmul_half_a_f32_rhs_half_out_body,u16,rhs:f32,u16,super::cta_stage_f32::stage_tiles_half_a_f32_rhs,super::cta_store::store_half);
