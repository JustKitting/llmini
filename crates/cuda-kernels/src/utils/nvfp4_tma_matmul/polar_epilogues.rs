// Fused native-FP4 Muon epilogues. Logical outputs stay compact even when the
// tensor-core operands have padded M/N extents.
#[derive(Clone, Copy)]
struct PolarStepInputs<'a> {
    source: &'a [f32],
    rows: u32,
    cols: u32,
    bound_scale: f32,
    a_scale: f32,
}

#[derive(Clone, Copy)]
struct GramFactorInputs<'a> {
    gram: &'a [f32],
    dim: u32,
    bound_scale: f32,
    b_scale: f32,
    c_scale: f32,
}

#[inline(always)]
fn polar_step_pair(
    inputs: PolarStepInputs<'_>,
    row: u32,
    col: u32,
    current0: f32,
    current1: f32,
) -> Option<(f32, f32)> {
    if row >= inputs.rows || col + 1 >= inputs.cols {
        return None;
    }
    let index = (row * inputs.cols + col) as usize;
    let (s0, s1) = load_f32x2_global_read_only(inputs.source.as_ptr(), index);
    Some((
        fma_f32(inputs.a_scale, s0 * inputs.bound_scale, current0),
        fma_f32(inputs.a_scale, s1 * inputs.bound_scale, current1),
    ))
}

#[inline(always)]
fn store_acc_polar_step_amax(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    _params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    inputs: PolarStepInputs<'_>,
    out: &mut DisjointSlice<f32>,
    local_amax: &mut f32,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    if let Some((v0, v1)) = polar_step_pair(inputs, row0, col, acc[0] * scale0, acc[1] * scale0) {
        *local_amax = max_f32(*local_amax, max_f32(abs_f32(v0), abs_f32(v1)));
        store_f32x2_global(out, row0 * inputs.cols + col, v0, v1);
    }
    if let Some((v0, v1)) = polar_step_pair(inputs, row1, col, acc[2] * scale1, acc[3] * scale1) {
        *local_amax = max_f32(*local_amax, max_f32(abs_f32(v0), abs_f32(v1)));
        store_f32x2_global(out, row1 * inputs.cols + col, v0, v1);
    }
}

#[inline(always)]
fn store_acc_polar_step_row_sumsq(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    _params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    inputs: PolarStepInputs<'_>,
    out: &mut DisjointSlice<f32>,
    sums: &mut [f32; POLAR_ROW_SUMS_PER_THREAD],
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    if let Some((v0, v1)) = polar_step_pair(inputs, row0, col, acc[0] * scale0, acc[1] * scale0) {
        let sum = &mut sums[(m_repeat * 2) as usize];
        *sum = fma_f32(v0, v0, *sum);
        *sum = fma_f32(v1, v1, *sum);
        store_f32x2_global(out, row0 * inputs.cols + col, v0, v1);
    }
    if let Some((v0, v1)) = polar_step_pair(inputs, row1, col, acc[2] * scale1, acc[3] * scale1) {
        let sum = &mut sums[(m_repeat * 2 + 1) as usize];
        *sum = fma_f32(v0, v0, *sum);
        *sum = fma_f32(v1, v1, *sum);
        store_f32x2_global(out, row1 * inputs.cols + col, v0, v1);
    }
}

#[inline(always)]
fn store_gram_factor_pair(
    inputs: GramFactorInputs<'_>,
    row: u32,
    col: u32,
    current0: f32,
    current1: f32,
    out: &mut DisjointSlice<f32>,
    local_amax: &mut f32,
    mirror: bool,
) {
    if row >= inputs.dim || col + 1 >= inputs.dim {
        return;
    }
    let index = row * inputs.dim + col;
    let (g0, g1) = load_f32x2_global_read_only(inputs.gram.as_ptr(), index as usize);
    let v0 = fma_f32(
        inputs.b_scale,
        g0 * inputs.bound_scale,
        inputs.c_scale * current0,
    );
    let v1 = fma_f32(
        inputs.b_scale,
        g1 * inputs.bound_scale,
        inputs.c_scale * current1,
    );
    *local_amax = max_f32(*local_amax, max_f32(abs_f32(v0), abs_f32(v1)));
    store_f32x2_global(out, index, v0, v1);
    if mirror {
        unsafe {
            *out.get_unchecked_mut((col * inputs.dim + row) as usize) = v0;
            *out.get_unchecked_mut(((col + 1) * inputs.dim + row) as usize) = v1;
        }
    }
}

#[inline(always)]
fn store_acc_gram_factor_symmetric_amax(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    _params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    inputs: GramFactorInputs<'_>,
    out: &mut DisjointSlice<f32>,
    local_amax: &mut f32,
    mirror: bool,
) {
    let row = tile.mma_row_base(m_repeat) + tile.group;
    let col = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    store_gram_factor_pair(
        inputs,
        row,
        col,
        acc[0] * scale0,
        acc[1] * scale0,
        out,
        local_amax,
        mirror,
    );
    store_gram_factor_pair(
        inputs,
        row + 8,
        col,
        acc[2] * scale1,
        acc[3] * scale1,
        out,
        local_amax,
        mirror,
    );
}
