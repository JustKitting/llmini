use rust_kernels_cuda::layer_norm::ROW_SIZE;

pub(super) fn sample_row_0(col: usize) -> f32 {
    -3.875 + col as f32 * 0.25
}

pub(super) fn sample_row_1(col: usize) -> f32 {
    -5.3125 + col as f32 * 0.375
}

pub(super) fn reference_layer_norm(
    x: &[f32; ROW_SIZE * 2],
    gamma: &[f32; ROW_SIZE],
    beta: &[f32; ROW_SIZE],
    row_count: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut out = reference_layer_norm_rows(x, row_count, ROW_SIZE, epsilon);
    for row in 0..row_count {
        let base = row * ROW_SIZE;
        for col in 0..ROW_SIZE {
            out[base + col] = out[base + col].mul_add(gamma[col], beta[col]);
        }
    }
    out
}

pub(super) fn reference_layer_norm_rows(
    x: &[f32],
    row_count: usize,
    row_len: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; row_count * row_len];
    for row in 0..row_count {
        let base = row * row_len;
        let mean = gpt_kernel_row_sum(x, base, row_len) / row_len as f32;
        let variance = gpt_kernel_row_variance_sum(x, base, row_len, mean) / row_len as f32;
        let inv_std = 1.0 / (variance + epsilon).sqrt();

        for col in 0..row_len {
            out[base + col] = (x[base + col] - mean) * inv_std;
        }
    }
    out
}

fn gpt_kernel_row_sum(x: &[f32], base: usize, row_len: usize) -> f32 {
    gpt_kernel_row_reduce(x, base, row_len, |value| value)
}

fn gpt_kernel_row_variance_sum(x: &[f32], base: usize, row_len: usize, mean: f32) -> f32 {
    gpt_kernel_row_reduce(x, base, row_len, |value| {
        let centered = value - mean;
        centered * centered
    })
}

fn gpt_kernel_row_reduce(
    x: &[f32],
    base: usize,
    row_len: usize,
    value: impl Fn(f32) -> f32,
) -> f32 {
    gpt_block_reduce_sum(|thread| {
        let mut sum = 0.0;
        for offset in [0, 256, 512] {
            let col = thread + offset;
            if col < row_len {
                sum += value(x[base + col]);
            }
        }
        sum
    })
}

fn gpt_block_reduce_sum(local: impl Fn(usize) -> f32) -> f32 {
    let mut warp_totals = [0.0_f32; 32];
    for (warp, warp_total) in warp_totals.iter_mut().enumerate().take(8) {
        let mut lanes = [0.0_f32; 32];
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = local(warp * 32 + lane);
        }
        *warp_total = warp_sum_lane0(lanes);
    }
    warp_sum_lane0(warp_totals)
}

fn warp_sum_lane0(mut lanes: [f32; 32]) -> f32 {
    for mask in [16, 8, 4, 2] {
        let previous = lanes;
        for lane in 0..32 {
            lanes[lane] += previous[lane ^ mask];
        }
    }
    lanes[0] + lanes[1]
}

pub(super) fn assert_row_amax(out: &[f32], amax: &[f32], row_count: usize, row_len: usize) {
    for (row, actual) in amax.iter().copied().enumerate().take(row_count) {
        let base = row * row_len;
        let expected = out[base..base + row_len]
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let error = (actual - expected).abs();
        assert!(error <= 1.0e-7, "row={row} error={error:.8e}");
    }
}
