use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    gradient::GemmGradientTransform,
    launcher::{Nvfp4GemmModule, Nvfp4GemmRoute},
    tma::TmaNvfp4DeviceScaleDescriptors,
};
use rust_kernels_cuda::optimizer::SymExpLinSlotDescriptor;
use std::error::Error;

mod common;

#[test]
#[ignore = "requires generated sm_120a PTX"]
fn fused_dense_gradient_matches_independent_cpu_reference() -> Result<(), Box<dyn Error>> {
    run_case(256, 384, 256, false, true, 0.37)
}

#[test]
#[ignore = "requires generated sm_120a PTX"]
fn fused_sparse_gradient_including_empty_tiles_matches_cpu_reference() -> Result<(), Box<dyn Error>>
{
    run_case(256, 256, 256, true, true, 0.0)
}

#[test]
#[ignore = "requires generated sm_120a PTX"]
fn fused_gradient_without_learned_scales_matches_cpu_reference() -> Result<(), Box<dyn Error>> {
    run_case(128, 128, 128, false, false, 0.9)
}

#[test]
#[ignore = "requires generated sm_120a PTX"]
fn fused_gradient_preserves_inactive_physical_padding() -> Result<(), Box<dyn Error>> {
    run_case(256, 128, 160, false, true, 0.37)
}

fn close(got: f32, want: f64, label: &str, index: usize) {
    assert!(got.is_finite());
    assert!(
        (got as f64 - want).abs() < 2e-4 * (1.0 + want.abs()),
        "{label} index={index} actual={got} expected={want}"
    );
}

fn run_case(
    m: usize,
    n: usize,
    active_rows: usize,
    routed: bool,
    learn: bool,
    schedule_beta: f32,
) -> Result<(), Box<dyn Error>> {
    const K: usize = 256;
    const BETA: f32 = 12.5;
    let (_, stream, ptx) = common::cuda_test_context()?;
    let gemm = Nvfp4GemmModule::from_module(ptx)?;
    let a: Vec<u8> = (0..m * K / 2)
        .map(|i| if (i / (K / 2)) % 5 == 0 { 0xaa } else { 0x22 })
        .collect();
    let b: Vec<u8> = (0..n * K / 2)
        .map(|i| if (i / (K / 2)) % 3 == 0 { 0xaa } else { 0x22 })
        .collect();
    let a = DeviceBuffer::from_host(&stream, &a)?;
    let b = DeviceBuffer::from_host(&stream, &b)?;
    let sa = DeviceBuffer::from_host(
        &stream,
        &common::encode_csf_scales(&vec![0x38; m * K / 16], m, K),
    )?;
    let sb = DeviceBuffer::from_host(
        &stream,
        &common::encode_csf_scales(&vec![0x38; n * K / 16], n, K),
    )?;
    let ga = DeviceBuffer::from_host(&stream, &[0.5f32])?;
    let gb = DeviceBuffer::from_host(&stream, &[0.25f32])?;
    let z: Vec<f32> = (0..active_rows * n)
        .map(|i| {
            if i % 9 == 0 {
                0.0
            } else {
                ((i % 31) as f32 - 15.0) * 0.001
            }
        })
        .collect();
    let x: Vec<f32> = z
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 9 == 0 { 0.0 } else { v * 0.7 + 0.001 })
        .collect();
    let zd = DeviceBuffer::from_host(&stream, &z)?;
    let xd = DeviceBuffer::from_host(&stream, &x)?;
    let controls = [1.4f32, 0.7, 1.2, 0.8, 0.9, 1.1];
    let scales = DeviceBuffer::from_host(&stream, &controls)?;
    let scale_grads = DeviceBuffer::<f32>::zeroed(&stream, 3)?;
    let tile_count = m / 128 * (n / 128);
    let norms = DeviceBuffer::from_host(&stream, &vec![-1.0f32; tile_count])?;
    let mut out = DeviceBuffer::from_host(&stream, &vec![1e30f32; m * n])?;
    let p = scales.cu_deviceptr();
    let g = scale_grads.cu_deviceptr();
    let desc = SymExpLinSlotDescriptor {
        grad: out.cu_deviceptr(),
        z_master: zd.cu_deviceptr(),
        x_master: xd.cu_deviceptr(),
        exponential_z: p,
        exponential_x: p + 4,
        exponential_grad: g,
        linear_z: p + 8,
        linear_x: p + 12,
        linear_grad: g + 4,
        curvature_z: p + 16,
        curvature_x: p + 20,
        curvature_grad: g + 8,
        rows: n as u32,
        cols: active_rows as u32,
        ..Default::default()
    };
    let gradient =
        GemmGradientTransform::new(desc, norms.cu_deviceptr(), schedule_beta, BETA, learn);
    let masks = DeviceBuffer::from_host(&stream, &[1u64, 1, 3, 0])?;
    let route = routed.then(|| Nvfp4GemmRoute::feature_m_token_k(&masks, 2, 2, 1));
    let mut tma = TmaNvfp4DeviceScaleDescriptors::new(&stream)?;
    gemm.prepare_tma_nvfp4_csf_scales_into(
        &stream, &a, &sa, &b, &sb, m as u32, K as u32, n as u32, &mut tma,
    )?;
    gemm.gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf_gradient(
        &stream, &tma, &mut out, m as u32, K as u32, n as u32, &ga, &gb, route, gradient,
    )?;
    let actual = out.to_host_vec(&stream)?;
    let actual_norms = norms.to_host_vec(&stream)?;
    let actual_grads = scale_grads.to_host_vec(&stream)?;
    let e = if learn {
        controls[0] + schedule_beta * (controls[1] - controls[0])
    } else {
        1.0
    };
    let l = if learn {
        controls[2] + schedule_beta * (controls[3] - controls[2])
    } else {
        1.0
    };
    let c = if learn {
        controls[4] + schedule_beta * (controls[5] - controls[4])
    } else {
        1.0
    };
    let mut expected_grads = [0.0f64; 3];
    let mut expected_norms = vec![0.0f64; tile_count];
    for row in 0..m {
        for col in 0..n {
            let i = row * n + col;
            let asgn = if row % 5 == 0 { -1.0 } else { 1.0 };
            let bsgn = if col % 3 == 0 { -1.0 } else { 1.0 };
            let grad = if routed && row >= 128 {
                0.0
            } else {
                K as f32 * 0.125 * asgn * bsgn
            };
            if row >= active_rows {
                close(actual[i], grad as f64, "inactive gradient", i);
                continue;
            }
            let raw = z[i] + schedule_beta * (x[i] - z[i]);
            let sign = if raw > 0.0 {
                1.0
            } else if raw < 0.0 {
                -1.0
            } else {
                0.0
            };
            let magnitude = raw.abs();
            let exp = (BETA * c * magnitude).exp();
            let transformed = grad * (e * c * exp + sign * l / BETA);
            close(actual[i], transformed as f64, "transformed gradient", i);
            expected_norms[row / 128 * (n / 128) + col / 128] +=
                (transformed as f64) * (transformed as f64);
            expected_grads[0] += (grad * sign * (exp - 1.0) / BETA) as f64;
            expected_grads[1] += (grad * magnitude / BETA) as f64;
            expected_grads[2] += (grad * sign * e * magnitude * exp) as f64;
        }
    }
    for i in 0..3 {
        close(
            actual_grads[i],
            if learn { expected_grads[i] } else { 0.0 },
            "scale gradient",
            i,
        );
    }
    for i in 0..tile_count {
        close(
            actual_norms[i],
            if learn { expected_norms[i] } else { -1.0 },
            "norm tile",
            i,
        );
    }
    println!("verified {m}x{n} transformed gradients; routed={routed} learned_scales={learn}");
    Ok(())
}
