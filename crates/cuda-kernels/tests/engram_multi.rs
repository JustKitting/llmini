//! Four-stream vector contract declared before implementation. These are native
//! primitive/oracle checks, not NVFP4 projector/full-model/long-loss evidence.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::{engram::kernels as legacy, engram_multi::kernels};
use std::error::Error;
mod common;
const N: usize = 32;
const D: usize = 2048;
const SEQ: usize = 16;
const S: usize = 4;
type Inputs = [Vec<f64>; 7]; // query,key,shared V,query gamma,key gamma,conv gamma,conv weights
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn forward(a: &Inputs) -> Vec<f64> {
    let mut u = vec![0.; N * S * D];
    let mut normalized = u.clone();
    for r in 0..N * S {
        let b = r * D;
        let stream = r % S;
        let token = r / S;
        let ih = 1.
            / ((0..D).map(|j| a[0][b + j].powi(2)).sum::<f64>() / D as f64 + f32::EPSILON as f64)
                .sqrt();
        let ik = 1.
            / ((0..D).map(|j| a[1][b + j].powi(2)).sum::<f64>() / D as f64 + f32::EPSILON as f64)
                .sqrt();
        let dot = (0..D)
            .map(|j| a[0][b + j] * a[3][stream * D + j] * a[1][b + j] * a[4][stream * D + j])
            .sum::<f64>()
            * ih
            * ik
            / (D as f64).sqrt();
        let sign = if dot > 0. {
            1.
        } else if dot < 0. {
            -1.
        } else {
            0.
        };
        let gate = sigmoid(sign * dot.abs().max(1e-6).sqrt());
        for j in 0..D {
            u[b + j] = gate * a[2][token * D + j];
        }
        let inv = 1. / ((0..D).map(|j| u[b + j].powi(2)).sum::<f64>() / D as f64 + 1e-5).sqrt();
        for j in 0..D {
            normalized[b + j] = u[b + j] * inv * a[5][stream * D + j];
        }
    }
    (0..N * S * D)
        .map(|i| {
            let r = i / D;
            let j = i % D;
            let token = r / S;
            let stream = r % S;
            let conv = (0..4)
                .filter_map(|tap| {
                    let lag = 3 * (3 - tap);
                    (token % SEQ >= lag).then(|| {
                        a[6][(stream * D + j) * 4 + tap]
                            * normalized[((token - lag) * S + stream) * D + j]
                    })
                })
                .sum::<f64>();
            a[0][i] + u[i] + conv * sigmoid(conv)
        })
        .collect()
}
fn grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn rows(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        ..grid(1)
    }
}

#[test]
#[ignore = "native d2048/four independent streams/shared V complete vector forward and adjoint"]
fn full_four_stream_gates_convolution_and_shared_value_adjoint() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = kernels::from_module(ptx.clone())?;
    let old = legacy::from_module(ptx)?;
    let host: [Vec<f32>; 7] = std::array::from_fn(|which| {
        let len = match which {
            0 | 1 => N * S * D,
            2 => N * D,
            3..=5 => S * D,
            _ => S * D * 4,
        };
        (0..len)
            .map(|i| match which {
                0 => {
                    if i / (S * D) == 0 {
                        0.
                    } else {
                        ((i * 17 % 101) as f32 - 50.) * 0.013 * (1. + (i / D % S) as f32 * 0.17)
                    }
                }
                1 => ((i * 29 % 89) as f32 - 44.) * 0.019,
                2 => ((i * 7 % 73) as f32 - 36.) * 0.009,
                3..=5 => 0.7 + ((i * 11 % 31) as f32) * 0.017 + which as f32 * 0.03,
                _ => ((i * 3 % 19) as f32 - 9.) * 0.007,
            })
            .collect()
    });
    let values: Inputs = std::array::from_fn(|i| host[i].iter().map(|&x| f64::from(x)).collect());
    let b: Vec<_> = host
        .iter()
        .map(|v| DeviceBuffer::from_host(&stream, v))
        .collect::<Result<_, _>>()?;
    let upstream = (0..N * S * D)
        .map(|i| ((i * 5 % 29) as f32 - 14.) * 0.003)
        .collect::<Vec<_>>();
    let g = DeviceBuffer::from_host(&stream, &upstream)?;
    let mut residual = DeviceBuffer::from_host(&stream, &host[0])?;
    let mut u = DeviceBuffer::zeroed(&stream, N * S * D)?;
    let mut un = DeviceBuffer::zeroed(&stream, N * S * D)?;
    let mut inv = DeviceBuffer::zeroed(&stream, N * S)?;
    let mut stats = DeviceBuffer::zeroed(&stream, N * S * 4)?;
    let mut pre = DeviceBuffer::zeroed(&stream, N * S * D)?;
    module.engram4_gate_kernel(
        &stream,
        rows(N * S),
        &b[0],
        &b[1],
        &b[2],
        &b[3],
        &b[4],
        &mut stats,
        &mut u,
        N as u32,
        D as u32,
        f32::EPSILON,
    )?;
    module.engram4_norm_kernel(
        &stream,
        rows(N * S),
        &u,
        &b[5],
        &mut un,
        &mut inv,
        N as u32,
        D as u32,
    )?;
    module.engram4_conv_add_kernel(
        &stream,
        grid(N * S * D),
        &u,
        &un,
        &b[6],
        &mut pre,
        &mut residual,
        N as u32,
        D as u32,
        SEQ as u32,
    )?;
    let result = residual.to_host_vec(&stream)?;
    let expected = forward(&values);
    let max_error = result
        .iter()
        .zip(&expected)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0_f64, f64::max);
    assert!(
        max_error < 2e-5,
        "all four stream outputs must match the reference: {max_error}"
    );
    let mut delta = pre;
    old.engram_conv_delta_kernel(&stream, grid(N * S * D), &g, &mut delta, (N * S * D) as u32)?;
    let mut dun = DeviceBuffer::zeroed(&stream, N * S * D)?;
    let mut du = DeviceBuffer::zeroed(&stream, N * S * D)?;
    module.engram4_conv_input_backward_kernel(
        &stream,
        grid(N * S * D),
        &delta,
        &b[6],
        &mut dun,
        N as u32,
        D as u32,
        SEQ as u32,
    )?;
    module.engram4_norm_backward_kernel(
        &stream,
        rows(N * S),
        &u,
        &dun,
        &b[5],
        &inv,
        &g,
        &mut du,
        N as u32,
        D as u32,
    )?;
    let mut dh = DeviceBuffer::from_host(&stream, &upstream)?;
    let mut dk = DeviceBuffer::zeroed(&stream, N * S * D)?;
    let mut dv = DeviceBuffer::zeroed(&stream, N * D)?;
    let mut dot = DeviceBuffer::zeroed(&stream, N * S)?;
    module.engram4_gate_backward_kernel(
        &stream,
        rows(N),
        &b[0],
        &b[1],
        &b[2],
        &b[3],
        &b[4],
        &stats,
        &du,
        &mut dh,
        &mut dk,
        &mut dv,
        &mut dot,
        N as u32,
        D as u32,
    )?;
    let mut dgh = DeviceBuffer::zeroed(&stream, S * D)?;
    let mut dgk = DeviceBuffer::zeroed(&stream, S * D)?;
    let mut dgu = DeviceBuffer::zeroed(&stream, S * D)?;
    let mut dw = DeviceBuffer::zeroed(&stream, S * D * 4)?;
    module.engram4_parameter_backward_kernel(
        &stream,
        grid(S * D * 16),
        &b[0],
        &b[1],
        &b[3],
        &b[4],
        &stats,
        &dot,
        &u,
        &un,
        &inv,
        &dun,
        &delta,
        &mut dgh,
        &mut dgk,
        &mut dgu,
        &mut dw,
        N as u32,
        D as u32,
        SEQ as u32,
    )?;
    let actual = [
        dh.to_host_vec(&stream)?,
        dk.to_host_vec(&stream)?,
        dv.to_host_vec(&stream)?,
        dgh.to_host_vec(&stream)?,
        dgk.to_host_vec(&stream)?,
        dgu.to_host_vec(&stream)?,
        dw.to_host_vec(&stream)?,
    ];
    assert!(actual.iter().flatten().all(|x| x.is_finite()));
    // At exact-zero queries use the demo's zero signed-root subgradient.
    assert_eq!(&actual[0][..S * D], &upstream[..S * D]);
    let objective = |a: &Inputs| {
        forward(a)
            .iter()
            .zip(&upstream)
            .map(|(&x, &g)| x * f64::from(g))
            .sum::<f64>()
    };
    for which in 0..7 {
        let indices = if which < 2 {
            (0..S)
                .flat_map(|s| {
                    [
                        (3 * S + s) * D + 7,
                        (12 * S + s) * D + 19,
                        (25 * S + s) * D + 37,
                    ]
                })
                .collect::<Vec<_>>()
        } else if which == 2 {
            vec![3 * D + 7, 12 * D + 19, 25 * D + 37]
        } else if which < 6 {
            (0..S).flat_map(|s| [s * D + 7, s * D + 19]).collect()
        } else {
            (0..S)
                .flat_map(|s| (0..4).map(move |tap| (s * D + 7) * 4 + tap))
                .collect()
        };
        for i in indices {
            let mut a = values.clone();
            let delta = 1e-5;
            a[which][i] += delta;
            let p = objective(&a);
            a[which][i] -= 2. * delta;
            let m = objective(&a);
            let expected = (p - m) / (2. * delta);
            let got = f64::from(actual[which][i]);
            assert!(
                (got - expected).abs() < 2e-4 + expected.abs() * 0.006,
                "full stream derivative group={which} index={i}: {got} != {expected}"
            );
        }
    }
    let duh = du.to_host_vec(&stream)?;
    let st = stats.to_host_vec(&stream)?;
    for token in 0..N {
        for j in 0..D {
            let expected = (0..S)
                .map(|s| {
                    f64::from(duh[(token * S + s) * D + j]) * f64::from(st[(token * S + s) * 4 + 3])
                })
                .sum::<f64>();
            let magnitude = (0..S)
                .map(|s| {
                    (f64::from(duh[(token * S + s) * D + j])
                        * f64::from(st[(token * S + s) * 4 + 3]))
                    .abs()
                })
                .sum::<f64>();
            assert!(
                (f64::from(actual[2][token * D + j]) - expected).abs() <= magnitude * 8e-7 + 1e-9,
                "shared V must sum four streams once, within FP32 reduction rounding"
            );
        }
    }
    println!(
        "engram4_vector_graph_verified streams=4 width=2048 shared_value=true independent_norms_and_conv=true all7_adjoint_groups=true max_output_error={max_error} updates=0"
    );
    Ok(())
}

#[test]
#[ignore = "native full-width stream-layout and one shared collision scatter"]
fn four_stream_copy_and_shared_memory_collision_scatter() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(kernels::from_module)?;
    let source = (0..N * S * D)
        .map(|i| i as f32 * 0.0001)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&stream, &source)?;
    let mut one = DeviceBuffer::zeroed(&stream, N * D)?;
    let mut all = DeviceBuffer::zeroed(&stream, N * S * D)?;
    for s in 0..S {
        module.engram4_stream_copy_kernel(
            &stream,
            grid(N * D),
            &input,
            &mut one,
            N as u32,
            D as u32,
            s as u32,
            true,
        )?;
        module.engram4_stream_copy_kernel(
            &stream,
            grid(N * D),
            &one,
            &mut all,
            N as u32,
            D as u32,
            s as u32,
            false,
        )?;
    }
    assert_eq!(all.to_host_vec(&stream)?, source);
    const HEADS: usize = 16;
    const HD: usize = 64;
    let ids = (0..N * HEADS)
        .map(|i| ((i % HEADS) * 7 + (i / HEADS) % 3) as u32)
        .collect::<Vec<_>>();
    let grad = (0..N * HEADS * HD)
        .map(|i| ((i * 11 % 47) as f32 - 23.) * 0.002)
        .collect::<Vec<_>>();
    let id = DeviceBuffer::from_host(&stream, &ids)?;
    let g = DeviceBuffer::from_host(&stream, &grad)?;
    let mut table = DeviceBuffer::zeroed(&stream, HEADS * 7 * HD)?;
    module.engram4_memory_scatter_kernel(
        &stream,
        grid(grad.len()),
        &id,
        &g,
        &mut table,
        N as u32,
        HEADS as u32,
        HD as u32,
    )?;
    let mut oracle = vec![0_f64; HEADS * 7 * HD];
    for i in 0..grad.len() {
        oracle[ids[i / HD] as usize * HD + i % HD] += f64::from(grad[i]);
    }
    for (got, want) in table.to_host_vec(&stream)?.iter().zip(oracle) {
        assert!((f64::from(*got) - want).abs() < 1e-6);
    }
    Ok(())
}
