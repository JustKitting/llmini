//! Predeclared exact-KDA pair-factor failures: all8192 native chunks, direct
//! negative intervals, six causal16x16 TC blocks, beta limits, ignored padded
//! NaNs, all16/64 boundaries, future-gate perturbations and host shape guards.
//! Host arithmetic below is an independent attention oracle, never training.
use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::kda_exact_pairs::{KdaExactPairModule, PairArgs};

mod common;
#[allow(dead_code)]
#[path = "causal_attention_backward_tc/f16.rs"]
mod half_reference;

const D: usize = 64;
const CHUNKS: usize = 8 * 32 * 32;
const GUARD: usize = 257;

fn q(case: usize, t: usize, h: usize) -> f32 {
    if t == 0 {
        return 0.0;
    }
    if case == 2 {
        return 1.0 / 64.0;
    }
    ((t * 13 + h * 7 + case) as i32 % 9 - 4) as f32 / 256.0
}
fn k(case: usize, t: usize, h: usize) -> f32 {
    if case == 2 {
        return 1.0 / 8.0;
    }
    ((t * 5 + h * 3 + case) as i32 % 5 - 2) as f32 / 16.0
}
fn g(case: usize, t: usize, h: usize) -> f32 {
    match case {
        0 => -0.01,
        1 => -0.3,
        2 => {
            if t == 0 {
                -600.0
            } else {
                -0.0001
            }
        }
        3 => -8.0,
        4 => 0.0,
        _ => -0.015 * (1 + (t + h * 3) % 7) as f32,
    }
}
fn beta(case: usize, t: usize) -> f32 {
    match case {
        6 => 0.0,
        7 => 1.0,
        _ => (1 + t % 4) as f32 / 8.0,
    }
}
// Mathematical binary16 RNE, independent of the CUDA conversion instruction.
fn half_round(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    let spacing = 2.0f64.powi((x.abs().log2().floor() as i32).max(-14) - 10);
    (x / spacing).round_ties_even() * spacing
}
fn half_ulp(x: f64) -> f64 {
    if x == 0.0 {
        2.0f64.powi(-24)
    } else {
        2.0f64.powi((x.abs().log2().floor() as i32).max(-14) - 10)
    }
}
fn interval(case: usize, first: usize, last: usize, h: usize) -> f32 {
    let mut sum = 0.0f32;
    for t in first..=last {
        sum += g(case, t, h);
    }
    sum
}
fn reference(case: usize, t: usize, j: usize) -> (f64, f64, f64, f64) {
    if j > t {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut l = 0.0;
    let mut c = 0.0;
    let mut ideal_l = 0.0;
    let mut ideal_c = 0.0;
    for h in 0..D {
        let qt = f64::from(q(case, t, h));
        let kt = f64::from(k(case, t, h));
        let kj = f64::from(k(case, j, h));
        let bt = f64::from(beta(case, t));
        let exact_log: f64 = (j + 1..=t).map(|s| f64::from(g(case, s, h))).sum();
        ideal_l += bt * kt * kj * exact_log.exp();
        ideal_c += qt * kj * exact_log.exp();
        if t / 16 == j / 16 {
            // The diagonal tile sums original log increments backwards.
            let mut log = 0.0f32;
            for s in (j + 1..=t).rev() {
                log += g(case, s, h);
            }
            let decay = f64::from(log).exp();
            l += bt * kt * kj * decay;
            c += qt * kj * decay;
        } else {
            let anchor = t / 16 * 16;
            let tq = f64::from(interval(case, anchor + 1, t, h)).exp();
            let tj = f64::from(interval(case, j + 1, anchor, h)).exp();
            l += half_round(bt * kt * tq) * half_round(kj * tj);
            c += half_round(qt * tq) * half_round(kj * tj);
        }
    }
    if j == t {
        l = 0.0;
        ideal_l = 0.0;
    }
    (l, half_round(c), ideal_l, ideal_c)
}

fn launch(
    module: &KdaExactPairModule,
    stream: &CudaStream,
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<f32>,
    g: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    l: &mut DeviceBuffer<f32>,
    c: &mut DeviceBuffer<u16>,
    batch_heads: u32,
    seq_len: u32,
) -> Result<(), cuda_core::DriverError> {
    module.pairs(PairArgs {
        stream,
        q,
        k,
        g,
        beta,
        lower: l,
        readout: c,
        batch_heads,
        seq_len,
    })
}

#[test]
#[ignore = "native B8/S2048/H32 exact causal factors; explicit GPU1 plus sanitizers"]
fn native_kda_exact_pairs_all_chunks_causal_tiles_and_guards() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(KdaExactPairModule::from_module)?;
    let count = CHUNKS * D * D;
    let mut qh = Vec::with_capacity(count);
    let mut kh = Vec::with_capacity(count);
    let mut gh = Vec::with_capacity(count);
    let mut bh = Vec::with_capacity(CHUNKS * D);
    for chunk in 0..CHUNKS {
        let case = chunk % 8;
        for t in 0..D {
            bh.push(beta(case, t));
            for h in 0..D {
                qh.push(if t == 0 && h == 0 {
                    (chunk + 1) as f32 / 16384.0
                } else {
                    q(case, t, h)
                });
                kh.push(k(case, t, h));
                gh.push(g(case, t, h));
            }
        }
    }
    let qd = DeviceBuffer::from_host(&stream, &qh)?;
    let kd = DeviceBuffer::from_host(&stream, &kh)?;
    let gd = DeviceBuffer::from_host(&stream, &gh)?;
    let bd = DeviceBuffer::from_host(&stream, &bh)?;
    let mut ls = vec![f32::NAN; count + GUARD];
    ls[count..].fill(f32::from_bits(0x3eaa1234));
    let mut cs = vec![0x7e00u16; count + GUARD];
    cs[count..].fill(0x3555);
    let mut ld = DeviceBuffer::from_host(&stream, &ls)?;
    let mut cd = DeviceBuffer::from_host(&stream, &cs)?;
    launch(
        &module, &stream, &qd, &kd, &gd, &bd, &mut ld, &mut cd, 256, 2048,
    )?;
    let actual_l = ld.to_host_vec(&stream)?;
    let actual_c = cd.to_host_vec(&stream)?;
    assert_eq!(
        actual_l[count..]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        ls[count..].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(&actual_c[count..], &cs[count..]);
    let refs = (0..8)
        .map(|case| {
            (0..D * D)
                .map(|i| reference(case, i / D, i % D))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut max_l_error = 0.0f64;
    let mut max_c_error = 0.0f64;
    let mut max_ideal_l_error = 0.0f64;
    let mut max_ideal_c_error = 0.0f64;
    for chunk in 0..CHUNKS {
        for i in 0..D * D {
            let t = i / D;
            let j = i % D;
            let at = chunk * D * D + i;
            let (el, mut ec, il, mut ic) = refs[chunk % 8][i];
            if i == 0 {
                ic = f64::from(qh[chunk * D * D]) * f64::from(kh[chunk * D * D]);
                ec = half_round(ic);
            }
            let al = f64::from(actual_l[at]);
            let ac = f64::from(half_reference::f16_bits_to_f32(actual_c[at]));
            assert!(
                al.is_finite() && ac.is_finite(),
                "chunk={chunk} row={t} col={j}"
            );
            let le = (al - el).abs();
            let ce = (ac - ec).abs();
            assert!(
                le <= 1e-5 + 2e-4 * el.abs(),
                "L chunk={chunk} row={t} col={j} actual={al} reference={el} error={le}"
            );
            assert!(
                ce <= 2e-6 + 1.1 * half_ulp(ec),
                "C chunk={chunk} row={t} col={j} actual={ac} reference={ec} error={ce}"
            );
            assert!(
                (al - il).abs() <= 1e-3 && (ac - ic).abs() <= 1e-4,
                "bounded factors disagree with direct recurrence pairs"
            );
            if j >= t {
                assert_eq!(actual_l[at].to_bits(), 0);
            }
            if j > t {
                assert_eq!(actual_c[at], 0);
            }
            max_l_error = max_l_error.max(le);
            max_c_error = max_c_error.max(ce);
            max_ideal_l_error = max_ideal_l_error.max((al - il).abs());
            max_ideal_c_error = max_ideal_c_error.max((ac - ic).abs());
        }
    }
    // Native prefix and future-gate checks use original inputs and actual mixed
    // precision output bits, including cuts within tiles and across64 chunks.
    for cut in [
        1usize, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 66, 127,
    ] {
        let chunks = cut.div_ceil(D);
        let small_count = chunks * D * D;
        let small_q = DeviceBuffer::from_host(&stream, &qh[..small_count])?;
        let small_k = DeviceBuffer::from_host(&stream, &kh[..small_count])?;
        let small_b = DeviceBuffer::from_host(&stream, &bh[..chunks * D])?;
        let mut future = gh[..small_count].to_vec();
        for t in cut % D..D {
            if cut % D != 0 {
                future[(chunks - 1) * D * D + t * D..(chunks - 1) * D * D + (t + 1) * D]
                    .fill(-31.0);
            }
        }
        let changed_g = DeviceBuffer::from_host(&stream, &future)?;
        let mut pl = DeviceBuffer::<f32>::zeroed(&stream, small_count)?;
        let mut pc = DeviceBuffer::<u16>::zeroed(&stream, small_count)?;
        launch(
            &module,
            &stream,
            &small_q,
            &small_k,
            &changed_g,
            &small_b,
            &mut pl,
            &mut pc,
            1,
            (chunks * D) as u32,
        )?;
        let changed_l = pl.to_host_vec(&stream)?;
        let changed_c = pc.to_host_vec(&stream)?;
        for token in 0..cut {
            let chunk = token / D;
            let row = token % D;
            for j in 0..D {
                let i = chunk * D * D + row * D + j;
                assert_eq!(
                    changed_l[i].to_bits(),
                    actual_l[i].to_bits(),
                    "future gate changed L prefix at cut={cut}"
                );
                assert_eq!(
                    changed_c[i], actual_c[i],
                    "future gate changed C prefix at cut={cut}"
                );
            }
        }
        // Padding is intentionally nonfinite; no retained output may read it.
        if cut % D != 0 {
            future[(chunks - 1) * D * D + (cut % D) * D..].fill(f32::NAN);
        }
        let padded_g = DeviceBuffer::from_host(&stream, &future)?;
        launch(
            &module, &stream, &small_q, &small_k, &padded_g, &small_b, &mut pl, &mut pc, 1,
            cut as u32,
        )?;
        let prefix_l = pl.to_host_vec(&stream)?;
        let prefix_c = pc.to_host_vec(&stream)?;
        for token in 0..chunks * D {
            let chunk = token / D;
            let row = token % D;
            for j in 0..D {
                let i = chunk * D * D + row * D + j;
                if token < cut {
                    assert_eq!(
                        prefix_l[i].to_bits(),
                        actual_l[i].to_bits(),
                        "truncated L cut={cut}"
                    );
                    assert_eq!(prefix_c[i], actual_c[i], "truncated C cut={cut}");
                } else {
                    assert_eq!(prefix_l[i].to_bits(), 0);
                    assert_eq!(prefix_c[i], 0);
                }
            }
        }
    }
    for (batch_heads, seq_len) in [(0, 2048), (256, 0), (u32::MAX, 2048), (256, u32::MAX)] {
        assert!(
            launch(
                &module,
                &stream,
                &qd,
                &kd,
                &gd,
                &bd,
                &mut ld,
                &mut cd,
                batch_heads,
                seq_len
            )
            .is_err()
        );
    }
    let short = DeviceBuffer::from_host(&stream, &[0.0f32])?;
    assert!(
        launch(
            &module, &stream, &short, &kd, &gd, &bd, &mut ld, &mut cd, 256, 2048
        )
        .is_err()
    );
    assert_eq!(
        ld.to_host_vec(&stream)?
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        actual_l.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(cd.to_host_vec(&stream)?, actual_c);
    println!(
        "native_exact_KDA_pairs_verified B=8 S=2048 H=32 chunk_batch={CHUNKS} pattern_cases=8 prefix_cuts=15 max_L_error={max_l_error:e} max_C_error={max_c_error:e} max_ideal_L_error={max_ideal_l_error:e} max_ideal_C_error={max_ideal_c_error:e} output_guards=unchanged optimizer_updates=0"
    );
    Ok(())
}
