//! Contract regression: distinct z/x views, not a convergence/training screen.
use super::{Module as Engram, ModuleWeights as Weights};
use crate::{
    AppResult,
    training::{optimizer::OptimizerScratch, runtime::Runtime},
};
use cuda_core::{CudaStream, DeviceBuffer};

fn fill(s: &CudaStream, dst: &mut DeviceBuffer<f32>, value: f32) -> AppResult {
    *dst = DeviceBuffer::from_host(s, &vec![value; dst.len()])?;
    Ok(())
}

fn fixture() -> AppResult<(Runtime, Weights, Engram, OptimizerScratch)> {
    let r = Runtime::new()?;
    let w = Weights::new(&r)?;
    let mut e = Engram::new(&r, &w)?;
    let s = r.stream.as_ref();
    for st in &mut e.nv_states {
        fill(s, &mut st.z_master, -0.5)?;
        fill(s, &mut st.x_master, 0.25)?;
    }
    for st in &mut e.projector_states {
        fill(s, &mut st.z_master, -0.03)?;
        fill(s, &mut st.x_master, 0.01)?;
        for (control, z, x) in [
            (&mut st.symexp_lin.exponential, 2.0, 1.0),
            (&mut st.symexp_lin.linear, 1.5, 0.5),
            (&mut st.symexp_lin.curvature, 1.25, 0.75),
        ] {
            fill(s, &mut control.z_master, z)?;
            fill(s, &mut control.x_master, x)?;
        }
    }
    for st in &mut e.fp_states {
        fill(s, &mut st.z_master, -0.5)?;
        fill(s, &mut st.x_master, 0.25)?;
    }
    let scratch = OptimizerScratch::new(s)?;
    Ok((r, w, e, scratch))
}

// Host verification only: decode stored E2M1/E4M3 payload, no CPU training.
fn e4m3(bits: u8) -> f64 {
    assert!(bits < 127);
    let exponent = (bits >> 3) as i32;
    let mantissa = (bits & 7) as f64;
    if exponent == 0 {
        mantissa * 2f64.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    }
}

fn sample(w: &crate::upload::UploadedNvfp4, s: &CudaStream) -> AppResult<Vec<f64>> {
    let bytes = w.bytes.to_host_vec(s)?;
    let scales = w.scales.to_host_vec(s)?;
    let global = f64::from(w.global_scale.to_host_vec(s)?[0]);
    Ok([0, 15, 16, 127, 128, w.len / 2, w.len - 1]
        .into_iter()
        .map(|i| {
            let nibble = (bytes[i / 2] >> (4 * (i % 2))) & 15;
            let magnitude = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(nibble & 7) as usize];
            let sign = if nibble & 8 == 0 { 1. } else { -1. };
            sign * magnitude
                * e4m3(rust_kernels_cuda::nvfp4::csf::scale_at(&scales, i / 16))
                * global
        })
        .collect())
}

fn snapshots(w: &Weights, s: &CudaStream) -> AppResult<Vec<(Vec<u8>, Vec<u8>, Vec<u32>)>> {
    w.nv.iter()
        .map(|t| {
            Ok((
                t.bytes.to_host_vec(s)?,
                t.scales.to_host_vec(s)?,
                t.global_scale
                    .to_host_vec(s)?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect(),
            ))
        })
        .collect()
}

#[test]
#[ignore = "native CUDA Engram state-view regression, zero optimizer updates"]
fn native_engram_materialization_evaluation_uses_averaged_x_for_every_tensor() -> AppResult {
    let (r, mut w, mut e, mut scratch) = fixture()?;
    e.materialize_evaluation(&r, &mut w, &mut scratch)?;
    for i in [0, 2, 4] {
        for actual in sample(&w.nv[i], &r.stream)? {
            assert!(
                (actual - 0.25).abs() < 0.02,
                "evaluation tensor {i} must select averaged x=0.25, got {actual}"
            );
        }
    }
    let beta = f64::from(crate::training::symexp_lin::beta());
    let expected = ((beta * 0.75 * 0.01).exp() - 1.0 + 0.5 * 0.01) / beta;
    for i in [1, 3] {
        for actual in sample(&w.nv[i], &r.stream)? {
            assert!(
                (actual - expected).abs() < 0.002,
                "evaluation projector {i} must use x masters and x SEL controls: {actual} vs {expected}"
            );
        }
    }
    for (i, fp) in w.fp.iter().enumerate() {
        assert!(
            fp.weight
                .to_host_vec(&r.stream)?
                .iter()
                .all(|v| (*v - 0.25).abs() < 1e-6),
            "evaluation FP32 tensor {i} must select x"
        );
    }
    println!(
        "Engram evaluation: all nine tensors and six SEL controls use averaged x; zero optimizer updates"
    );
    Ok(())
}

#[test]
#[ignore = "native CUDA Engram state-view regression, zero optimizer updates"]
fn native_engram_materialization_training_controls_match_native_interpolation() -> AppResult {
    let (r, mut w, mut e, mut scratch) = fixture()?;
    let beta = 0.93;
    e.materialize_training(&r, &mut w, &mut scratch, beta)?;
    let expected = -0.5 + beta * (0.25 - -0.5);
    for (i, fp) in w.fp.iter().enumerate() {
        let values = fp.weight.to_host_vec(&r.stream)?;
        assert!(
            values.iter().all(|v| (*v - expected).abs() < 1e-6),
            "training FP32 tensor {i} must use z+beta*(x-z)={expected}, got {}",
            values[0]
        );
    }
    let training = snapshots(&w, &r.stream)?;
    let controls =
        w.fp.iter()
            .map(|t| t.weight.to_host_vec(&r.stream))
            .collect::<Result<Vec<_>, _>>()?;
    e.materialize_evaluation(&r, &mut w, &mut scratch)?;
    e.materialize_training(&r, &mut w, &mut scratch, beta)?;
    assert_eq!(
        training,
        snapshots(&w, &r.stream)?,
        "evaluation must not corrupt restored training payload/scales/globals"
    );
    for (fp, expected) in w.fp.iter().zip(controls) {
        assert_eq!(fp.weight.to_host_vec(&r.stream)?, expected);
    }
    assert!(
        e.nv_states[0]
            .x_master
            .to_host_vec(&r.stream)?
            .iter()
            .all(|x| *x == 0.25)
    );
    assert!(
        e.nv_states[0]
            .z_master
            .to_host_vec(&r.stream)?
            .iter()
            .all(|x| *x == -0.5)
    );
    println!(
        "Engram training: correct beta orientation and bit-exact restoration; zero optimizer updates"
    );
    Ok(())
}
