//! Native recipe counterexamples; host arithmetic only checks CUDA results.
use super::*;
use super::{Module as Engram, ModuleWeights as Weights};
use crate::{
    AppResult,
    training::{optimizer::OptimizerScratch, runtime::Runtime},
};

fn decoded_sample(s: &CudaStream, w: &crate::upload::UploadedNvfp4) -> AppResult<Vec<f64>> {
    let bytes = w.bytes.to_host_vec(s)?;
    let scales = w.scales.to_host_vec(s)?;
    let global = f64::from(w.global_scale.to_host_vec(s)?[0]);
    let stride = (w.len / 262144).max(1);
    Ok((0..w.len)
        .step_by(stride)
        .map(|i| {
            let code = (bytes[i / 2] >> (4 * (i % 2))) & 15;
            let sf = rust_kernels_cuda::nvfp4::csf::scale_at(&scales, i / 16);
            let exponent = (sf >> 3) as i32;
            let fraction = f64::from(sf & 7);
            let scale = if exponent == 0 {
                fraction * 2f64.powi(-9)
            } else {
                (1. + fraction / 8.) * 2f64.powi(exponent - 7)
            };
            let mag = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(code & 7) as usize];
            (if code & 8 == 0 { mag } else { -mag }) * scale * global
        })
        .collect())
}

#[test]
#[ignore = "native reference initialization contract; zero training updates"]
fn native_engram_reference_initialization_distributions() -> AppResult {
    let r = Runtime::new()?;
    let w = Weights::new(&r)?;
    for (i, t) in w.nv.iter().enumerate() {
        let a = decoded_sample(&r.stream, t)?;
        assert!(a.iter().all(|v| v.is_finite()));
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        let rms = (a.iter().map(|v| v * v).sum::<f64>() / a.len() as f64).sqrt();
        let expected = if i == 0 {
            1.
        } else {
            1. / (3. * MEMORY as f64).sqrt()
        };
        println!("reference_init tensor={i} mean={mean} rms={rms} expected_rms={expected}");
        assert!(
            (rms / expected - 1.).abs() < 0.12,
            "reference table is normal(0,1); nn.Linear weight/bias are uniform +/-1/sqrt(fan_in): tensor={i} rms={rms}"
        );
        assert!(mean.abs() < expected * 0.12);
        if i == 0 {
            let tails = a.iter().filter(|v| v.abs() > 2.).count() as f64 / a.len() as f64;
            assert!(
                tails > 0.02 && tails < 0.09,
                "not a Gaussian NVFP4 initializer: tails={tails}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "native actual Engram Adam update; one isolated correctness fixture, no training experiment"]
fn native_engram_reference_embedding_lr_is_five_times_base_only() -> AppResult {
    let r = Runtime::new()?;
    let s = r.stream.as_ref();
    let mut w = Weights::new(&r)?;
    let mut e = Engram::new(&r, &w)?;
    let mut scratch = OptimizerScratch::new(s)?;
    for st in &e.nv_states {
        for a in [&st.z_master, &st.x_master, &st.first, &st.second] {
            zero(s, a)?;
        }
    }
    for st in &e.fp_states {
        for a in [&st.z_master, &st.x_master, &st.first, &st.second] {
            zero(s, a)?;
        }
    }
    for g in e.ng.iter_mut().chain(e.fg.iter_mut()) {
        let host = vec![0.75; g.len()];
        *g = DeviceBuffer::from_host(s, &host)?;
        s.synchronize()?;
    }
    e.update(&r, &mut w, &mut scratch, 1, 1., 0.25)?;
    let lr = crate::training::optimizer_apply::adam_debug_config(1).learning_rate as f64;
    for (index, st) in [0, 2, 4].into_iter().zip(&e.nv_states) {
        let z = st.z_master.to_host_vec(s)?;
        let x = st.x_master.to_host_vec(s)?;
        let expected = -lr * if index == 0 { 5. } else { 1. };
        for i in [0, 15, 16, 127, z.len() / 2, z.len() - 1] {
            assert!(
                (z[i] as f64 - expected).abs() < lr * 0.0001,
                "Engram embedding-only LR contract: tensor={index} index={i} actual={} expected={expected}",
                z[i]
            );
            assert_eq!(z[i], x[i], "first-update average must select updated z");
        }
    }
    for st in &e.fp_states {
        let z = st.z_master.to_host_vec(s)?;
        assert!(
            z.iter().all(|v| (*v as f64 + lr).abs() < lr * 0.0001),
            "embedding LR multiplier leaked into norms/convolution"
        );
    }
    println!(
        "native reference recipe: table LR=5x, biases/norms/conv=1x, one isolated Adam fixture, no model training"
    );
    Ok(())
}
