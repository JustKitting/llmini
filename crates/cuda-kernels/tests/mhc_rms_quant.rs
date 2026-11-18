//! Declared before the fused mHC RMS/NVFP4 producer: preserve the exact256-
//! thread RMS sum, inverse bits, normalized-value rounding, derived maxima,
//! row globals, native FP4 payload and canonical compressed CSF scale bytes.
//! Full B8/S2048 rows, zero/tiny/subnormal/large/outlier inputs, odd row counts,
//! complete guards and rejected shapes; no new normalization architecture.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::mhc::kernels as legacy;
use rust_kernels_cuda::mhc_rms_quant::{MhcRmsQuantModule, RmsQuantArgs};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseDerivedAmaxArgs};
use std::error::Error;
mod common;
const WIDTH: usize = 8192;
const GUARD: usize = 257;
fn value(row: usize, j: usize) -> f32 {
    match row % 8 {
        0 => 0.0,
        1 => {
            if j % 2 == 0 {
                1e-30
            } else {
                -1e-30
            }
        }
        2 => {
            if j % 2 == 0 {
                1e10
            } else {
                -1e10
            }
        }
        3 => {
            if j == (row * 17) % WIDTH {
                60000.0
            } else {
                0.00013
            }
        }
        4 => ((j % 7) as i32 - 3) as f32 / 1024.0,
        6 => {
            if j % 2 == 0 {
                f32::from_bits(0x80000000)
            } else {
                0.0
            }
        }
        7 => f32::from_bits((j % 13) as u32 + 1) * if j % 2 == 0 { 1.0 } else { -1.0 },
        _ => {
            (((row * 17 + j * 19) % 101) as i32 - 50) as f32
                * 0.013
                * (1.0 + ((row / 8) % 31) as f32 / 32.0)
        }
    }
}
fn fp_equal(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} i={i} old={a:e} new={b:e}"
        );
    }
}
#[test]
#[ignore = "native complete B8/S2048/8192-wide exact RMS and NVFP4/CSF fused production"]
fn native_mhc_rms_quant_preserves_inverse_payload_scales_and_rounding() -> Result<(), Box<dyn Error>>
{
    let (ctx, stream, ptx) = common::cuda_test_context()?;
    // Keep the reference independent of the shared quantizer extraction.
    // This frozen native module predates that extraction and must still load
    // through the ordinary quantizer interface without missing-symbol probes.
    let reference_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles/20261116/mhc-vector-batch.ptx");
    let reference = ctx.load_module_from_file(reference_path.to_str().unwrap())?;
    let old = legacy::from_module(reference.clone())?;
    let quant = Nvfp4QuantModule::from_module(reference)?;
    let new = MhcRmsQuantModule::from_module(ptx)?;
    let mut bytes_compared = 0usize;
    for n in [1usize, 3, 7, 8, 9, 31, 128, 16384] {
        let elements = n * WIDTH;
        let scale_count = rust_kernels_cuda::nvfp4::csf::rowwise_encoded_len(n, WIDTH);
        let input = (0..elements)
            .map(|i| value(i / WIDTH, i % WIDTH))
            .collect::<Vec<_>>();
        let xd = DeviceBuffer::from_host(&stream, &input)?;
        drop(input);
        let mut normalized = DeviceBuffer::zeroed(&stream, elements)?;
        let mut inv0 = DeviceBuffer::from_host(&stream, &vec![0.375f32; n + GUARD])?;
        let mut inv1 = DeviceBuffer::from_host(&stream, &vec![0.375f32; n + GUARD])?;
        let mut max0 = DeviceBuffer::from_host(&stream, &vec![0.375f32; n + GUARD])?;
        let mut max1 = DeviceBuffer::from_host(&stream, &vec![0.375f32; n + GUARD])?;
        let mut global0 = DeviceBuffer::from_host(&stream, &vec![0.375f32; n + GUARD])?;
        let mut global1 = DeviceBuffer::from_host(&stream, &vec![0.375f32; n + GUARD])?;
        let mut payload0 = DeviceBuffer::from_host(&stream, &vec![0xA5u8; elements / 2 + GUARD])?;
        let mut payload1 = DeviceBuffer::from_host(&stream, &vec![0xA5u8; elements / 2 + GUARD])?;
        let mut scales0 = DeviceBuffer::from_host(&stream, &vec![0xA5u8; scale_count + GUARD])?;
        let mut scales1 = DeviceBuffer::from_host(&stream, &vec![0xA5u8; scale_count + GUARD])?;
        old.mhc_rms_kernel(
            &stream,
            LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &xd,
            &mut normalized,
            &mut inv0,
            n as u32,
            WIDTH as u32,
        )?;
        quant.fp32_to_nvfp4_four_six_rowwise_derived_amax(Nvfp4QuantRowwiseDerivedAmaxArgs {
            stream: &stream,
            x: &normalized,
            amax: &mut max0,
            out_fp4: &mut payload0,
            out_scales: &mut scales0,
            out_global_scale: &mut global0,
            row_count: n as u32,
            row_len: WIDTH as u32,
        })?;
        new.forward(RmsQuantArgs {
            stream: &stream,
            input: &xd,
            inverse: &mut inv1,
            amax: &mut max1,
            payload: &mut payload1,
            scales: &mut scales1,
            global_scales: &mut global1,
            rows: n as u32,
        })?;
        for (a, b, label) in [
            (&inv0, &inv1, "inverse"),
            (&max0, &max1, "amax"),
            (&global0, &global1, "row global"),
        ] {
            let a = a.to_host_vec(&stream)?;
            let b = b.to_host_vec(&stream)?;
            fp_equal(&a, &b, label);
            assert!(b[n..].iter().all(|&x| x == 0.375));
        }
        for (a, b, active, label) in [
            (&payload0, &payload1, elements / 2, "FP4 payload"),
            (&scales0, &scales1, scale_count, "CSF scales"),
        ] {
            let a = a.to_host_vec(&stream)?;
            let b = b.to_host_vec(&stream)?;
            for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                assert_eq!(a, b, "{label} n={n} i={i}");
            }
            assert!(b[active..].iter().all(|&x| x == 0xA5));
            bytes_compared += active;
        }
    }
    let one = DeviceBuffer::from_host(&stream, &[1.0f32])?;
    let mut inv = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    let mut max = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    let mut global = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    let mut payload = DeviceBuffer::from_host(&stream, &[0xA5u8; GUARD])?;
    let mut scales = DeviceBuffer::from_host(&stream, &[0xA5u8; GUARD])?;
    for rows in [0, u32::MAX, 16384] {
        assert!(
            new.forward(RmsQuantArgs {
                stream: &stream,
                input: &one,
                inverse: &mut inv,
                amax: &mut max,
                payload: &mut payload,
                scales: &mut scales,
                global_scales: &mut global,
                rows
            })
            .is_err()
        );
    }
    for data in [&inv, &max, &global] {
        assert!(data.to_host_vec(&stream)?.iter().all(|&x| x == 0.375));
    }
    for data in [&payload, &scales] {
        assert!(data.to_host_vec(&stream)?.iter().all(|&x| x == 0xA5));
    }
    println!(
        "native_mhc_fused_rms_quant_verified max_rows=16384 width=8192 row_counts=8 inverse_amax_globals_exact=true FP4_payload_CSF_scales_exact=true original_RMS_reduction_order=true bytes_compared={bytes_compared} global_normalized_FP32_plane=false"
    );
    Ok(())
}
