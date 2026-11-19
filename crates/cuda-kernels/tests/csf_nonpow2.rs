//! Complete proposed MLP geometries: declare before relaxing codec assumptions.
use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor, csf,
};
mod common;

#[test]
fn canonical_csf_full5504_matrix_and11008_bias_roundtrip() {
    for (rows, k) in [(129usize, 5504usize), (2048, 5504), (1, 11008)] {
        for full in [false, true] {
            let raw = (0..rows * (k / 16))
                .map(|i| {
                    if full {
                        (i.wrapping_mul(73) % 256) as u8
                    } else {
                        (i.wrapping_mul(37) % 127) as u8
                    }
                })
                .collect::<Vec<_>>();
            let encoded = csf::encode_host(&raw, rows, k);
            for (g, &expected) in raw.iter().enumerate() {
                assert_eq!(
                    csf::scale_at(&encoded, g),
                    expected,
                    "canonical CSF rows={rows} k={k} group={g} full={full}"
                );
            }
        }
    }
}

fn scale(code: u8) -> f32 {
    let exponent = code >> 3;
    let mantissa = f32::from(code & 7);
    if exponent == 0 {
        mantissa / 512.
    } else {
        (1. + mantissa / 8.) * 2f32.powi(i32::from(exponent) - 7)
    }
}

#[test]
#[ignore = "native full2048x5504 down-weight and11008 bias decode/transpose, zero updates"]
fn native_csf_full_nonpower2_mlp_operands_decode_byte_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let (_, stream, module) = common::cuda_test_module(Nvfp4DecodeModule::from_module)?;
    for (rows, k) in [(2048, 5504), (1, 11008)] {
        let raw = (0..rows * (k / 16))
            .map(|i| 0x30 + (i % 39) as u8)
            .collect::<Vec<_>>();
        let bytes = DeviceBuffer::from_host(&stream, &vec![0x22; rows * k / 2])?;
        let scales = DeviceBuffer::from_host(&stream, &csf::encode_host(&raw, rows, k))?;
        let global = DeviceBuffer::from_host(&stream, &[0.5])?;
        let mut output = DeviceBuffer::zeroed(&stream, rows * k)?;
        module.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
            stream: &stream,
            input: Nvfp4DeviceTensor::new(&bytes, &scales, &global),
            output: &mut output,
            rows: rows as u32,
            cols: k as u32,
        })?;
        let actual = output.to_host_vec(&stream)?;
        for r in 0..rows {
            for c in 0..k {
                let expected = scale(raw[r * (k / 16) + c / 16]) * 0.5;
                assert_eq!(
                    actual[c * rows + r].to_bits(),
                    expected.to_bits(),
                    "native CSF transpose rows={rows} k={k} row={r} col={c}"
                );
            }
        }
        println!(
            "native_nonpower2_csf_verified rows={rows} k={k} every_decoded_transpose_value=bit_exact values={} updates=0",
            rows * k
        );
    }
    Ok(())
}
