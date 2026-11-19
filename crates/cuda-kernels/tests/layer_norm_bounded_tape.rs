//! Declared after the real15559 reproducer, before the bounded-tape repair.
//! All normalization work is CUDA; host f64 only verifies derivatives.
use cuda_core::DeviceBuffer;
use rust_kernels_cuda::{
    layer_norm::{GptLayerNormArgs, GptLayerNormSaveXhatF16Args, LayerNormModule},
    layer_norm_backward::{
        LayerNormBackwardInputAddAmaxArgs, LayerNormBackwardInputAddArgs, LayerNormBackwardModule,
        LayerNormBackwardParamArgs,
    },
    nvfp4::Nvfp4DeviceTensor,
};
use std::error::Error;
mod common;
#[test]
#[ignore = "native2048/4096 standardized16-bit tape, large raw outliers/offsets and complete LN derivatives"]
fn bounded_layernorm_tape_preserves_forward_and_finite_complete_adjoint()
-> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let f = LayerNormModule::from_module(ptx.clone())?;
    let b = LayerNormBackwardModule::from_module(ptx)?;
    for d in [2048_usize, 4096] {
        let n = 4;
        let x = (0..n * d)
            .map(|i| match i / d {
                0 => {
                    if i % d == 283 {
                        200000.
                    } else {
                        ((i * 13 % 31) as f32 - 15.) * 0.02
                    }
                }
                1 => 1000000. + ((i * 7 % 29) as f32 - 14.) * 0.25,
                2 => 7.,
                _ => ((i * 11 % 53) as f32 - 26.) * 0.03125,
            })
            .collect::<Vec<_>>();
        let dy = (0..n * d)
            .map(|i| ((i * 17 % 37) as f32 - 18.) * 0.003)
            .collect::<Vec<_>>();
        let direct = (0..n * d)
            .map(|i| ((i % 11) as f32 - 5.) * 0.007)
            .collect::<Vec<_>>();
        let xd = DeviceBuffer::from_host(&stream, &x)?;
        let gd = DeviceBuffer::from_host(&stream, &dy)?;
        let dd = DeviceBuffer::from_host(&stream, &direct)?;
        let wb = DeviceBuffer::from_host(&stream, &common::nvfp4::one_pair_bytes(d))?;
        let ws = DeviceBuffer::from_host(
            &stream,
            &rust_kernels_cuda::nvfp4::csf::encode_host(&common::nvfp4::one_scales(d), 1, d),
        )?;
        let gs = DeviceBuffer::from_host(&stream, &[1.])?;
        let zb = DeviceBuffer::zeroed(&stream, d / 2)?;
        let weight = Nvfp4DeviceTensor::new(&wb, &ws, &gs);
        let bias = Nvfp4DeviceTensor::new(&zb, &ws, &gs);
        let mut plain = DeviceBuffer::zeroed(&stream, n * d)?;
        let mut y = DeviceBuffer::zeroed(&stream, n * d)?;
        let mut mean = DeviceBuffer::zeroed(&stream, n)?;
        let mut inv = DeviceBuffer::zeroed(&stream, n)?;
        let mut amax = DeviceBuffer::zeroed(&stream, n)?;
        let mut tape = DeviceBuffer::zeroed(&stream, n * d)?;
        f.gpt_layer_norm(GptLayerNormArgs {
            stream: &stream,
            residual: &xd,
            weight,
            bias,
            normalized: &mut plain,
            normalized_amax: &mut amax,
            mean: &mut mean,
            inv_std: &mut inv,
            row_count: n as u32,
            embedding_dim: d as u32,
            epsilon: 1e-5,
            output_scale: 0.25,
        })?;
        f.gpt_layer_norm_save_xhat_f16(GptLayerNormSaveXhatF16Args {
            stream: &stream,
            residual: &xd,
            weight,
            bias,
            normalized: &mut y,
            normalized_amax: &mut amax,
            mean: &mut mean,
            inv_std: &mut inv,
            xhat_f16: &mut tape,
            row_count: n as u32,
            embedding_dim: d as u32,
            epsilon: 1e-5,
            output_scale: 0.25,
        })?;
        assert_eq!(
            plain.to_host_vec(&stream)?,
            y.to_host_vec(&stream)?,
            "forward output must remain bit exact"
        );
        assert!(
            tape.to_host_vec(&stream)?
                .iter()
                .all(|v| v & 0x7c00 != 0x7c00)
        );
        let mut dx = DeviceBuffer::zeroed(&stream, n * d)?;
        let mut chunks = DeviceBuffer::zeroed(&stream, n)?;
        if d == 2048 {
            b.backward_input_add_amax_xhat_f16(LayerNormBackwardInputAddAmaxArgs {
                stream: &stream,
                residual: &tape,
                d_normalized: &gd,
                mean: &mean,
                inv_std: &inv,
                weight,
                direct: &dd,
                d_residual: &mut dx,
                chunk_amax: &mut chunks,
                output_scale: 0.25,
                row_count: n as u32,
                embedding_dim: d as u32,
            })?;
        } else {
            b.backward_input_add_xhat_f16(LayerNormBackwardInputAddArgs {
                stream: &stream,
                residual: &tape,
                d_normalized: &gd,
                mean: &mean,
                inv_std: &inv,
                weight,
                direct: &dd,
                d_residual: &mut dx,
                output_scale: 0.25,
                row_count: n as u32,
                embedding_dim: d as u32,
            })?;
        }
        let mut dw = DeviceBuffer::zeroed(&stream, d)?;
        let mut db = DeviceBuffer::zeroed(&stream, d)?;
        b.backward_params_xhat_f16(LayerNormBackwardParamArgs {
            stream: &stream,
            residual: &tape,
            d_normalized: &gd,
            mean: &mean,
            inv_std: &inv,
            d_weight: &mut dw,
            d_bias: &mut db,
            output_scale: 0.25,
            row_count: n as u32,
            embedding_dim: d as u32,
        })?;
        let means = mean.to_host_vec(&stream)?;
        let inverse = inv.to_host_vec(&stream)?;
        let actual = dx.to_host_vec(&stream)?;
        let weightgrad = dw.to_host_vec(&stream)?;
        let biasgrad = db.to_host_vec(&stream)?;
        b.backward_params_xhat_f16(LayerNormBackwardParamArgs {
            stream: &stream,
            residual: &tape,
            d_normalized: &gd,
            mean: &mean,
            inv_std: &inv,
            d_weight: &mut dw,
            d_bias: &mut db,
            output_scale: 0.25,
            row_count: n as u32,
            embedding_dim: d as u32,
        })?;
        assert_eq!(
            weightgrad,
            dw.to_host_vec(&stream)?,
            "parameter gradients must overwrite each call"
        );
        assert_eq!(
            biasgrad,
            db.to_host_vec(&stream)?,
            "parameter gradients must overwrite each call"
        );
        let mut expected_w = vec![0.; d];
        let mut expected_b = vec![0.; d];
        for row in 0..n {
            let h = (0..d)
                .map(|j| {
                    (f64::from(x[row * d + j]) - f64::from(means[row])) * f64::from(inverse[row])
                })
                .collect::<Vec<_>>();
            let dy_sum = (0..d)
                .map(|j| f64::from(dy[row * d + j]) * 0.25)
                .sum::<f64>();
            let dot = (0..d)
                .map(|j| h[j] * f64::from(dy[row * d + j]) * 0.25)
                .sum::<f64>();
            for j in 0..d {
                let g = f64::from(dy[row * d + j]) * 0.25;
                let want = (g - dy_sum / d as f64 - h[j] * dot / d as f64)
                    * f64::from(inverse[row])
                    + f64::from(direct[row * d + j]);
                assert!(actual[row * d + j].is_finite());
                assert!(
                    (f64::from(actual[row * d + j]) - want).abs() < 1e-5 + want.abs() * 0.003,
                    "input derivative d={d} row={row} col={j}"
                );
                expected_w[j] += g * h[j];
                expected_b[j] += g;
            }
        }
        for j in 0..d {
            assert!(weightgrad[j].is_finite() && biasgrad[j].is_finite());
            assert!(
                (f64::from(weightgrad[j]) - expected_w[j]).abs()
                    < 8e-5 + expected_w[j].abs() * 0.003
            );
            assert!((f64::from(biasgrad[j]) - expected_b[j]).abs() < 1e-7);
        }
        println!(
            "bounded_normalization_tape d={d} max_raw=1000000 forward=bit_exact input_params_direct_amax=finite updates=0"
        );
    }
    Ok(())
}
