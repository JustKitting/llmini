//! Production full B8/S2048 MLP forward/backward, with stored-operand oracles.
//! This isolates the MLP; complete-model acceptance remains a separate gate.
use super::*;
use crate::{training::runtime::Runtime, upload::UploadedNvfp4};
use gpt2_nvfp4::*;
use rust_kernels_cuda::nvfp4::csf;
use rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;
#[path = "../../../crates/cuda-kernels/tests/common/f16_round.rs"]
mod half_input;

struct Operand {
    gpu: UploadedNvfp4,
    bytes: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
    k: usize,
}
impl Operand {
    fn new(s: &CudaStream, rows: usize, k: usize, salt: u32) -> AppResult<Self> {
        let bytes = (0..rows * k / 2)
            .map(|i| {
                let mut x = (i as u32).wrapping_add(salt).wrapping_mul(747796405);
                x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
                (x ^ (x >> 22)) as u8
            })
            .collect::<Vec<_>>();
        let raw = (0..rows * k / 16)
            .map(|i| 0x30 + ((i + salt as usize) % 7) as u8)
            .collect::<Vec<_>>();
        let scales = csf::encode_host(&raw, rows, k);
        let global = if rows == 1 { 1. / 32. } else { 1. / 64. };
        let gpu = UploadedNvfp4::from_host(s, &bytes, &scales, global, rows * k)?;
        Ok(Self {
            gpu,
            bytes,
            scales,
            global,
            k,
        })
    }
    fn at(&self, row: usize, col: usize) -> f64 {
        let i = row * self.k + col;
        decode(
            &self.bytes,
            csf::scale_at(&self.scales, i / 16),
            self.global,
            i,
        )
    }
}
fn decode(bytes: &[u8], scale: u8, global: f32, i: usize) -> f64 {
    let c = (bytes[i / 2] >> (4 * (i % 2))) & 15;
    let fp4 = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(c & 7) as usize];
    let exp = (scale >> 3) as i32;
    let frac = (scale & 7) as f64;
    let sf = if exp == 0 {
        frac / 512.
    } else {
        (1. + frac / 8.) * 2f64.powi(exp - 7)
    };
    (if c & 8 == 0 { fp4 } else { -fp4 }) * sf * f64::from(global)
}
fn row(s: &CudaStream, b: &DeviceBuffer<f32>, index: usize, width: usize) -> AppResult<Vec<f32>> {
    let mut out = vec![0.; width];
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            out.as_mut_ptr(),
            b.cu_deviceptr() + (index * width * 4) as u64,
            width * 4,
            s.cu_stream(),
        )?;
    }
    s.synchronize()?;
    Ok(out)
}
fn read_rowwise(
    s: &CudaStream,
    input: rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor<'_>,
    n: usize,
    k: usize,
    selected: &[usize],
) -> AppResult<Vec<Vec<f64>>> {
    let bytes = input.bytes.to_host_vec(s)?;
    let mut scales = vec![0; 128];
    scales[..4].copy_from_slice(&(k as u32).to_le_bytes());
    scales[4..8].copy_from_slice(&(n as u32).to_le_bytes());
    scales[8] = (k / 16).trailing_zeros() as u8;
    scales[9] = 7;
    scales.extend(input.scales.to_host_vec(s)?);
    let globals = input.global_scales.to_host_vec(s)?;
    Ok(selected
        .iter()
        .map(|&r| {
            (0..k)
                .map(|c| {
                    let i = r * k + c;
                    decode(&bytes, csf::scale_at(&scales, i / 16), globals[r], i)
                })
                .collect()
        })
        .collect())
}
fn close(actual: f32, expected: f64, tolerance: f64, label: &str) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
        "{label}: actual={actual:e} expected={expected:e} tolerance={tolerance:e}"
    );
}

#[test]
#[ignore = "native full B8/S2048 paired11008/product5504 production MLP, dense+routed; zero optimizer updates"]
fn native_full_bounded_mlp_stored_operands_and_complete_reverse_graph() -> AppResult {
    assert_eq!(
        (GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_MLP, GPT2_MLP_UP),
        (8, 2048, 5504, 11008)
    );
    let rt = Runtime::new()?;
    eprintln!("bounded_mlp_phase=matching_native_runtime_loaded");
    let s = rt.stream.as_ref();
    let n = GPT2_TOKEN_ROWS;
    let d = GPT2_N_EMBD;
    let h = GPT2_MLP;
    let up = Operand::new(s, 2 * h, d, 17)?;
    let ub = Operand::new(s, 1, 2 * h, 31)?;
    let down = Operand::new(s, d, h, 73)?;
    let db = Operand::new(s, 1, d, 113)?;
    let projections = MlpProjectionTensors {
        up: MlpUpTensors {
            weight: up.gpu.mma(),
            weight_device: up.gpu.device(),
            bias: ub.gpu.device(),
        },
        down: MlpDownTensors {
            weight: down.gpu.mma(),
            weight_device: down.gpu.device(),
            bias: db.gpu.device(),
        },
    };
    let selected = [0, 127, 128, 2047, 2048, n - 1];
    for block_index in [0, 1] {
        let mut normalized = DeviceBuffer::from_host(
            s,
            &(0..n * d)
                .map(|i| ((i * 37 + i / d * 13) % 257) as f32 / 64. - 2.)
                .collect::<Vec<_>>(),
        )?;
        let mut residual = DeviceBuffer::from_host(s, &vec![0.125; n * d])?;
        let mut amax = DeviceBuffer::from_host(s, &vec![2.; n])?;
        let mut mean = DeviceBuffer::zeroed(s, n)?;
        let mut inv = DeviceBuffer::from_host(s, &vec![1.; n])?;
        let mut paired = DeviceBuffer::zeroed(s, n * 2 * h)?;
        let mut activation = DeviceBuffer::zeroed(s, n * h)?;
        let mut scratch_input = RowwiseNvfp4Buffers::new(s, n * d, n)?;
        let mut scratch_act = RowwiseNvfp4Buffers::new(s, n * h, n)?;
        let mut saved_input = RowwiseNvfp4Buffers::new(s, n * d, n)?;
        let mut saved_act = RowwiseNvfp4Buffers::new(s, n * h, n)?;
        let mut masks = DeviceBuffer::zeroed(s, GPT2_MLP_ROUTE_MASKS)?;
        let mut scratch_masks = DeviceBuffer::zeroed(s, GPT2_MLP_ROUTE_MASKS)?;
        let mut scores = DeviceBuffer::zeroed(s, n / 128 * 43)?;
        let mut raw16 = DeviceBuffer::zeroed(s, 1)?;
        let mut desc = TmaNvfp4DeviceScaleDescriptors::new(s)?;
        let audit = RecomputeAudit {
            expected: RefCell::new(vec![None; 16]),
            checked: RefCell::new(vec![false; 16]),
        };
        MlpWeights::forward(MlpForwardArgs {
            projection_observer: Some(&audit),
            block_index,
            module: &rt.mlp,
            tma_module: &rt.tma_gemm,
            quant_module: &rt.quant,
            scratch: MlpScratch {
                paired: &mut paired,
                input_nvfp4: scratch_input.scratch(),
                activation_nvfp4: scratch_act.scratch(),
                activation: &mut activation,
                route_scores: &mut scores,
                route_masks: &mut scratch_masks,
                tma_descriptors: &mut desc,
            },
            projections,
            input_prequantized: false,
            hidden: HiddenStateDevice {
                stream: s,
                batch_size: 8,
                seq_len: 2048,
                row_count: n as u32,
                residual: &mut residual,
                normalized: &mut normalized,
                normalized_amax: &mut amax,
                mean: &mut mean,
                inv_std: &mut inv,
            },
            tape: Some(MlpForwardTape {
                up_input_nvfp4: saved_input.tape(),
                pre_activation_f16: &mut raw16,
                down_input_nvfp4: saved_act.tape(),
                route_masks: &mut masks,
            }),
        })?;
        eprintln!("bounded_mlp_phase=native_forward_complete block={block_index}");
        let route = masks.to_host_vec(s)?;
        if block_index == 1 {
            assert!(route[..n / 128]
                .iter()
                .all(|m| m.count_ones() == 32 && *m >> 43 == 0));
        }
        let inputs = read_rowwise(s, saved_input.saved(), n, d, &selected)?;
        let acts = read_rowwise(s, saved_act.saved(), n, h, &selected)?;
        let raw = selected
            .iter()
            .map(|&r| row(s, &paired, r, 2 * h))
            .collect::<AppResult<Vec<_>>>()?;
        for (sample, &r) in selected.iter().enumerate() {
            for col in [0, 127, 128, h - 1, h, 2 * h - 1] {
                let expected = ub.at(0, col)
                    + (0..d)
                        .map(|k| inputs[sample][k] * up.at(col, k))
                        .sum::<f64>();
                close(
                    raw[sample][col],
                    expected,
                    2e-5 + expected.abs() * 2e-5,
                    "paired native affine",
                );
            }
            let output = row(s, &residual, r, d)?;
            for col in [0, 127, 1023, d - 1] {
                let mut expected = 0.125 + db.at(0, col);
                let mut magnitude = 0.;
                for k in 0..h {
                    if block_index == 0 || route[r / 128] & (1 << (k / 128)) != 0 {
                        let product = acts[sample][k] * down.at(col, k);
                        expected += product;
                        magnitude += product.abs();
                    }
                }
                close(
                    output[col],
                    expected,
                    2e-5 + magnitude * 2e-5,
                    "native routed down affine+residual",
                );
            }
        }
        eprintln!("bounded_mlp_phase=stored_operand_affine_oracles_passed block={block_index}");
        let incoming = DeviceBuffer::from_host(
            s,
            &(0..n * d)
                .map(|i| ((i * 71) % 131) as f32 / 512. - 0.125)
                .collect::<Vec<_>>(),
        )?;
        let mut down_scratch = LinearScratch::new(s, h, d)?;
        let mut up_scratch = LinearScratch::new(s, d, 2 * h)?;
        let mut dact = DeviceBuffer::zeroed(s, n * h)?;
        let mut dx = DeviceBuffer::zeroed(s, n * d)?;
        let mut duw = DeviceBuffer::zeroed(s, d * 2 * h)?;
        let mut dub = DeviceBuffer::zeroed(s, 2 * h)?;
        let mut ddw = DeviceBuffer::zeroed(s, h * d)?;
        let mut ddb = DeviceBuffer::zeroed(s, d)?;
        let dummy = DeviceBuffer::zeroed(s, 1)?;
        let norm = LayerNormSaved {
            row_count: n as u32,
            residual: &raw16,
            mean: &mean,
            inv_std: &inv,
        };
        let saved = BlockForwardSaved {
            batch_size: 8,
            seq_len: 2048,
            row_count: n as u32,
            ln_1: norm,
            qkv_input_nvfp4: saved_input.saved(),
            qkv: &raw16,
            attention_out: &raw16,
            headwise_gate_input: None,
            attention_probs: None,
            kda_v_new: None,
            kda_akk_inv: None,
            kda_w: None,
            kda_aqk: None,
            attention_log_sum_exp: &dummy,
            c_proj_input_nvfp4: saved_input.saved(),
            ln_2: norm,
            mlp_up_input_nvfp4: saved_input.saved(),
            mlp_up: &raw16,
            mlp_down_input_nvfp4: saved_act.saved(),
            mlp_route_masks: &masks,
        };
        mlp_backward(MlpBackwardArgs {
            block_index,
            stream: s,
            modules: MlpBackwardModules {
                tma: &rt.tma_gemm,
                transpose: &rt.transpose,
                decode: &rt.decode,
                linear: &rt.linear,
                quant: &rt.quant,
                mlp: &rt.mlp,
            },
            saved,
            projections,
            d_residual_out: &incoming,
            precomputed_d_residual_amax_chunks: None,
            grads: MlpBackwardGrads {
                d_mlp_up: &mut dact,
                d_ln_2_normalized: &mut dx,
                d_c_proj_weight: &mut ddw,
                d_c_proj_bias: &mut ddb,
                d_c_fc_weight: &mut duw,
                d_c_fc_bias: &mut dub,
            },
            scratch: MlpBackwardScratch {
                paired: &mut paired,
                tma_descriptors: &mut desc,
                projection_observer: Some(&audit),
                down_linear: down_scratch.parts(),
                up_linear: up_scratch.parts(),
            },
            seeds: MlpBackwardSeeds::from_rng(&mut Gpt2Rng::new(0x47505432)),
        })?;
        assert!(audit.checked.borrow()[block_index]);
        for (sample, &r) in selected.iter().enumerate() {
            let e = row(s, &dact, r, h)?;
            let derivatives = row(s, &paired, r, 2 * h)?;
            for c in 0..h {
                let g = f64::from(half_input::round(raw[sample][c]));
                let v = f64::from(half_input::round(raw[sample][h + c]));
                let tg = (g / 4.).tanh();
                let tv = (v / 25.).tanh();
                let sigmoid = 1. / (1. + (-g).exp());
                let incoming = if block_index == 0 || route[r / 128] & (1 << (c / 128)) != 0 {
                    f64::from(e[c])
                } else {
                    0.
                };
                let dg = incoming
                    * 25.
                    * tv
                    * ((1. - tg * tg) * sigmoid + 4. * tg * sigmoid * (1. - sigmoid));
                let dv = incoming * 4. * tg * sigmoid * (1. - tv * tv);
                close(
                    derivatives[c],
                    dg,
                    2e-7 + dg.abs() * 2e-5,
                    "complete gate adjoint",
                );
                close(
                    derivatives[h + c],
                    dv,
                    2e-7 + dv.abs() * 2e-5,
                    "complete up adjoint",
                );
            }
        }
        for (name, b) in [
            ("up_weight", &duw),
            ("up_bias", &dub),
            ("down_weight", &ddw),
            ("down_bias", &ddb),
            ("input_adjoint", &dx),
        ] {
            let values = b.to_host_vec(s)?;
            assert!(values.iter().all(|x| x.is_finite()), "nonfinite {name}");
            assert!(values.iter().any(|x| *x != 0.), "dead {name}");
            if name == "up_weight" {
                assert!(
                    values[..d * h].iter().any(|x| *x != 0.)
                        && values[d * h..].iter().any(|x| *x != 0.)
                );
            }
        }
        println!(
            "native_bounded_mlp_verified B=8 S=2048 d=2048 paired=11008 product=5504 block={block_index} stored_NVFP4_affines=independent_oracle all_paired_recomputed_values=bit_exact masked_analytic_adjoints=verified all_reverse_groups=finite_active updates=0"
        );
    }
    Ok(())
}
