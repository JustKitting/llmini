use cuda_core::{CudaStream, DeviceBuffer};
mod common;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_quant::*;
use rust_kernels_cuda::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;
use std::error::Error;
struct Out {
    b: DeviceBuffer<u8>,
    s: DeviceBuffer<u8>,
    a: DeviceBuffer<f32>,
    global: DeviceBuffer<f32>,
    rows: usize,
    k: usize,
}
impl Out {
    fn new(st: &CudaStream, rows: usize, k: usize) -> Result<Self, Box<dyn Error>> {
        let scratch = rust_kernels_cuda::linear_backward::MsEdenOperandScratchBuffer::new(
            st,
            rows * k,
            rows,
        )?;
        Ok(Self {
            b: scratch.bytes,
            s: scratch.scales,
            a: scratch.chunk_amax,
            global: scratch.global_scale,
            rows,
            k,
        })
    }
    fn digest(
        &self,
        st: &CudaStream,
        scalar: &DeviceBuffer<f32>,
    ) -> Result<(usize, u64), Box<dyn Error>> {
        let raw = self.s.to_host_vec(st)?;
        let decoded = decode(&raw, self.rows, self.k);
        let globals = vec![scalar.to_host_vec(st)?[0]; self.rows];
        let bytes = self.b.to_host_vec(st)?;
        let mut all = bytes.clone();
        all.extend(&decoded);
        for x in globals {
            all.extend(x.to_le_bytes());
        }
        let hash = all.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ (b as u64)).wrapping_mul(0x100000001b3)
        });
        Ok((all.len(), hash))
    }
    fn emit(
        &self,
        st: &CudaStream,
        name: &str,
        scalar: &DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let (len, hash) = self.digest(st, scalar)?;
        let (want_len, want_hash) = current_rounding_golden(name);
        assert_eq!(len, want_len, "{name} length");
        assert_eq!(
            hash, want_hash,
            "{name} payload, scale bytes and global scales"
        );
        println!("exact current_stochastic_recipe {name} bytes={len} digest={hash:016x}");

        Ok(())
    }
}
fn decode(p: &[u8], rows: usize, k: usize) -> Vec<u8> {
    (0..rows)
        .flat_map(|r| (0..k / 16).map(move |kg| decode_one_six(p, rows, k, r, kg)))
        .collect()
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn direct_compressed_quantization_preserves_original_bytes() -> Result<(), Box<dyn Error>> {
    run_direct_quantizers(|out, st, name, scalar| out.emit(st, name, scalar))
}

fn run_direct_quantizers(
    mut verify: impl FnMut(&Out, &CudaStream, &str, &DeviceBuffer<f32>) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(ptx)?;
    for (rows, cols) in [(128usize, 128usize), (256, 2048)] {
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let row = i / cols;
                let col = i % cols;
                let a = (((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) >> 9) % 65536)
                    as f32
                    / 32768.0
                    - 1.0;
                if row % 17 == 0 {
                    0.0
                } else {
                    a * 2.0f32.powi((col / 128 % 7) as i32 - 3)
                }
            })
            .collect();
        let xd = DeviceBuffer::from_host(&st, &x)?;
        let mut row = Out::new(&st, rows, cols)?;
        q.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
            QuartetBackwardMsEdenDeviceScaleQuantArgs {
                stream: &st,
                x: &xd,
                out_fp4: &mut row.b,
                out_scales: &mut row.s,

                out_chunk_amax: &mut row.a,
                out_global_scale: &mut row.global,
                row_count: rows as u32,
                src_row_len: cols as u32,
                dst_row_len: cols as u32,
                sign_seed: 77,
                scale_seed: 89,
            },
        )?;
        verify(&row, &st, &format!("row-{rows}-{cols}"), &row.global)?;
        let mut tr = Out::new(&st, cols, rows)?;
        q.fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
            MsEdenTransposeDeviceScaleQuantArgs {
                stream: &st,
                x: &xd,
                out_fp4: &mut tr.b,
                out_scales: &mut tr.s,

                out_chunk_amax: &mut tr.a,
                global_scale: &row.global,
                source_rows: rows as u32,
                source_cols: cols as u32,
                dst_row_len: rows as u32,
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: 77,
                scale_seed: 89,
            },
        )?;
        verify(&tr, &st, &format!("transpose-{rows}-{cols}"), &row.global)?;
        for with_bias in [false, true] {
            let mut r = Out::new(&st, rows, cols)?;
            let mut t = Out::new(&st, cols, rows)?;
            let mut bias = DeviceBuffer::<f32>::zeroed(&st, cols)?;
            let args = MsEdenPairDeviceScaleQuantArgs {
                stream: &st,
                x: &xd,
                out_fp4: &mut r.b,
                out_scales: &mut r.s,

                transpose_out_fp4: &mut t.b,
                transpose_out_scales: &mut t.s,

                out_chunk_amax: &mut r.a,
                out_global_scale: &mut r.global,
                row_count: rows as u32,
                src_row_len: cols as u32,
                dst_row_len: cols as u32,
                transpose_dst_row_len: rows as u32,
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: 77,
                scale_seed: 89,
                transpose_scale_seed: 97,
                precomputed_chunk_count: None,
            };
            if with_bias {
                assert!(q.fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_with_bias(args,&mut bias)?);
                let actual = bias.to_host_vec(&st)?;
                for col in 0..cols {
                    let want = (0..rows).map(|i| x[i * cols + col] as f64).sum::<f64>();
                    assert!((actual[col] as f64 - want).abs() < 0.001, "bias col={col}");
                }
            } else {
                q.fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                    args,
                )?;
            }
            verify(
                &r,
                &st,
                &format!("pair-row-{with_bias}-{rows}-{cols}"),
                &r.global,
            )?;
            verify(
                &t,
                &st,
                &format!("pair-transpose-{with_bias}-{rows}-{cols}"),
                &r.global,
            )?;
        }
        let b: Vec<u8> = (0..rows * cols / 2)
            .map(|i| [0x21, 0x32, 0x14, 0x95][i % 4])
            .collect();
        let s: Vec<u8> = (0..rows * cols / 16)
            .map(|i| 0x30 + ((i * 13) % 25) as u8)
            .collect();
        let g: Vec<f32> = (0..rows).map(|i| 0.01 + 0.001 * (i % 7) as f32).collect();
        let bd = DeviceBuffer::from_host(&st, &b)?;
        let sd = DeviceBuffer::from_host(
            &st,
            &rust_kernels_cuda::nvfp4::csf::encode_rowwise_host(&s, rows, cols),
        )?;
        let primary = DeviceBuffer::from_host(
            &st,
            &rust_kernels_cuda::nvfp4::csf::encode_host(&s, rows, cols),
        )?;
        let gd = DeviceBuffer::from_host(&st, &g)?;
        let scalar = DeviceBuffer::from_host(&st, &[0.01f32])?;
        let mut n = Out::new(&st, cols, rows)?;
        q.nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
            Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
                stream: &st,
                input: Nvfp4DeviceTensor::new(&bd, &primary, &scalar),
                out_fp4: &mut n.b,
                out_scales: &mut n.s,

                out_chunk_amax: &mut n.a,
                out_global_scale: &mut n.global,
                source_rows: rows as u32,
                source_cols: cols as u32,
                dst_row_len: rows as u32,
                sign_seed: 77,
                scale_seed: 89,
            },
        )?;
        verify(
            &n,
            &st,
            &format!("nvfp4-transpose-{rows}-{cols}"),
            &n.global,
        )?;
        let mut n = Out::new(&st, cols, rows)?;
        q.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
            RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs {
                stream: &st,
                input: Nvfp4RowwiseDeviceTensor::new(&bd, &sd, &gd),
                out_fp4: &mut n.b,
                out_scales: &mut n.s,

                out_chunk_amax: &mut n.a,
                out_global_scale: &mut n.global,
                source_rows: rows as u32,
                source_cols: cols as u32,
                dst_row_len: rows as u32,
                sign_seed: 77,
                scale_seed: 89,
            },
        )?;
        verify(
            &n,
            &st,
            &format!("rowwise-transpose-{rows}-{cols}"),
            &n.global,
        )?;
    }
    println!("PASS direct quantization bytes and scales");
    Ok(())
}

fn current_rounding_golden(name: &str) -> (usize, u64) {
    include_str!("fixtures/csf-stochastic-rounding-20261115.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            (fields.len() == 3 && fields[0] == name).then(|| {
                (
                    fields[1].parse().unwrap(),
                    u64::from_str_radix(fields[2], 16).unwrap(),
                )
            })
        })
        .unwrap_or_else(|| panic!("missing current rounding fixture {name}"))
}

#[test]
#[ignore = "reference capture only; requires the exact hash-bound pre-extension stochastic-rounding PTX"]
fn capture_preserved_stochastic_recipe_direct_quantizer_reference() -> Result<(), Box<dyn Error>> {
    let path = std::env::var("CUDA_OXIDE_PTX_PATH")?;
    let digest = std::process::Command::new("sha256sum")
        .arg(&path)
        .output()?;
    assert!(digest.status.success());
    assert_eq!(
        std::str::from_utf8(&digest.stdout)?
            .split_whitespace()
            .next(),
        Some("16ca641357b34ca0ee190e50fa9e285fd4cc721b1d24afcbed743e802efae99e")
    );
    let mut names = std::collections::BTreeSet::new();
    run_direct_quantizers(|out, st, name, scalar| {
        let (len, hash) = out.digest(st, scalar)?;
        assert_eq!(len, golden(name).0);
        assert!(names.insert(name.to_string()));
        println!("STOCHASTIC_REFERENCE {name} {len} {hash:016x}");
        Ok(())
    })?;
    assert_eq!(names.len(), 16);
    Ok(())
}

// Historical pre-stability outputs, retained verbatim. These predate the
// upward-covering scales, post-Hadamard headroom and seeded stochastic payloads
// repaired in notes/backward_requantization_stability_20261114.md. The direct
// regression above now uses the separately hash-bound current-recipe fixture.
// Each digest covers all FP4 bytes, decoded E4M3 scale bytes, and FP32 row scales.
fn golden(name: &str) -> (usize, u64) {
    match name {
        "nvfp4-transpose-128-128" => (9728, 0x7a2bacd2a9c7fef5),
        "nvfp4-transpose-256-2048" => (303104, 0x7c6d3d41d6fdf91d),
        "pair-row-false-128-128" => (9728, 0x6c636447a80705ac),
        "pair-row-false-256-2048" => (295936, 0x1db8c1bae05351f8),
        "pair-row-true-128-128" => (9728, 0x6c636447a80705ac),
        "pair-row-true-256-2048" => (295936, 0x1db8c1bae05351f8),
        "pair-transpose-false-128-128" => (9728, 0x6b92202d73ad5db9),
        "pair-transpose-false-256-2048" => (303104, 0xdc7c490edbcd21e2),
        "pair-transpose-true-128-128" => (9728, 0x6b92202d73ad5db9),
        "pair-transpose-true-256-2048" => (303104, 0xdc7c490edbcd21e2),
        "row-128-128" => (9728, 0x6c636447a80705ac),
        "row-256-2048" => (295936, 0x1db8c1bae05351f8),
        "rowwise-transpose-128-128" => (9728, 0x928372a8f62f195d),
        "rowwise-transpose-256-2048" => (303104, 0x7036c1328bd35c15),
        "transpose-128-128" => (9728, 0x6b92202d73ad5db9),
        "transpose-256-2048" => (303104, 0xdc7c490edbcd21e2),
        _ => panic!("unknown fixture {name}"),
    }
}

// Captured from accepted 3cef1e18 / PTX ed481715982c2173 before changing the
// paired kernel's lane ownership. Cover every physical width in the 1B trace.
#[ignore = "requires generated sm_120a PTX"]
#[test]
fn paired_csf_full_training_shapes_preserve_original_bytes() -> Result<(), Box<dyn Error>> {
    for (cols, active, row, trans) in [
        (8192, 8192, 0x258705cdb73cf2e3, 0x5460038dfe95083b),
        (2048, 2048, 0xc767db537bb93848, 0xe209c1a7ee267d69),
        (8320, 8320, 0xe627903e0762f69b, 0x0d53564598e8a387),
        (32000, 32000, 0xcaae3e499b3dd73d, 0x13d4b585c9932905),
        (6272, 6272, 0x7076b4b9dbd97345, 0x650f4d41c293d152),
        (4096, 4096, 0x1345a54fe6703b1c, 0x93fb35508258bf8c),
        (6272, 6176, 0xe1a2a0430404dbcd, 0x4663fbd48a7b926f),
    ] {
        check_pair_csf_reference(8192, cols, active, row, trans)?;
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn paired_csf_multi_super_rows_preserve_original_bytes() -> Result<(), Box<dyn Error>> {
    check_pair_csf_reference(16384, 128, 128, 0x3b68828e488fea7c, 0x03a70b5b0d5d5ea9)
}

fn check_pair_csf_reference(
    rows: usize,
    cols: usize,
    active: usize,
    expected_row: u64,
    expected_trans: u64,
) -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(ptx)?;
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            let a = (((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) >> 9) % 65536)
                as f32
                / 32768.0
                - 1.0;
            if row % 17 == 0 || col >= active {
                0.0
            } else {
                a * 2.0_f32.powi((col / 128 % 7) as i32 - 3)
            }
        })
        .collect();
    let xd = DeviceBuffer::from_host(&st, &x)?;
    let mut row = Out::new(&st, rows, cols)?;
    let mut trans = Out::new(&st, cols, rows)?;
    let mut bias = DeviceBuffer::<f32>::zeroed(&st, cols)?;
    assert!(
        q.fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_with_bias(
            MsEdenPairDeviceScaleQuantArgs {
                stream: &st,
                x: &xd,
                out_fp4: &mut row.b,
                out_scales: &mut row.s,
                transpose_out_fp4: &mut trans.b,
                transpose_out_scales: &mut trans.s,
                out_chunk_amax: &mut row.a,
                out_global_scale: &mut row.global,
                row_count: rows as u32,
                src_row_len: cols as u32,
                dst_row_len: cols as u32,
                transpose_dst_row_len: rows as u32,
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: 77,
                scale_seed: 89,
                transpose_scale_seed: 97,
                precomputed_chunk_count: None,
            },
            &mut bias,
        )?
    );
    let global = row.global.to_host_vec(&st)?[0].to_le_bytes();
    for (label, output, expected) in [
        ("row", &row, expected_row),
        ("transpose", &trans, expected_trans),
    ] {
        let bytes = output.b.to_host_vec(&st)?;
        let scales = decode(&output.s.to_host_vec(&st)?, output.rows, output.k);
        let mut digest = 0xcbf29ce484222325_u64;
        for &byte in bytes.iter().chain(&scales) {
            digest = (digest ^ byte as u64).wrapping_mul(0x100000001b3);
        }
        for _ in 0..output.rows {
            for byte in global {
                digest = (digest ^ byte as u64).wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(digest, expected, "{label} {rows}x{cols} active={active}");
    }
    let actual = bias.to_host_vec(&st)?;
    let mut expected = vec![0.0f64; cols];
    let mut absolute = vec![0.0f64; cols];
    for source_row in x.chunks_exact(cols) {
        for (col, &value) in source_row.iter().enumerate() {
            expected[col] += value as f64;
            absolute[col] += (value as f64).abs();
        }
    }
    for col in 0..cols {
        assert!(
            (actual[col] as f64 - expected[col]).abs() <= 1e-5 + absolute[col] * 2e-6,
            "bias {rows}x{cols} active={active} col={col}"
        );
    }
    println!(
        "exact paired CSF {rows}x{cols} active={active}: FP4 bytes, scale bytes and global scales; bias agrees with F64 bound"
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn primary_transpose_preserves_active_prefix_with_physical_padding() -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(ptx)?;
    for (active, stored, cols) in [(128usize, 256usize, 128usize), (6272, 8320, 2048)] {
        let bytes: Vec<u8> = (0..stored * cols / 2)
            .map(|i| [0x21, 0x32, 0x14, 0x95][i % 4])
            .collect();
        let scales: Vec<u8> = (0..stored * cols / 16)
            .map(|i| 0x30 + ((i * 13) % 25) as u8)
            .collect();
        let physical = rust_kernels_cuda::nvfp4::csf::encode_host(&scales, stored, cols);
        let logical =
            rust_kernels_cuda::nvfp4::csf::encode_host(&scales[..active * cols / 16], active, cols);
        let pb = DeviceBuffer::from_host(&st, &bytes)?;
        let ps = DeviceBuffer::from_host(&st, &physical)?;
        let lb = DeviceBuffer::from_host(&st, &bytes[..active * cols / 2])?;
        let ls = DeviceBuffer::from_host(&st, &logical)?;
        let global = DeviceBuffer::from_host(&st, &[0.01f32])?;
        let mut outputs = Vec::new();
        for input in [
            Nvfp4DeviceTensor::new(&lb, &ls, &global),
            Nvfp4DeviceTensor::new(&pb, &ps, &global),
        ] {
            let mut out = Out::new(&st, cols, active)?;
            q.nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
                    stream: &st,
                    input,
                    out_fp4: &mut out.b,
                    out_scales: &mut out.s,
                    out_chunk_amax: &mut out.a,
                    out_global_scale: &mut out.global,
                    source_rows: active as u32,
                    source_cols: cols as u32,
                    dst_row_len: active as u32,
                    sign_seed: 77,
                    scale_seed: 89,
                },
            )?;
            outputs.push(out);
        }
        assert_eq!(
            outputs[0].b.to_host_vec(&st)?,
            outputs[1].b.to_host_vec(&st)?,
            "FP4 active={active} stored={stored}"
        );
        let a = decode(&outputs[0].s.to_host_vec(&st)?, cols, active);
        let b = decode(&outputs[1].s.to_host_vec(&st)?, cols, active);
        assert_eq!(a, b, "scales active={active} stored={stored}");
        assert_eq!(
            outputs[0].global.to_host_vec(&st)?,
            outputs[1].global.to_host_vec(&st)?
        );
        assert_eq!(
            outputs[0].global.to_host_vec(&st)?,
            outputs[1].global.to_host_vec(&st)?
        );
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn persistent_materialization_128_columns_preserves_original_bytes() -> Result<(), Box<dyn Error>> {
    use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantArgs;
    use rust_kernels_cuda::optimizer::{OptimizerModule, ScheduleFreeMaterializeArgs};
    const LEN: usize = 128 * 128;
    let (_, st, ptx) = common::cuda_test_context()?;
    let quant = Nvfp4QuantModule::from_module(ptx.clone())?;
    let opt = OptimizerModule::from_module(ptx)?;
    let x: Vec<f32> = (0..LEN)
        .map(|i| (((i * 17 + 13) % 251) as f32 - 125.) * 0.001)
        .collect();
    let z: Vec<f32> = (0..LEN)
        .map(|i| (((i * 29 + 7) % 277) as f32 - 138.) * 0.001)
        .collect();
    let xd = DeviceBuffer::from_host(&st, &x)?;
    let zd = DeviceBuffer::from_host(&st, &z)?;
    let mut amax = DeviceBuffer::from_host(&st, &[x.iter().fold(0f32, |a, b| a.max(b.abs()))])?;
    let mut chunks = DeviceBuffer::<f32>::zeroed(&st, LEN.div_ceil(256))?;
    // Digests of payload, decoded E4M3 bytes and global scale from the original PTX.
    for (scheduled, expected) in [(false, 0x8b6e8a7e1dfff073u64), (true, 0xd8f0e69e822b79df)] {
        let mut b = DeviceBuffer::<u8>::zeroed(&st, LEN / 2)?;
        let initial = rust_kernels_cuda::nvfp4::csf::encode_host(&vec![0; LEN / 16], 128, 128);
        let mut s = DeviceBuffer::from_host(&st, &initial)?;
        let mut g = DeviceBuffer::<f32>::zeroed(&st, 1)?;
        if scheduled {
            opt.materialize_schedule_free(ScheduleFreeMaterializeArgs {
                stream: &st,
                bytes: &mut b,
                scales: &mut s,
                global_scale: &mut g,
                z_master: &zd,
                x_master: &xd,
                amax: &mut amax,
                chunk_amax: &mut chunks,
                len: LEN as u32,
                scale_rows: 128,
                scale_k: 128,
                beta: 0.7,
                symexp_lin_beta: 12.5,
                symexp_lin_scales: None,
            })?;
        } else {
            quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
                stream: &st,
                x: &xd,
                amax: &amax,
                out_fp4: &mut b,
                out_scales: &mut s,
                out_global_scale: &mut g,
                group_count: (LEN / 16) as u32,
            })?;
        }
        let mut all = b.to_host_vec(&st)?;
        let packed = s.to_host_vec(&st)?;
        all.extend((0..LEN / 16).map(|i| rust_kernels_cuda::nvfp4::csf::scale_at(&packed, i)));
        all.extend(g.to_host_vec(&st)?[0].to_le_bytes());
        let digest = all.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        assert_eq!(all.len(), 9220);
        assert_eq!(
            digest, expected,
            "128-column persistent materialization, scheduled={scheduled}"
        );
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn canonical_rowwise_materialization_preserves_original_bytes() -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(ptx)?;
    // FNV-1a of payload + original E4M3 bytes + row globals from the native
    // quantizer. Both producer variants must match that independent oracle.
    for (rows, cols, hash) in [
        (1usize, 16usize, 0x7ae32c9f6169785eu64),
        (3, 80, 0xf787ea4ca1e768fc),
        (3, 128, 0xba595e1d681332c4),
        (129, 256, 0x484eb3d70045c086),
        (17, 2048, 0xabe7dbf7ea809134),
        (17, 8192, 0x65fa8ffa758b9dc1),
    ] {
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| {
                if i / cols % 17 == 0 {
                    0.
                } else {
                    (((i * 17 + 13) % 251) as f32 - 125.) * 0.003
                }
            })
            .collect();
        let xd = DeviceBuffer::from_host(&st, &x)?;
        for fused in [false, true] {
            let mut a = DeviceBuffer::<f32>::zeroed(&st, rows)?;
            let mut b = DeviceBuffer::<u8>::zeroed(&st, rows * cols / 2)?;
            let mut s = DeviceBuffer::<u8>::zeroed(
                &st,
                rust_kernels_cuda::nvfp4::csf::rowwise_encoded_len(rows, cols),
            )?;
            let mut g = DeviceBuffer::<f32>::zeroed(&st, rows)?;
            if fused {
                q.fp32_to_nvfp4_four_six_rowwise_derived_amax(Nvfp4QuantRowwiseDerivedAmaxArgs {
                    stream: &st,
                    x: &xd,
                    amax: &mut a,
                    out_fp4: &mut b,
                    out_scales: &mut s,
                    out_global_scale: &mut g,
                    row_count: rows as u32,
                    row_len: cols as u32,
                })?;
            } else {
                q.row_amax_f32(RowAmaxArgs {
                    stream: &st,
                    x: &xd,
                    out: &mut a,
                    row_count: rows as u32,
                    row_len: cols as u32,
                })?;
                q.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
                    stream: &st,
                    x: &xd,
                    amax: &a,
                    out_fp4: &mut b,
                    out_scales: &mut s,
                    out_global_scale: &mut g,
                    group_count: (rows * cols / 16) as u32,
                    row_len: cols as u32,
                })?;
            }
            let k = cols.div_ceil(128) * 128;
            let scales = decode(&s.to_host_vec(&st)?, rows, k);
            let mut all = b.to_host_vec(&st)?;
            for row in 0..rows {
                all.extend(&scales[row * k / 16..row * k / 16 + cols / 16]);
            }
            for v in g.to_host_vec(&st)? {
                all.extend(v.to_le_bytes());
            }
            let actual = all.iter().fold(0xcbf29ce484222325u64, |h, &v| {
                (h ^ u64::from(v)).wrapping_mul(0x100000001b3)
            });
            assert_eq!(
                actual, hash,
                "rowwise rows={rows} cols={cols} fused={fused}"
            );
        }
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn zero_primary_and_rowwise_scales_preserve_real_amax() -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(ptx)?;
    let rows = 128;
    let cols = 128;
    let bytes = DeviceBuffer::from_host(&st, &vec![0u8; rows * cols / 2])?;
    let raw = vec![0u8; rows * cols / 16];
    let primary = DeviceBuffer::from_host(
        &st,
        &rust_kernels_cuda::nvfp4::csf::encode_host(&raw, rows, cols),
    )?;
    let rowwise = DeviceBuffer::from_host(
        &st,
        &rust_kernels_cuda::nvfp4::csf::encode_rowwise_host(&raw, rows, cols),
    )?;
    let scalar = DeviceBuffer::from_host(&st, &[1.0f32])?;
    let globals = DeviceBuffer::from_host(&st, &vec![1.0f32; rows])?;
    for is_rowwise in [false, true] {
        let mut out = Out::new(&st, cols, rows)?;
        if is_rowwise {
            q.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs{stream:&st,input:Nvfp4RowwiseDeviceTensor::new(&bytes,&rowwise,&globals),out_fp4:&mut out.b,out_scales:&mut out.s,out_chunk_amax:&mut out.a,out_global_scale:&mut out.global,source_rows:rows as u32,source_cols:cols as u32,dst_row_len:rows as u32,sign_seed:77,scale_seed:89})?;
        } else {
            q.nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
                    stream: &st,
                    input: Nvfp4DeviceTensor::new(&bytes, &primary, &scalar),
                    out_fp4: &mut out.b,
                    out_scales: &mut out.s,
                    out_chunk_amax: &mut out.a,
                    out_global_scale: &mut out.global,
                    source_rows: rows as u32,
                    source_cols: cols as u32,
                    dst_row_len: rows as u32,
                    sign_seed: 77,
                    scale_seed: 89,
                },
            )?;
        }
        assert_eq!(out.global.to_host_vec(&st)?, [1.0]);
        assert!(out.b.to_host_vec(&st)?.iter().all(|&v| v & 0x77 == 0));
        assert!(
            decode(&out.s.to_host_vec(&st)?, cols, rows)
                .iter()
                .all(|&v| v == 0)
        );
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn six_bit_records_preserve_controlled_scales() -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(ptx)?;
    for (amplitudes, expected, _base, _extension) in [
        (
            [1.0f32, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0],
            [0x60u8, 0x64, 0x68, 0x6a, 0x6c, 0x70, 0x74, 0x78],
            0x60u8,
            0x5400u16,
        ),
        (
            [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
            [0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0x74, 0x78],
            0x48,
            0xea50,
        ),
        (
            [
                1.0 / 1024.0,
                0.125,
                1.0 / 128.0,
                8.0,
                0.5,
                1.0 / 512.0,
                4.0,
                1.0 / 256.0,
            ],
            [0x10, 0x48, 0x28, 0x78, 0x58, 0x18, 0x70, 0x20],
            255,
            0,
        ),
    ] {
        let x: Vec<f32> = (0..128 * 128).map(|i| amplitudes[(i % 128) / 16]).collect();
        let xd = DeviceBuffer::from_host(&st, &x)?;
        let mut out = Out::new(&st, 128, 128)?;
        let mut row_globals = DeviceBuffer::<f32>::zeroed(&st, 128)?;
        let mut amax = DeviceBuffer::<f32>::zeroed(&st, 128)?;
        q.fp32_to_nvfp4_four_six_rowwise_derived_amax(Nvfp4QuantRowwiseDerivedAmaxArgs {
            stream: &st,
            x: &xd,
            amax: &mut amax,
            out_fp4: &mut out.b,
            out_scales: &mut out.s,
            out_global_scale: &mut row_globals,
            row_count: 128,
            row_len: 128,
        })?;
        let s = out.s.to_host_vec(&st)?;
        let logical = decode(&s, 128, 128);
        for row in 0..128 {
            assert_eq!(&logical[row * 8..row * 8 + 8], &expected);
            assert_eq!(
                s[768 + row],
                expected
                    .iter()
                    .enumerate()
                    .fold(0u8, |m, (i, &v)| m | (((v >> 6) & 1) << i))
            );
        }
        assert!(out.b.to_host_vec(&st)?.iter().all(|&b| b == 0x77));
        let x = DeviceBuffer::from_host(&st, &x[..128])?;
        let a = DeviceBuffer::from_host(&st, &[8.0f32])?;
        let mut b = DeviceBuffer::<u8>::zeroed(&st, 64)?;
        let mut s = DeviceBuffer::from_host(
            &st,
            &rust_kernels_cuda::nvfp4::csf::encode_host(&[0; 8], 1, 128),
        )?;
        let mut g = DeviceBuffer::<f32>::zeroed(&st, 1)?;
        q.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
            stream: &st,
            x: &x,
            amax: &a,
            out_fp4: &mut b,
            out_scales: &mut s,
            out_global_scale: &mut g,
            group_count: 8,
        })?;
        let s = s.to_host_vec(&st)?;
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(rust_kernels_cuda::nvfp4::csf::scale_at(&s, i), v);
        }
        assert_eq!(&s[128..136], &expected);
        assert!(b.to_host_vec(&st)?.iter().all(|&v| v == 0x77));
    }
    Ok(())
}

fn decode_one_six(p: &[u8], rows: usize, k: usize, row: usize, kg: usize) -> u8 {
    let tiles = rows.div_ceil(128) * (k / 128);
    let stride = p.len() / tiles;
    let bytes = (stride - 768) / 128;
    let t = row / 128 * (k / 128) + kg / 8;
    let r = row % 128;
    let at = t * stride + (r / 16) * 96 + (r % 8) * 12 + (r % 16 / 8) * 3 + (kg % 8 / 4) * 6;
    let word = u32::from(p[at]) | (u32::from(p[at + 1]) << 8) | (u32::from(p[at + 2]) << 16);
    let q = ((word >> ((kg % 4) * 6)) & 63) as u8;
    let at = t * stride + 768 + r * bytes;
    if bytes == 1 {
        q | (((p[at] >> (kg % 8)) & 1) << 6)
    } else {
        q | (((p[at + kg % 8 / 4] >> ((kg % 4) * 2)) & 3) << 6)
    }
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_full_training_shapes_preserve_original_bytes() -> Result<(), Box<dyn Error>> {
    for (rows, cols, stored, k, primary, rowwise) in [
        (
            8192,
            8192,
            8192,
            8192,
            0x381e3e8d285cca7au64,
            0x421bf3b1109260feu64,
        ),
        (
            8192,
            2048,
            8192,
            8192,
            0x9ef57892aaec9210u64,
            0x3072967bdc353c5au64,
        ),
        (
            8192,
            4096,
            8192,
            8192,
            0xf8a4aa7cbde3aa33u64,
            0x07d2e51b0a10e41au64,
        ),
        (
            2048,
            8192,
            2048,
            2048,
            0x0b6789f35029f1c0u64,
            0x236dd55edaf3d0e7u64,
        ),
        (
            8320,
            2048,
            8320,
            8320,
            0x8dba93c1b611c954u64,
            0x2bde11b344c5aed5u64,
        ),
        (
            2048,
            2048,
            2048,
            2048,
            0x20977dd48c636a58u64,
            0x444a20dcfeb58c9eu64,
        ),
        (
            32000,
            2048,
            32000,
            32000,
            0xa11566b3f01ecde8u64,
            0x7fbcf61e9ec3738au64,
        ),
        (
            6272,
            2048,
            6272,
            6272,
            0x3d89992b75be557eu64,
            0xca3d0693039b304du64,
        ),
        (
            4096,
            4096,
            4096,
            4096,
            0x6c7e65a979364dc3u64,
            0x6f8be45ee8f8a48fu64,
        ),
        (
            2048,
            4096,
            2048,
            2048,
            0x1837556919791ae8u64,
            0x502b29cacf2c9cc5u64,
        ),
    ] {
        check_nvfp4_transpose_reference(rows, cols, stored, k, primary, rowwise)?;
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_stored_prefix_preserves_original_bytes() -> Result<(), Box<dyn Error>> {
    for (rows, cols, stored, k, primary, rowwise) in [(
        6272,
        2048,
        8320,
        6272,
        0x3d89992b75be557eu64,
        0xca3d0693039b304du64,
    )] {
        check_nvfp4_transpose_reference(rows, cols, stored, k, primary, rowwise)?;
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_zero_padding_preserves_original_bytes() -> Result<(), Box<dyn Error>> {
    for (rows, cols, stored, k, primary, rowwise) in [(
        128,
        128,
        256,
        256,
        0x1bc09cd16f56ec1bu64,
        0x4263db8ffcd31286u64,
    )] {
        check_nvfp4_transpose_reference(rows, cols, stored, k, primary, rowwise)?;
    }
    Ok(())
}

// Captured from accepted cc00ca9c / PTX aaa791e2 before changing this producer.
// Covers signed FP4 zero, every FP4 code, zero globals, tiny and large scales.
fn check_nvfp4_transpose_reference(
    rows: usize,
    cols: usize,
    stored: usize,
    k: usize,
    primary: u64,
    rowwise: u64,
) -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(ptx)?;
    let b: Vec<u8> = (0..stored * cols / 2)
        .map(|i| ((i as u32).wrapping_mul(37).wrapping_add(11) & 255) as u8)
        .collect();
    let codes = [0x00u8, 0x01, 0x02, 0x20, 0x40, 0x5f, 0x70, 0x7e];
    let scales: Vec<u8> = (0..stored * cols / 16)
        .map(|i| codes[(i * 13 + i / 17) % codes.len()])
        .collect();
    let globals: Vec<f32> = (0..stored)
        .map(|i| {
            if i % 17 == 0 {
                0.0
            } else {
                0.01 + 0.001 * (i % 7) as f32
            }
        })
        .collect();
    let bd = DeviceBuffer::from_host(&st, &b)?;
    let sd = DeviceBuffer::from_host(
        &st,
        &rust_kernels_cuda::nvfp4::csf::encode_rowwise_host(&scales, stored, cols),
    )?;
    let pd = DeviceBuffer::from_host(
        &st,
        &rust_kernels_cuda::nvfp4::csf::encode_host(&scales, stored, cols),
    )?;
    let gd = DeviceBuffer::from_host(&st, &globals)?;
    let scalar = DeviceBuffer::from_host(&st, &[0.01f32])?;
    for use_rowwise in [false, true] {
        let mut n = Out::new(&st, cols, k)?;
        // Force every required payload and compressed-scale field to be written,
        // including the zero-padded source rows.
        n.b = DeviceBuffer::from_host(&st, &vec![0xa5u8; n.b.len()])?;
        n.s = DeviceBuffer::from_host(&st, &vec![0xa5u8; n.s.len()])?;
        if use_rowwise {
            q.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs{
                stream:&st,input:Nvfp4RowwiseDeviceTensor::new(&bd,&sd,&gd),out_fp4:&mut n.b,out_scales:&mut n.s,out_chunk_amax:&mut n.a,out_global_scale:&mut n.global,source_rows:rows as u32,source_cols:cols as u32,dst_row_len:k as u32,sign_seed:77,scale_seed:89,
            })?;
        } else {
            q.nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
                    stream: &st,
                    input: Nvfp4DeviceTensor::new(&bd, &pd, &scalar),
                    out_fp4: &mut n.b,
                    out_scales: &mut n.s,
                    out_chunk_amax: &mut n.a,
                    out_global_scale: &mut n.global,
                    source_rows: rows as u32,
                    source_cols: cols as u32,
                    dst_row_len: k as u32,
                    sign_seed: 77,
                    scale_seed: 89,
                },
            )?;
        }
        let payload = n.b.to_host_vec(&st)?;
        let decoded = decode(&n.s.to_host_vec(&st)?, n.rows, n.k);
        let g = n.global.to_host_vec(&st)?[0].to_le_bytes();
        let mut digest = 0xcbf29ce484222325u64;
        for &v in payload.iter().chain(decoded.iter()) {
            digest = (digest ^ v as u64).wrapping_mul(0x100000001b3);
        }
        for _ in 0..n.rows {
            for v in g {
                digest = (digest ^ v as u64).wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(
            digest,
            if use_rowwise { rowwise } else { primary },
            "rowwise={use_rowwise} {rows}x{cols} stored={stored} k={k}"
        );
        println!(
            "exact NVFP4 transpose rowwise={use_rowwise} {rows}x{cols} stored={stored} k={k}: FP4, scales, global and complete padding"
        );
    }
    Ok(())
}
