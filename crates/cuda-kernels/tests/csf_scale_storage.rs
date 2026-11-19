use cuda_core::DeviceBuffer;
mod common;
use common::encode_csf_scales as encode;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};
use std::error::Error;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn compressed_scales_preserve_nonzero_gemm_and_relu2() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let native = Sm120ScalePackModule::from_module(ptx.clone())?;
    let gemm = Nvfp4GemmModule::from_module(ptx)?;
    let mut scale_bytes = 0;
    let mut fallback_rows = 0;
    for (rows, k) in [(128usize, 128usize), (129, 256), (256, 2048)] {
        let groups = k / 16;
        let padded = rows.div_ceil(128) * 128;
        let logical: Vec<u8> = (0..rows * groups)
            .map(|i| {
                let row = i / groups;
                let col = i % groups;
                match row % 5 {
                    0 => (32 + col % 8) as u8,
                    1 => {
                        if col % 8 == 7 {
                            47
                        } else {
                            32
                        }
                    }
                    2 => {
                        if col % 8 == 7 {
                            48
                        } else {
                            32
                        }
                    }
                    3 => ((row + col * 37) % 256) as u8,
                    _ => 0,
                }
            })
            .collect();
        let packed = encode(&logical, rows, k);
        for row in 0..padded {
            for kg in 0..groups {
                if packed.len() / ((padded / 128) * (k / 128)) == 1024 {
                    fallback_rows += 1;
                }
                let actual = decode_one_six(&packed, padded, k, row, kg);
                let want = if row < rows {
                    logical[row * groups + kg]
                } else {
                    0
                };
                assert_eq!(actual, want, "codec rows={rows} k={k} row={row} kg={kg}");
                scale_bytes += 1;
            }
        }
    }
    println!("codec_exact_bytes={scale_bytes} fallback_rows={fallback_rows}");

    let mut values_checked = 0;
    for (m, n, k) in [(128usize, 128usize, 128usize), (256, 384, 2048)] {
        let make_scales = |rows: usize| -> Vec<u8> {
            (0..rows * k / 16)
                .map(|i| {
                    let row = i / (k / 16);
                    let col = i % (k / 16);
                    if row % 8 < 4 && col % 3 == 0 {
                        0
                    } else if row % 7 == 0 && col % 8 == 7 {
                        0x58
                    } else {
                        (0x30 + (row % 8) + (col % 8)) as u8
                    }
                })
                .collect()
        };
        let av: Vec<u8> = (0..m * k / 2)
            .map(|i| 0x11 + ((i % 3) as u8) * 0x11)
            .collect();
        let bv: Vec<u8> = (0..n * k / 2)
            .map(|i| [0x21, 0x32, 0x14, 0x95][i % 4])
            .collect();
        let a = DeviceBuffer::from_host(&stream, &av)?;
        let b = DeviceBuffer::from_host(&stream, &bv)?;
        let sa = DeviceBuffer::from_host(&stream, &make_scales(m))?;
        let sb = DeviceBuffer::from_host(&stream, &make_scales(n))?;
        let mut na = DeviceBuffer::<u8>::zeroed(&stream, m * k / 16)?;
        let mut nb = DeviceBuffer::<u8>::zeroed(&stream, n * k / 16)?;
        native.pack_pair(
            &stream, &sa, &mut na, m as u32, &sb, &mut nb, n as u32, k as u32,
        )?;
        let ca = DeviceBuffer::from_host(&stream, &encode(&make_scales(m), m, k))?;
        let cb = DeviceBuffer::from_host(&stream, &encode(&make_scales(n), n, k))?;
        let global = DeviceBuffer::from_host(&stream, &[0.01f32])?;
        let mut dn = TmaNvfp4DeviceScaleDescriptors::new(&stream)?;
        let mut dc = TmaNvfp4DeviceScaleDescriptors::new(&stream)?;
        println!(
            "gemm_buffers m={m} n={n} k={k} a={:#x} b={:#x} ca={:#x} cb={:#x}",
            a.cu_deviceptr(),
            b.cu_deviceptr(),
            ca.cu_deviceptr(),
            cb.cu_deviceptr()
        );
        gemm.prepare_tma_nvfp4_device_scales_into(
            &stream, &a, &na, &b, &nb, m as u32, k as u32, n as u32, &mut dn,
        )?;
        gemm.prepare_tma_nvfp4_csf_scales_into(
            &stream, &a, &ca, &b, &cb, m as u32, k as u32, n as u32, &mut dc,
        )?;
        let mut on = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
        let mut oc = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
        gemm.gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
            &stream, &dn, &mut on, m as u32, k as u32, n as u32, &global, &global,
        )?;
        stream.synchronize()?;
        println!("native_completed m={m} n={n} k={k}");
        let row_globals = DeviceBuffer::from_host(&stream, &vec![0.01f32; m])?;
        gemm.gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf(
            &stream,
            &dc,
            &mut oc,
            m as u32,
            k as u32,
            n as u32,
            &row_globals,
            &global,
            None,
        )?;
        let on = on.to_host_vec(&stream)?;
        let oc = oc.to_host_vec(&stream)?;
        assert!(
            on.iter().any(|v| v.abs() > 1e-8),
            "nonzero GEMM reference required"
        );
        for (i, (a, b)) in on.iter().zip(&oc).enumerate() {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "GEMM m={m} n={n} k={k} index={i} native={a} csf={b}"
            );
            values_checked += 1;
        }
        let canonical_host = rust_kernels_cuda::nvfp4::csf::encode_host(&make_scales(n), n, k);
        let canonical = DeviceBuffer::from_host(&stream, &canonical_host)?;
        let mut dh = TmaNvfp4DeviceScaleDescriptors::new(&stream)?;
        gemm.prepare_tma_nvfp4_csf_weight_into(
            &stream, &a, &ca, &b, &canonical, m as u32, k as u32, n as u32, &mut dh,
        )?;
        let mut hybrid = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
        gemm.gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
            &stream,
            &dh,
            &mut hybrid,
            m as u32,
            k as u32,
            n as u32,
            &global,
            &global,
        )?;
        let hybrid = hybrid.to_host_vec(&stream)?;
        for (i, (x, y)) in on.iter().zip(&hybrid).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "canonical forward m={m} n={n} k={k} index={i}"
            );
            values_checked += 1;
        }
        println!("canonical_forward_bit_exact m={m} n={n} k={k}");
        let bias_bytes = DeviceBuffer::from_host(&stream, &vec![0u8; n / 2])?;
        let bias_scales = rust_kernels_cuda::nvfp4::csf::encode_host(
            &vec![0x38; n / 16],
            1,
            n.next_power_of_two(),
        );
        let bias_scales = DeviceBuffer::from_host(&stream, &bias_scales)?;
        let bias_global = DeviceBuffer::from_host(&stream, &[1.0f32])?;
        let bias = rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor::new(
            &bias_bytes,
            &bias_scales,
            &bias_global,
        );
        let mut residual = DeviceBuffer::from_host(&stream, &vec![1.0f32; m * n])?;
        gemm.gemm_tma_nvfp4_rowwise_a_scale_residual(
            &stream,
            &dh,
            &mut residual,
            bias,
            m as u32,
            k as u32,
            n as u32,
            &row_globals,
            &global,
            None,
        )?;
        for (i, (x, y)) in on.iter().zip(residual.to_host_vec(&stream)?).enumerate() {
            assert_eq!(
                (x + 1.0).to_bits(),
                y.to_bits(),
                "residual m={m} n={n} k={k} index={i}"
            );
            values_checked += 1;
        }
        let padded_n = n + 128;
        let pb: Vec<u8> = (0..padded_n * k / 2)
            .map(|i| [0x21, 0x32, 0x14, 0x95][i % 4])
            .collect();
        let pb = DeviceBuffer::from_host(&stream, &pb)?;
        let ps = rust_kernels_cuda::nvfp4::csf::encode_host(&make_scales(padded_n), padded_n, k);
        let ps = DeviceBuffer::from_host(&stream, &ps)?;
        let mut dp = TmaNvfp4DeviceScaleDescriptors::new(&stream)?;
        gemm.prepare_tma_nvfp4_csf_weight_into(
            &stream, &a, &ca, &pb, &ps, m as u32, k as u32, n as u32, &mut dp,
        )?;
        let mut po = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
        gemm.gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
            &stream, &dp, &mut po, m as u32, k as u32, n as u32, &global, &global,
        )?;
        let po = po.to_host_vec(&stream)?;
        for (i, (x, y)) in on.iter().zip(&po).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "padded canonical B active={n} stored={padded_n} k={k} index={i}"
            );
            values_checked += 1;
        }
        println!("padded_canonical_B_bit_exact active={n} stored={padded_n} k={k}");
        let pre: Vec<u16> = (0..m * n)
            .map(|i| if i % 3 == 0 { 0xbc00 } else { 0x3c00 })
            .collect();
        let pre = DeviceBuffer::from_host(&stream, &pre)?;
        let mut fused = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
        let mut amax = DeviceBuffer::<f32>::zeroed(
            &stream,
            rust_kernels_cuda::nvfp4_tma_matmul::kernels::tma_nvfp4_output_amax_chunks(
                m as u32, n as u32,
            ) as usize,
        )?;
        gemm.gemm_tma_nvfp4_rowwise_a_scale_relu2_backward_f16_with_output_amax(
            &stream,
            &dc,
            &pre,
            &mut fused,
            &mut amax,
            m as u32,
            k as u32,
            n as u32,
            &row_globals,
            &global,
            None,
        )?;
        let fv = fused.to_host_vec(&stream)?;
        for (i, v) in fv.iter().enumerate() {
            let want = on[i] * 2.0 * if i % 3 == 0 { 0.0 } else { 1.0 };
            assert_eq!(v.to_bits(), want.to_bits(), "fused ReLU2 at {i}");
            values_checked += 1;
        }
        assert!(amax.to_host_vec(&stream)?.iter().all(|x| x.is_finite()));
        println!("gemm_bit_exact m={m} n={n} k={k}");
    }
    println!("PASS scale_bytes={scale_bytes} gemm_values={values_checked}");
    Ok(())
}

#[test]
fn compact_vectors_preserve_all_scale_bytes() {
    for len in [16usize, 128, 256, 2048, 6144] {
        let source: Vec<u8> = (0..len / 16).map(|i| (i * 37 % 256) as u8).collect();
        let k = len.max(128).next_power_of_two();
        let encoded = rust_kernels_cuda::nvfp4::csf::encode_host(&source, 1, k);
        assert_eq!(
            encoded.len(),
            rust_kernels_cuda::nvfp4::csf::encoded_len(1, k)
        );
        for (i, &wanted) in source.iter().enumerate() {
            assert_eq!(rust_kernels_cuda::nvfp4::csf::scale_at(&encoded, i), wanted);
        }
    }
}

#[test]
fn mixed_zero_scale_bytes_stay_exact() {
    for values in [
        [0, 96, 101, 0, 112, 96, 101, 0],
        [0, 1, 63, 0, 1, 63, 0, 0],
        [0, 127, 127, 0, 127, 0, 127, 0],
        [0, 1, 127, 0, 1, 127, 0, 1],
    ] {
        let vector = rust_kernels_cuda::nvfp4::csf::encode_host(&values, 1, 128);
        for (i, value) in values.into_iter().enumerate() {
            assert_eq!(rust_kernels_cuda::nvfp4::csf::scale_at(&vector, i), value);
        }
    }
}

#[test]
fn compact_vectors_preserve_every_high_bit_mask() {
    for mask in 0u16..256 {
        let values: [u8; 8] = std::array::from_fn(|i| {
            ((i * 7 + mask as usize) % 64) as u8 | ((((mask >> i) & 1) as u8) << 6)
        });
        let vector = rust_kernels_cuda::nvfp4::csf::encode_host(&values, 1, 128);
        for (i, value) in values.into_iter().enumerate() {
            assert_eq!(
                rust_kernels_cuda::nvfp4::csf::scale_at(&vector, i),
                value,
                "high-bit mask {mask}"
            );
        }
    }
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn sparse_k_routes_preserve_cpu_reference_across_64_tiles() -> Result<(), Box<dyn Error>> {
    let (_, st, ptx) = common::cuda_test_context()?;
    let gemm = Nvfp4GemmModule::from_module(ptx)?;
    let (m, n, k) = (128usize, 128usize, 8192usize);
    let mut payload = vec![0u8; m * k / 2];
    for row in 0..m {
        for tile in 0..64 {
            payload[row * k / 2 + tile * 64] = 0x02;
        }
    }
    let a = DeviceBuffer::from_host(&st, &payload)?;
    let b = DeviceBuffer::from_host(&st, &payload)?;
    let source: Vec<u8> = (0..m * k / 16)
        .map(|i| 0x20 + ((i % (k / 16)) / 8) as u8)
        .collect();
    let a_scales = DeviceBuffer::from_host(
        &st,
        &rust_kernels_cuda::nvfp4::csf::encode_rowwise_host(&source, m, k),
    )?;
    let b_scales = DeviceBuffer::from_host(
        &st,
        &rust_kernels_cuda::nvfp4::csf::encode_rowwise_host(&vec![0x38; n * k / 16], n, k),
    )?;
    let ga = DeviceBuffer::from_host(&st, &vec![1.0f32; m])?;
    let gb = DeviceBuffer::from_host(&st, &[1.0f32])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&st, m * n)?;
    let mut desc = TmaNvfp4DeviceScaleDescriptors::new(&st)?;
    gemm.prepare_tma_nvfp4_csf_scales_into(
        &st, &a, &a_scales, &b, &b_scales, m as u32, k as u32, n as u32, &mut desc,
    )?;
    for active in [1u32, 2, 4, 5, 7, 16, 31, 32, 33, 47, 48, 63, 64] {
        let mut mask = 0u64;
        for i in 0..active {
            mask |= 1u64 << ((i * 37 + 11) % 64);
        }
        let mut masks = vec![0u64; 65];
        masks[0] = mask;
        for bit in 0..64 {
            masks[1 + bit] = (mask >> bit) & 1;
        }
        let masks = DeviceBuffer::from_host(&st, &masks)?;
        let route =
            rust_kernels_cuda::nvfp4_tma_matmul::launcher::Nvfp4GemmRoute::token_m_feature_k(
                &masks, 1, 64, active,
            );
        gemm.gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer_csf(
            &st,
            &desc,
            &mut out,
            m as u32,
            k as u32,
            n as u32,
            &ga,
            &gb,
            Some(route),
        )?;
        let expected = (0..64)
            .filter(|&bit| mask & (1u64 << bit) != 0)
            .map(|bit| {
                let raw = 0x20u32 + bit;
                f32::from_bits((((raw >> 3) + 120) << 23) | ((raw & 7) << 20))
            })
            .sum::<f32>();
        for value in out.to_host_vec(&st)? {
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "active={active} mask={mask:#x}"
            );
        }
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
