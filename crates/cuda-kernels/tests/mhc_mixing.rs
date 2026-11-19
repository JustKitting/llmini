//! Predeclared operation batch: fuse per-feature mHC mixing adjoints and batch
//! the exact existing warp/block reduction trees. Preserve every output bit,
//! coefficient orientation, pre-gradient accumulation/overwrite mode, all
//! actual B8/S2048/d2048 rows, tail features/rows, guards and shape rejection.
use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::mhc::kernels as legacy;
use rust_kernels_cuda::mhc_mixing::{MhcMixingModule, PostBackwardArgs, PreBackwardArgs};
use std::error::Error;
mod common;
const GUARD: usize = 257;
fn x(i: usize) -> f32 {
    ((i * 17 % 101) as i32 - 50) as f32 * 0.013
}
fn g(i: usize) -> f32 {
    ((i * 13 % 31) as i32 - 15) as f32 * 0.007
}
fn branch(i: usize) -> f32 {
    ((i * 7 % 89) as i32 - 44) as f32 * 0.017
}
fn coeff(i: usize) -> f32 {
    ((i * 11 % 23) as i32 - 11) as f32 * 0.043
}
fn equal_bits(
    stream: &cuda_core::CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    label: &str,
) -> Result<usize, Box<dyn Error>> {
    let a = a.to_host_vec(stream)?;
    let b = b.to_host_vec(stream)?;
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} i={i} old={a:e} new={b:e}"
        );
    }
    Ok(a.len())
}
#[test]
#[ignore = "native complete B8/S2048/d2048 mHC mixing adjoints with exact reduction trees"]
fn native_mhc_mixing_preserves_all_adjoint_bits_and_reduction_order() -> Result<(), Box<dyn Error>>
{
    let (_, stream, ptx) = common::cuda_test_context()?;
    let old = legacy::from_module(ptx.clone())?;
    let new = MhcMixingModule::from_module(ptx)?;
    let mut compared = 0usize;
    let mut independent_max = 0.0f64;
    for (n, d) in [
        (16384usize, 2048usize),
        (1, 1),
        (1, 31),
        (7, 255),
        (8, 256),
        (9, 257),
        (31, 2048),
        (2, 2048),
    ] {
        let nd = n * d;
        let wide = nd * 4;
        let xd = DeviceBuffer::from_host(&stream, &(0..wide).map(x).collect::<Vec<_>>())?;
        let gd = DeviceBuffer::from_host(&stream, &(0..wide).map(g).collect::<Vec<_>>())?;
        let fd = DeviceBuffer::from_host(&stream, &(0..nd).map(branch).collect::<Vec<_>>())?;
        let cd = DeviceBuffer::from_host(&stream, &(0..n * 24).map(coeff).collect::<Vec<_>>())?;
        let mut old_df = DeviceBuffer::from_host(&stream, &vec![-0.25f32; nd + GUARD])?;
        let mut new_df = DeviceBuffer::from_host(&stream, &vec![-0.25f32; nd + GUARD])?;
        let mut old_dx = DeviceBuffer::from_host(&stream, &vec![0.125f32; wide + GUARD])?;
        let mut new_dx = DeviceBuffer::from_host(&stream, &vec![0.125f32; wide + GUARD])?;
        let mut old_dc = DeviceBuffer::from_host(&stream, &vec![0.375f32; n * 24 + GUARD])?;
        let mut new_dc = DeviceBuffer::from_host(&stream, &vec![0.375f32; n * 24 + GUARD])?;
        let config = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        old.mhc_post_backward_kernel(
            &stream,
            config,
            &xd,
            &fd,
            &cd,
            &gd,
            &mut old_df,
            &mut old_dx,
            &mut old_dc,
            n as u32,
            d as u32,
        )?;
        new.post_backward(PostBackwardArgs {
            stream: &stream,
            input: &xd,
            branch: &fd,
            coefficients: &cd,
            gradient: &gd,
            df: &mut new_df,
            dx: &mut new_dx,
            dc: &mut new_dc,
            rows: n as u32,
            dim: d as u32,
        })?;
        compared += equal_bits(&stream, &old_df, &new_df, "post df")?;
        compared += equal_bits(&stream, &old_dx, &new_dx, "post dx")?;
        compared += equal_bits(&stream, &old_dc, &new_dc, "post dc")?;
        let df_host = new_df.to_host_vec(&stream)?;
        let dx_host = new_dx.to_host_vec(&stream)?;
        let dc_host = new_dc.to_host_vec(&stream)?;
        for r in [0, n / 2, n - 1] {
            for j in [0, d / 2, d - 1] {
                let terms = (0..4)
                    .map(|k| f64::from(coeff(r * 24 + 4 + k)) * f64::from(g((r * 4 + k) * d + j)));
                let expected = terms.clone().sum::<f64>();
                let bound = 4e-7 * terms.map(f64::abs).sum::<f64>() + 1e-10;
                let error = (f64::from(df_host[r * d + j]) - expected).abs();
                independent_max = independent_max.max(error);
                assert!(error <= bound);
                for h in 0..4 {
                    let terms = (0..4).map(|k| {
                        f64::from(coeff(r * 24 + 8 + h * 4 + k)) * f64::from(g((r * 4 + k) * d + j))
                    });
                    let expected = terms.clone().sum::<f64>();
                    let bound = 4e-7 * terms.map(f64::abs).sum::<f64>() + 1e-10;
                    let error = (f64::from(dx_host[(r * 4 + h) * d + j]) - expected).abs();
                    independent_max = independent_max.max(error);
                    assert!(error <= bound);
                }
            }
        }
        for r in [0, n - 1] {
            for k in 0..4 {
                for h in 0..5 {
                    let terms = (0..d).map(|j| {
                        f64::from(g((r * 4 + k) * d + j))
                            * f64::from(if h == 4 {
                                branch(r * d + j)
                            } else {
                                x((r * 4 + h) * d + j)
                            })
                    });
                    let expected = terms.clone().sum::<f64>();
                    let bound = 1e-5 * terms.map(f64::abs).sum::<f64>() + 1e-9;
                    let at = if h == 4 {
                        r * 24 + 4 + k
                    } else {
                        r * 24 + 8 + h * 4 + k
                    };
                    let error = (f64::from(dc_host[at]) - expected).abs();
                    independent_max = independent_max.max(error);
                    assert!(error <= bound);
                }
            }
        }
        assert!(df_host[nd..].iter().all(|&x| x == -0.25));
        assert!(dx_host[wide..].iter().all(|&x| x == 0.125));
        assert!(dc_host[n * 24..].iter().all(|&x| x == 0.375));
        drop(df_host);
        drop(dx_host);
        drop(dc_host);
        for overwrite in [false, true] {
            old.mhc_pre_backward_kernel(
                &stream,
                config,
                &xd,
                &cd,
                &fd,
                &mut old_dx,
                &mut old_dc,
                n as u32,
                d as u32,
                overwrite,
            )?;
            new.pre_backward(PreBackwardArgs {
                stream: &stream,
                input: &xd,
                coefficients: &cd,
                gradient: &fd,
                dx: &mut new_dx,
                dc: &mut new_dc,
                rows: n as u32,
                dim: d as u32,
                overwrite,
            })?;
            compared += equal_bits(&stream, &old_dx, &new_dx, "pre dx")?;
            compared += equal_bits(&stream, &old_dc, &new_dc, "pre dc")?;
        }
    }
    let one = DeviceBuffer::from_host(&stream, &[1.0f32])?;
    let mut a = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    let mut b = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    let mut c = DeviceBuffer::from_host(&stream, &[0.375f32; GUARD])?;
    for (rows, dim) in [
        (0, 2048),
        (8, 0),
        (u32::MAX, 2048),
        (8, u32::MAX),
        (16384, 2048),
    ] {
        assert!(
            new.post_backward(PostBackwardArgs {
                stream: &stream,
                input: &one,
                branch: &one,
                coefficients: &one,
                gradient: &one,
                df: &mut a,
                dx: &mut b,
                dc: &mut c,
                rows,
                dim
            })
            .is_err()
        );
        assert!(
            new.pre_backward(PreBackwardArgs {
                stream: &stream,
                input: &one,
                coefficients: &one,
                gradient: &one,
                dx: &mut b,
                dc: &mut c,
                rows,
                dim,
                overwrite: false
            })
            .is_err()
        );
    }
    for data in [&a, &b, &c] {
        assert!(data.to_host_vec(&stream)?.iter().all(|&x| x == 0.375));
    }
    println!(
        "native_mhc_mixing_verified max_rows=16384 dim=2048 shapes=8 pre_modes=2 all_output_bits_exact=true original_XOR_reduction_order=true compared_values={compared} independent_max_error={independent_max:e}"
    );
    Ok(())
}
