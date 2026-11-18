use cuda_core::{DeviceBuffer, LaunchConfig};
use rust_kernels_cuda::{
    mhc::kernels,
    mhc_window_post::{MhcWindowPostModule, PostWindowArgs},
};
use std::error::Error;

mod common;

#[test]
#[ignore = "compare all three native CUDA replay tapes to the separate post maps"]
fn mhc_replay_fusion_matches_each_post_map() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_context()?;
    let ordinary = kernels::from_module(module.clone())?;
    let fused = MhcWindowPostModule::from_module(module)?;
    for (rows, dim) in [(1usize, 1usize), (3, 257)] {
        let features = rows * dim;
        let wide = 4 * features;
        let values = |len: usize, salt: usize| {
            (0..len)
                .map(|i| {
                    let x = ((i * 13 + salt * 7) % 97) as i32 - 48;
                    x as f32 * 0.021
                })
                .collect::<Vec<_>>()
        };
        let input = DeviceBuffer::from_host(&stream, &values(wide, 0))?;
        let branches = (1..=3)
            .map(|salt| DeviceBuffer::from_host(&stream, &values(features, salt)))
            .collect::<Result<Vec<_>, _>>()?;
        let coefficients = (4..=6)
            .map(|salt| DeviceBuffer::from_host(&stream, &values(rows * 24, salt)))
            .collect::<Result<Vec<_>, _>>()?;
        let poison = 0x7fc01234;
        let initial = vec![f32::from_bits(poison); wide + 19];
        let mut reference = Vec::new();
        for step in 0..3 {
            let mut next = DeviceBuffer::from_host(&stream, &initial)?;
            ordinary.mhc_post_kernel(
                &stream,
                LaunchConfig {
                    grid_dim: (wide.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                reference.last().unwrap_or(&input),
                &branches[step],
                &coefficients[step],
                &mut next,
                rows as u32,
                dim as u32,
            )?;
            reference.push(next);
        }
        let mut rebuilt = [
            DeviceBuffer::from_host(&stream, &initial)?,
            DeviceBuffer::from_host(&stream, &initial)?,
            DeviceBuffer::from_host(&stream, &initial)?,
        ];
        let [first, second, third] = &mut rebuilt;
        fused.rebuild(PostWindowArgs {
            stream: &stream,
            input: &input,
            branches: [&branches[0], &branches[1], &branches[2]],
            coefficients: [&coefficients[0], &coefficients[1], &coefficients[2]],
            out: [first, second, third],
            rows: rows as u32,
            dim: dim as u32,
        })?;
        for (step, (got, expected)) in rebuilt.iter().zip(&reference).enumerate() {
            let got = got.to_host_vec(&stream)?;
            let expected = expected.to_host_vec(&stream)?;
            for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "rows={rows} dim={dim} step={step} index={i}"
                );
            }
            assert!(got[wide..].iter().all(|x| x.to_bits() == poison));
        }
    }
    Ok(())
}
