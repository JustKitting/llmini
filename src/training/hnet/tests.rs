use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_MLP_ROUTE_TILE, GPT2_N_EMBD};
use rust_kernels_cuda::hnet_stage::{RatioObjective, StageShape};

use super::Trainer;
use crate::AppResult;

#[test]
#[ignore = "requires sm_120a PTX and the full native KDA backbone"]
fn native_routed_backbone_executes_and_reverses_real_blocks() -> AppResult {
    let mut trainer = Trainer::new_full_kda(0x47505432)?;
    let stream = trainer.runtime.stream.clone();
    let stage = trainer.runtime.hnet_stage_module()?;
    const BATCH: usize = 2;
    const OUTER: usize = 256;
    const WIDTH: usize = 512;
    let lengths = [192, 91];
    let mut queries = [0x3c00_u16, 0].repeat(BATCH * OUTER);
    for row in [41, 98, OUTER + 32] {
        queries[2 * row..2 * row + 2].copy_from_slice(&[0, 0x3c00]);
    }
    let keys = [0x3c00_u16, 0].repeat(BATCH * OUTER);
    let encoded: Vec<_> = (0..BATCH * OUTER * WIDTH)
        .map(|index| [0x3000_u16, 0xb000, 0x3400, 0x2800][(index + index / WIDTH) % 4])
        .collect();
    let q = DeviceBuffer::from_host(&stream, &queries)?;
    let k = DeviceBuffer::from_host(&stream, &keys)?;
    let valid = DeviceBuffer::from_host(&stream, &lengths)?;
    let encoded = DeviceBuffer::from_host(&stream, &encoded)?;
    let plan = stage.route(
        &stream,
        &q,
        &k,
        &valid,
        StageShape {
            batch: BATCH as u32,
            outer_stride: OUTER as u32,
            features: WIDTH as u32,
            router_width: 2,
            inner_alignment: GPT2_MLP_ROUTE_TILE as u32,
        },
        RatioObjective {
            target: 4.0,
            weight: 0.03,
        },
    )?;
    assert_eq!(plan.inner_lengths().to_host_vec(&stream)?, [3, 2]);
    assert_eq!(plan.inner_stride(), GPT2_MLP_ROUTE_TILE as u32);
    let gathered = stage.gather(&plan, &encoded)?.to_host_vec(&stream)?;
    let padding = DeviceBuffer::zeroed(&stream, GPT2_N_EMBD - WIDTH)?;
    let tape = trainer.forward_routed_backbone(&stage, &plan, &encoded, Some(&padding))?;
    let native = tape.output().to_host_vec(&stream)?;
    assert_ne!(native, gathered);
    for (sequence, valid) in [3, 2].into_iter().enumerate() {
        let row_base = sequence * plan.inner_stride() as usize;
        let real = &native[row_base * WIDTH..(row_base + valid) * WIDTH];
        assert!(real.iter().all(|&bits| bits & 0x7c00 != 0x7c00));
        let tail =
            &native[(row_base + valid) * WIDTH..(row_base + plan.inner_stride() as usize) * WIDTH];
        assert!(tail.iter().all(|&bits| bits == 0));
    }

    let output_gradient = (0..BATCH * OUTER * WIDTH)
        .map(|i| [0.0001_f32, -0.0002, 0.0003][i % 3])
        .collect::<Vec<_>>();
    let output_gradient = DeviceBuffer::from_host(&stream, &output_gradient)?;
    let expanded = stage.expand(&plan, tape.output())?;
    let reverse = stage.backward(&expanded, &output_gradient)?;
    drop(expanded);
    let inner_adjoint = tape.backward(&reverse.d_inner)?;
    let d_padding = inner_adjoint.d_padding.unwrap().to_host_vec(&stream)?;
    assert_eq!(d_padding.len(), GPT2_N_EMBD - WIDTH);
    assert!(d_padding.iter().all(|grad| grad.is_finite()));
    assert!(d_padding.iter().any(|grad| *grad != 0.0));
    let encoder_adjoint = stage
        .scatter(&plan, &inner_adjoint.d_features)?
        .to_host_vec(&stream)?;
    assert!(encoder_adjoint.iter().all(|value| value.is_finite()));
    assert!(
        encoder_adjoint
            .chunks_exact(WIDTH)
            .enumerate()
            .all(|(row, grad)| {
                let selected = [0, 41, 98, OUTER, OUTER + 32].contains(&row);
                selected == grad.iter().any(|value| *value != 0.0)
            })
    );
    assert_eq!(trainer.buffers.optimizer_state.step, 0);
    Ok(())
}
