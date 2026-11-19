use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::loss::{CrossEntropyArgs, CrossEntropyParams, LossModule};

mod common;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cross_entropy_loss_only_is_bit_exact_at_model_vocabulary() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(LossModule::from_module)?;
    let vocab_size = 32000;
    let mut logits = vec![f32::NEG_INFINITY; 4 * vocab_size];
    logits[1024] = 0.0;
    logits[2047] = 100.0;
    logits[vocab_size + 24] = 0.0;
    logits[2 * vocab_size - 1] = 0.0;
    logits[2 * vocab_size + 31] = -200.0;
    logits[3 * vocab_size - 1] = 200.0;
    for (index, value) in logits[3 * vocab_size..].iter_mut().enumerate() {
        *value = (index % 31) as f32 * 0.125 - 2.0;
    }
    let logits = DeviceBuffer::from_host(&stream, &logits)?;
    let targets = DeviceBuffer::from_host(&stream, &[2047_u32, 24, 31, 145])?;
    let mut full_loss = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
    let mut read_only_loss = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
    let mut gradients = DeviceBuffer::<f32>::zeroed(&stream, 4 * vocab_size)?;
    let mut amax = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
    let params = CrossEntropyParams {
        token_count: 4,
        vocab_size: vocab_size as u32,
    };
    module.cross_entropy(CrossEntropyArgs {
        stream: &stream,
        logits: &logits,
        targets: &targets,
        losses: &mut full_loss,
        dlogits: &mut gradients,
        dlogits_row_amax: &mut amax,
        token_count: params.token_count,
        vocab_size: params.vocab_size,
    })?;
    assert!(module.try_cross_entropy_loss(
        &stream,
        &logits,
        &targets,
        &mut read_only_loss,
        params,
    )?);
    let full_loss = full_loss.to_host_vec(&stream)?;
    let read_only_loss = read_only_loss.to_host_vec(&stream)?;
    assert!(full_loss[0] > 0.0 && full_loss.iter().all(|loss| loss.is_finite()));
    for (&actual, &expected) in read_only_loss.iter().zip(&full_loss) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    Ok(())
}

const TOKEN_COUNT: usize = 2;
const VOCAB_SIZE: usize = 4;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cross_entropy_writes_losses_and_dlogits() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(LossModule::from_module)?;

    let logits = [
        0.0_f32,
        f32::NEG_INFINITY,
        0.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        0.0,
        f32::NEG_INFINITY,
        0.0,
    ];
    let targets = [0_u32, 3];
    let logits_dev = DeviceBuffer::from_host(&stream, &logits)?;
    let targets_dev = DeviceBuffer::from_host(&stream, &targets)?;
    let mut losses_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT)?;
    let mut dlogits_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * VOCAB_SIZE)?;
    let mut row_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT)?;

    module.cross_entropy(CrossEntropyArgs {
        stream: &stream,
        logits: &logits_dev,
        targets: &targets_dev,
        losses: &mut losses_dev,
        dlogits: &mut dlogits_dev,
        dlogits_row_amax: &mut row_amax_dev,
        token_count: TOKEN_COUNT as u32,
        vocab_size: VOCAB_SIZE as u32,
    })?;

    let losses = losses_dev.to_host_vec(&stream)?;
    let dlogits = dlogits_dev.to_host_vec(&stream)?;
    let row_amax = row_amax_dev.to_host_vec(&stream)?;
    let expected = expected_loss_and_grad(&logits, &targets);

    common::assert_slice_close(&losses, &expected.0, TOLERANCE);
    common::assert_slice_close(&dlogits, &expected.1, TOLERANCE);

    for (row, actual_amax) in row_amax.iter().enumerate() {
        let base = row * VOCAB_SIZE;
        let expected_amax = expected.1[base..base + VOCAB_SIZE]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);
        common::assert_close(*actual_amax, expected_amax, TOLERANCE);
    }

    Ok(())
}

fn expected_loss_and_grad(logits: &[f32], targets: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut losses = vec![0.0_f32; TOKEN_COUNT];
    let mut grad = vec![0.0_f32; TOKEN_COUNT * VOCAB_SIZE];

    for row in 0..TOKEN_COUNT {
        let row_base = row * VOCAB_SIZE;
        let row_logits = &logits[row_base..row_base + VOCAB_SIZE];
        let row_max = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = row_logits
            .iter()
            .map(|value| (*value - row_max).exp())
            .sum::<f32>();
        let target = targets[row] as usize;
        losses[row] = denom.ln() + row_max - row_logits[target];

        for col in 0..VOCAB_SIZE {
            let probability = (row_logits[col] - row_max).exp() / denom;
            grad[row_base + col] =
                (probability - if col == target { 1.0 } else { 0.0 }) / TOKEN_COUNT as f32;
        }
    }

    (losses, grad)
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cross_entropy_preserves_small_target_gradients_and_masked_rows() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(LossModule::from_module)?;
    let cases = [
        ([9.0,-9.0,-9.0],0_u32),
        ([37.0,0.0,f32::NEG_INFINITY],0),
        ([100.0,0.0,f32::NEG_INFINITY],0),
        ([0.0,0.0,0.0],1),
        ([5.0,f32::NEG_INFINITY,f32::NEG_INFINITY],0),
        ([f32::MAX,-f32::MAX,f32::NEG_INFINITY],0),
        ([0.0,37.0,f32::NEG_INFINITY],0),
    ];
    for (values,target) in cases {
        let logits=DeviceBuffer::from_host(&stream,&values)?;
        let targets=DeviceBuffer::from_host(&stream,&[target])?;
        let mut losses=DeviceBuffer::<f32>::zeroed(&stream,1)?;
        let mut gradient=DeviceBuffer::<f32>::zeroed(&stream,3)?;
        let mut amax=DeviceBuffer::<f32>::zeroed(&stream,1)?;
        module.cross_entropy(CrossEntropyArgs {stream:&stream,logits:&logits,targets:&targets,
            losses:&mut losses,dlogits:&mut gradient,dlogits_row_amax:&mut amax,token_count:1,vocab_size:3})?;
        let loss=losses.to_host_vec(&stream)?[0];
        let g=gradient.to_host_vec(&stream)?;
        let amax=amax.to_host_vec(&stream)?[0];
        assert!(loss.is_finite() && g.iter().all(|v|v.is_finite()),"{values:?} {loss} {g:?}");
        let sum=g.iter().map(|&v|v as f64).sum::<f64>().abs();
        let norm=g.iter().map(|v|(*v as f64).abs()).sum::<f64>();
        assert!(sum<=norm*2e-7,"gradient sum {sum:e} for {values:?}: {g:?}");
        assert_eq!(amax,g.iter().map(|v|v.abs()).fold(0.0,f32::max));
        if values[0]==9.0 || values[0]==37.0 || values[0]==100.0 {
            assert!(loss>0.0 && g[0]<0.0,"tail collapsed for {values:?}: {loss:e} {g:?}");
            let tail=(values[1] as f64-values[0] as f64).exp();
            let count=if values[2].is_finite(){2.0}else{1.0};
            let exact_loss=(count*tail).ln_1p();
            let tolerance=if values[0]==100.0 {0.05}else{3e-6};
            assert!((loss as f64/exact_loss-1.0).abs()<tolerance,"loss tail {loss:e} vs {exact_loss:e}");
        }
        println!("cuda_ce logits={values:?} target={target} loss={loss:e} gradient={g:?} amax={amax:e}");
    }
    Ok(())
}
