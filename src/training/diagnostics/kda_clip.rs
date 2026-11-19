use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{AttentionDims, uses_full_attention};

use crate::AppResult;
use crate::training::env::env_usize;

static LOGGER: OnceLock<Mutex<ClipLogger>> = OnceLock::new();

struct ClipLogger {
    interval: usize,
    heads: BufWriter<File>,
    layers: BufWriter<File>,
}

pub(in crate::training) fn initialize(run_dir: &Path) -> AppResult {
    let Some(interval) = env_usize("TRAIN_KDA_CLIP_LOG_INTERVAL").filter(|n| *n > 0) else {
        return Ok(());
    };
    let mut heads = BufWriter::new(File::create(run_dir.join("kda_clip_heads.csv"))?);
    let mut layers = BufWriter::new(File::create(run_dir.join("kda_clip_layers.csv"))?);
    writeln!(
        heads,
        "optimizer_step,layer,head,attention_kind,q_norm_max,k_norm_max,score,clip_factor,clipped,tau"
    )?;
    writeln!(
        layers,
        "optimizer_step,layer,attention_kind,heads,clipped_heads,clip_fraction,min_factor,mean_factor,max_score,max_q_norm,max_k_norm,non_finite_heads"
    )?;
    heads.flush()?;
    layers.flush()?;
    LOGGER
        .set(Mutex::new(ClipLogger {
            interval,
            heads,
            layers,
        }))
        .map_err(|_| "KDA clipping diagnostics already initialized")?;
    println!(
        "kda_clip_diagnostics interval={interval} step_basis=accepted_optimizer_updates run_dir={}",
        run_dir.display()
    );
    Ok(())
}

/// Read the factors already produced by the normal clipping kernels. This
/// diagnostic never changes device buffers or the optimizer's clipping rule.
pub(in crate::training) fn record(
    stream: &CudaStream,
    qk_norm_max: &DeviceBuffer<f32>,
    factors: &DeviceBuffer<f32>,
    optimizer_step: u32,
    active_depth: usize,
    tau: f32,
) -> AppResult {
    let Some(logger) = LOGGER.get() else {
        return Ok(());
    };
    let mut logger = logger
        .lock()
        .map_err(|_| "KDA clipping diagnostics lock poisoned")?;
    if optimizer_step != 1 && !(optimizer_step as usize).is_multiple_of(logger.interval) {
        return Ok(());
    }

    let norms = qk_norm_max.to_host_vec(stream)?;
    let factors = factors.to_host_vec(stream)?;
    for layer in 0..active_depth {
        let full_attention = uses_full_attention(layer);
        let dims = AttentionDims::new(full_attention);
        let count = dims.head_count as usize;
        // Match the offsets passed to prepare_kda_muon_clip_factor exactly.
        let norm_offset = layer * 2 * count;
        let factor_offset = layer * count;
        let kind = if full_attention { "full" } else { "kda" };
        let mut clipped_heads = 0usize;
        let mut non_finite_heads = 0usize;
        let mut min_factor = 1.0f32;
        let mut sum_factor = 0.0f64;
        let mut max_score = 0.0f32;
        let mut max_q = 0.0f32;
        let mut max_k = 0.0f32;
        for head in 0..count {
            let q = norms[norm_offset + head];
            let k = norms[norm_offset + count + head];
            let factor = factors[factor_offset + head];
            let score = q * k / (dims.head_dim as f32).sqrt();
            let clipped = factor < 1.0;
            clipped_heads += usize::from(clipped);
            non_finite_heads += usize::from(
                !q.is_finite() || !k.is_finite() || !score.is_finite() || !factor.is_finite(),
            );
            min_factor = min_factor.min(factor);
            sum_factor += factor as f64;
            max_score = max_score.max(score);
            max_q = max_q.max(q);
            max_k = max_k.max(k);
            writeln!(
                logger.heads,
                "{optimizer_step},{layer},{head},{kind},{q:.9e},{k:.9e},{score:.9e},{factor:.9e},{},{tau:.9e}",
                u8::from(clipped)
            )?;
        }
        writeln!(
            logger.layers,
            "{optimizer_step},{layer},{kind},{count},{clipped_heads},{:.9e},{min_factor:.9e},{:.9e},{max_score:.9e},{max_q:.9e},{max_k:.9e},{non_finite_heads}",
            clipped_heads as f64 / count as f64,
            sum_factor / count as f64
        )?;
    }
    logger.heads.flush()?;
    logger.layers.flush()?;
    Ok(())
}
