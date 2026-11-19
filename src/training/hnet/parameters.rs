use std::{fs::File, io::BufWriter, path::Path};

use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{GPT2_N_EMBD, Gpt2Rng, KDA_CHUNK_SIZE};
use rust_kernels_cuda::{
    f16_tc_matmul::F16ConvertArgs,
    hnet_byte_head::{BYTE_BOS, BYTE_LOGIT_STRIDE},
    hnet_optimizer::{CompactAdamModule, CompactAdamState, CompactAdamStep},
    nvfp4::{Nvfp4DecodeTransposeArgs, csf},
    optimizer::ScheduleFreeMaterializeArgs,
};

use super::{
    Trainer,
    hierarchy::{HierarchyAdjoints, HierarchyWeights},
    outer::{OuterParameterAdjoints, OuterWeights},
    projection::StageWeights,
};
use crate::{
    AppResult,
    training::{
        optimizer::OptimizerScratch,
        optimizer_apply::{WeightUpdateResult, adam_debug_config},
        runtime::Runtime,
    },
    upload::UploadedNvfp4,
};

pub(super) const OUTER_WIDTH: usize = 512;

struct Parameter {
    name: String,
    tensor: UploadedNvfp4,
    state: CompactAdamState,
    gradient: DeviceBuffer<f32>,
    decay: bool,
    lr_multiplier: f32,
}

struct OuterBlock {
    qkv: Parameter,
    output: Parameter,
    norm_scale: Parameter,
    norm_bias: Parameter,
}

pub(super) struct Parameters {
    optimizer: CompactAdamModule,
    encoder: Vec<OuterBlock>,
    decoder: Vec<OuterBlock>,
    encoder_norm: Parameter,
    backbone_norm: Parameter,
    decoder_norm: Parameter,
    embedding: Parameter,
    readout: Parameter,
    query: Parameter,
    key: Parameter,
    residual: Parameter,
    residual_bias: Parameter,
    padding: Parameter,
    represented_padding: DeviceBuffer<u16>,
    scheduled: DeviceBuffer<f32>,
}

pub(super) struct WeightViews<'a> {
    encoder: Vec<OuterWeights<'a>>,
    decoder: Vec<OuterWeights<'a>>,
    encoder_norm: &'a UploadedNvfp4,
    backbone_norm: &'a UploadedNvfp4,
    decoder_norm: &'a UploadedNvfp4,
    stage: StageWeights<'a>,
    padding: &'a DeviceBuffer<u16>,
    readout: &'a UploadedNvfp4,
    pub embedding: &'a UploadedNvfp4,
}

impl WeightViews<'_> {
    pub fn hierarchy(&self) -> HierarchyWeights<'_> {
        HierarchyWeights {
            encoder: &self.encoder,
            decoder: &self.decoder,
            encoder_norm: self.encoder_norm,
            backbone_norm: self.backbone_norm,
            decoder_norm: self.decoder_norm,
            stage: self.stage,
            width_padding: Some(self.padding),
            byte_readout: self.readout,
        }
    }
}

impl Parameters {
    pub fn save_weights(&self, stream: &CudaStream, path: &Path) -> AppResult {
        use crate::checkpoint::{CheckpointWriter, write_nvfp4_tensor};
        let parameters = self
            .encoder
            .iter()
            .chain(&self.decoder)
            .flat_map(|block| {
                [
                    &block.qkv,
                    &block.output,
                    &block.norm_scale,
                    &block.norm_bias,
                ]
            })
            .chain([
                &self.encoder_norm,
                &self.backbone_norm,
                &self.decoder_norm,
                &self.embedding,
                &self.readout,
                &self.query,
                &self.key,
                &self.residual,
                &self.residual_bias,
                &self.padding,
            ]);
        let mut writer = CheckpointWriter::new(BufWriter::new(File::create_new(path)?));
        writer.write_header(u32::try_from(parameters.clone().count())?)?;
        for parameter in parameters {
            write_nvfp4_tensor(&mut writer, stream, &parameter.name, &parameter.tensor)?;
        }
        writer.finish()
    }

    pub fn new(trainer: &mut Trainer, layers: usize, ratio: f32, seed: u64) -> AppResult<Self> {
        if layers == 0 || !ratio.is_finite() || ratio <= 1.0 {
            return Err(
                "the H-Net encoder and decoder need layers and a compression ratio above one"
                    .into(),
            );
        }
        let runtime = &trainer.runtime;
        let stream = &runtime.stream;
        let optimizer = runtime.hnet_optimizer_module()?;
        let width = OUTER_WIDTH;
        let added = GPT2_N_EMBD
            .checked_sub(width)
            .filter(|&added| added > 0 && added.is_multiple_of(128))
            .ok_or("the native H-Net backbone needs a wider aligned residual")?;
        let packed_qkv = (4 * width + width / KDA_CHUNK_SIZE).next_multiple_of(128);
        let outer_lr = (ratio * GPT2_N_EMBD as f32 / width as f32).sqrt();
        let mut rng = Gpt2Rng::new(seed);
        let mut initialize = |name: String,
                              rows: usize,
                              cols: usize,
                              values: &[f32],
                              decay: bool,
                              lr_multiplier: f32| {
            let len = rows * cols;
            let scales = csf::encode_host(&vec![0; len / 16], rows, cols);
            let mut tensor =
                UploadedNvfp4::from_host(stream, &vec![0; len / 2], &scales, 0.0, len)?;
            let initial = DeviceBuffer::from_host(stream, values)?;
            pack(
                runtime,
                &mut trainer.buffers.optimizer,
                &mut tensor,
                &initial,
            )?;
            let state = optimizer.from_nvfp4(stream, tensor.device(), len as u32)?;
            stream.synchronize()?;
            Ok::<_, Box<dyn std::error::Error>>(Parameter {
                name,
                tensor,
                state,
                gradient: DeviceBuffer::zeroed(stream, len)?,
                decay,
                lr_multiplier,
            })
        };
        let mut build_stack = |prefix: &str| -> AppResult<Vec<OuterBlock>> {
            let mut blocks = Vec::with_capacity(layers);
            let residual_std = 0.02 / ((2 * layers) as f32).sqrt();
            let active_qkv = (4 * width + width / KDA_CHUNK_SIZE) * width;
            for layer in 0..layers {
                let name = format!("hnet.{prefix}.{layer}");
                let mut qkv = normal(&mut rng, active_qkv, 0.02);
                qkv.resize(packed_qkv * width, 0.0);
                blocks.push(OuterBlock {
                    qkv: initialize(
                        format!("{name}.qkv"),
                        packed_qkv,
                        width,
                        &qkv,
                        true,
                        outer_lr,
                    )?,
                    output: initialize(
                        format!("{name}.output"),
                        width,
                        width,
                        &normal(&mut rng, width * width, residual_std),
                        true,
                        outer_lr,
                    )?,
                    norm_scale: initialize(
                        format!("{name}.norm_scale"),
                        1,
                        width,
                        &vec![1.0; width],
                        false,
                        outer_lr,
                    )?,
                    norm_bias: initialize(
                        format!("{name}.norm_bias"),
                        1,
                        width,
                        &vec![0.0; width],
                        false,
                        outer_lr,
                    )?,
                });
            }
            Ok(blocks)
        };
        let encoder = build_stack("encoder")?;
        let decoder = build_stack("decoder")?;
        let encoder_norm = initialize(
            "hnet.encoder.rms_scale".into(),
            1,
            width,
            &vec![1.0; width],
            false,
            outer_lr,
        )?;
        let backbone_norm = initialize(
            "hnet.backbone.rms_scale".into(),
            1,
            GPT2_N_EMBD,
            &vec![1.0; GPT2_N_EMBD],
            false,
            1.0,
        )?;
        let decoder_norm = initialize(
            "hnet.decoder.rms_scale".into(),
            1,
            width,
            &vec![1.0; width],
            false,
            outer_lr,
        )?;
        let vocabulary = (BYTE_BOS + 1) as usize;
        let classes = BYTE_LOGIT_STRIDE as usize;
        let mut output_weights = normal(&mut rng, vocabulary.saturating_sub(1) * width, 0.02);
        output_weights.resize(classes * width, 0.0);
        let mut identity = vec![0.0; width * width];
        for i in 0..width {
            identity[i * width + i] = 1.0;
        }
        let embedding = initialize(
            "hnet.embedding".into(),
            vocabulary,
            width,
            &normal(&mut rng, vocabulary * width, 0.02),
            false,
            outer_lr,
        )?;
        let readout = initialize(
            "hnet.byte_readout".into(),
            classes,
            width,
            &output_weights,
            true,
            outer_lr,
        )?;
        let query = initialize(
            "hnet.router.query".into(),
            width,
            width,
            &identity,
            true,
            outer_lr,
        )?;
        let key = initialize(
            "hnet.router.key".into(),
            width,
            width,
            &identity,
            true,
            outer_lr,
        )?;
        let residual = initialize(
            "hnet.skip".into(),
            width,
            width,
            &vec![0.0; width * width],
            true,
            outer_lr,
        )?;
        let bias_bound = (width as f32).sqrt().recip();
        let residual_bias: Vec<_> = (0..width)
            .map(|_| {
                let uniform = rng.next_u32() as f32 / 4294967296.0;
                (2.0 * uniform - 1.0) * bias_bound
            })
            .collect();
        let residual_bias = initialize(
            "hnet.skip.bias".into(),
            1,
            width,
            &residual_bias,
            false,
            outer_lr,
        )?;
        let padding = initialize(
            "hnet.width_padding".into(),
            1,
            added,
            &vec![0.0; added],
            false,
            1.0,
        )?;
        let parameters = Self {
            optimizer,
            encoder,
            decoder,
            encoder_norm,
            backbone_norm,
            decoder_norm,
            embedding,
            readout,
            query,
            key,
            residual,
            residual_bias,
            padding,
            represented_padding: DeviceBuffer::zeroed(stream, added)?,
            scheduled: DeviceBuffer::zeroed(stream, packed_qkv * width)?,
        };
        let mut gradients = Vec::new();
        for block in parameters.encoder.iter().chain(&parameters.decoder) {
            for parameter in [
                &block.qkv,
                &block.output,
                &block.norm_scale,
                &block.norm_bias,
            ] {
                gradients.push((parameter.name.as_str(), &parameter.gradient));
            }
        }
        for parameter in [
            &parameters.encoder_norm,
            &parameters.backbone_norm,
            &parameters.decoder_norm,
            &parameters.embedding,
            &parameters.readout,
            &parameters.query,
            &parameters.key,
            &parameters.residual,
            &parameters.residual_bias,
            &parameters.padding,
        ] {
            gradients.push((parameter.name.as_str(), &parameter.gradient));
        }
        trainer.prepare_hnet_optimizer(&gradients)?;
        Ok(parameters)
    }

    pub fn weights(&self) -> WeightViews<'_> {
        WeightViews {
            encoder: self.encoder.iter().map(OuterBlock::weights).collect(),
            decoder: self.decoder.iter().map(OuterBlock::weights).collect(),
            encoder_norm: &self.encoder_norm.tensor,
            backbone_norm: &self.backbone_norm.tensor,
            decoder_norm: &self.decoder_norm.tensor,
            stage: StageWeights {
                query: &self.query.tensor,
                key: &self.key.tensor,
                residual: &self.residual.tensor,
                residual_bias: &self.residual_bias.tensor,
            },
            padding: &self.represented_padding,
            readout: &self.readout.tensor,
            embedding: &self.embedding.tensor,
        }
    }

    pub fn materialize(
        &mut self,
        runtime: &Runtime,
        scratch: &mut OptimizerScratch,
        beta: f32,
    ) -> AppResult {
        let stream = &runtime.stream;
        for parameter in self
            .encoder
            .iter_mut()
            .chain(&mut self.decoder)
            .flat_map(OuterBlock::parameters_mut)
            .chain([
                &mut self.encoder_norm,
                &mut self.backbone_norm,
                &mut self.decoder_norm,
                &mut self.embedding,
                &mut self.readout,
                &mut self.query,
                &mut self.key,
                &mut self.residual,
                &mut self.residual_bias,
                &mut self.padding,
            ])
        {
            self.optimizer
                .write_view(stream, &parameter.state, beta, &mut self.scheduled)?;
            pack(runtime, scratch, &mut parameter.tensor, &self.scheduled)?;
        }
        runtime
            .decode
            .decode_transpose_f32(Nvfp4DecodeTransposeArgs {
                stream,
                input: self.padding.tensor.device(),
                output: &mut self.scheduled,
                rows: 1,
                cols: self.padding.tensor.scale_k,
            })?;
        runtime.f16_tc_matmul.fp32_to_f16(F16ConvertArgs {
            stream,
            src: &self.scheduled,
            dst: &mut self.represented_padding,
            element_count: self.padding.tensor.scale_k,
        })?;
        Ok(())
    }

    pub fn bind_gradients(
        &mut self,
        stream: &CudaStream,
        adjoints: &HierarchyAdjoints,
        lookup: &DeviceBuffer<f32>,
    ) -> AppResult {
        for (stack, gradients) in [
            (&mut self.encoder, &adjoints.encoder),
            (&mut self.decoder, &adjoints.decoder),
        ] {
            if stack.len() != gradients.len() {
                return Err("outer adjoints do not match the H-Net stack".into());
            }
            for (block, gradient) in stack.iter_mut().zip(gradients) {
                block.bind_gradients(stream, gradient)?;
            }
        }
        let padding = adjoints
            .width_padding
            .as_ref()
            .ok_or("the learned width has no adjoint")?;
        for (parameter, gradient) in [
            (&mut self.encoder_norm, &adjoints.encoder_norm),
            (&mut self.backbone_norm, &adjoints.backbone_norm),
            (&mut self.decoder_norm, &adjoints.decoder_norm),
            (&mut self.embedding, lookup),
            (&mut self.readout, &adjoints.byte_readout),
            (&mut self.query, &adjoints.stage.query),
            (&mut self.key, &adjoints.stage.key),
            (&mut self.residual, &adjoints.stage.residual),
            (&mut self.residual_bias, &adjoints.stage.residual_bias),
            (&mut self.padding, padding),
        ] {
            parameter.bind_gradient(stream, gradient)?;
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        runtime: &Runtime,
        step: u32,
        result: &WeightUpdateResult,
        rng: &mut Gpt2Rng,
    ) -> AppResult {
        if result.trace.update_skipped {
            return Ok(());
        }
        let adam = adam_debug_config(step);
        for parameter in self
            .encoder
            .iter_mut()
            .chain(&mut self.decoder)
            .flat_map(OuterBlock::parameters_mut)
            .chain([
                &mut self.encoder_norm,
                &mut self.backbone_norm,
                &mut self.decoder_norm,
                &mut self.embedding,
                &mut self.readout,
                &mut self.query,
                &mut self.key,
                &mut self.residual,
                &mut self.residual_bias,
                &mut self.padding,
            ])
        {
            self.optimizer.update(
                &runtime.stream,
                &mut parameter.state,
                &parameter.gradient,
                CompactAdamStep {
                    learning_rate: adam.learning_rate * parameter.lr_multiplier,
                    weight_decay: if parameter.decay {
                        adam.weight_decay
                    } else {
                        0.0
                    },
                    beta1: adam.beta1,
                    beta2: adam.beta2,
                    beta1_correction: adam.beta1_correction,
                    beta2_correction: adam.beta2_correction,
                    eps: adam.eps,
                    average: result.average_coefficient,
                    grad_scale: result.grad_scale,
                    seed: rng.next_u32(),
                },
            )?;
        }
        Ok(())
    }
}

impl OuterBlock {
    fn weights(&self) -> OuterWeights<'_> {
        OuterWeights {
            qkv: &self.qkv.tensor,
            output: &self.output.tensor,
            norm_scale: &self.norm_scale.tensor,
            norm_bias: &self.norm_bias.tensor,
        }
    }

    fn parameters_mut(&mut self) -> [&mut Parameter; 4] {
        [
            &mut self.qkv,
            &mut self.output,
            &mut self.norm_scale,
            &mut self.norm_bias,
        ]
    }

    fn bind_gradients(
        &mut self,
        stream: &CudaStream,
        gradient: &OuterParameterAdjoints,
    ) -> AppResult {
        for (parameter, gradient) in self.parameters_mut().into_iter().zip([
            &gradient.qkv,
            &gradient.output,
            &gradient.norm_scale,
            &gradient.norm_bias,
        ]) {
            parameter.bind_gradient(stream, gradient)?;
        }
        Ok(())
    }
}

impl Parameter {
    fn bind_gradient(&mut self, stream: &CudaStream, gradient: &DeviceBuffer<f32>) -> AppResult {
        if self.gradient.len() != gradient.len() {
            return Err(format!("gradient shape changed for {}", self.name).into());
        }
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                self.gradient.cu_deviceptr(),
                gradient.cu_deviceptr(),
                gradient.len() * size_of::<f32>(),
                stream.cu_stream(),
            )?;
        }
        Ok(())
    }
}

fn pack(
    runtime: &Runtime,
    scratch: &mut OptimizerScratch,
    tensor: &mut UploadedNvfp4,
    source: &DeviceBuffer<f32>,
) -> AppResult {
    runtime
        .optimizer
        .materialize_schedule_free(ScheduleFreeMaterializeArgs {
            stream: &runtime.stream,
            bytes: &mut tensor.bytes,
            scales: &mut tensor.scales,
            global_scale: &mut tensor.global_scale,
            z_master: source,
            x_master: source,
            amax: &mut scratch.amax,
            chunk_amax: &mut scratch.chunk_amax,
            len: tensor.len as u32,
            scale_rows: tensor.scale_rows,
            scale_k: tensor.scale_k,
            beta: 0.0,
            symexp_lin_beta: 0.0,
            symexp_lin_scales: None,
        })?;
    Ok(())
}

fn normal(rng: &mut Gpt2Rng, len: usize, stddev: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let u = (rng.next_u32() as f32 + 1.0) / 4294967296.0;
        let v = rng.next_u32() as f32 / 4294967296.0;
        let radius = (-2.0 * u.ln()).sqrt() * stddev;
        let (sine, cosine) = (std::f32::consts::TAU * v).sin_cos();
        values.push(radius * cosine);
        if values.len() < len {
            values.push(radius * sine);
        }
    }
    values
}
