use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use cuda_core::DeviceBuffer;
use gpt2_nvfp4::GPT2_N_EMBD;
use rust_kernels_cuda::{
    hnet_byte_head::{BYTE_BOS, BYTE_LOGIT_STRIDE},
    nvfp4::csf,
};

use super::{
    outer::OuterWeights,
    parameters::{OUTER_WIDTH, WeightViews, added_width, packed_qkv_width, represent_padding},
    projection::StageWeights,
};
use crate::{
    AppResult, checkpoint::CheckpointReader, training::runtime::Runtime, upload::UploadedNvfp4,
};

pub(super) const BACKBONE_FILE: &str = "hnet-backbone.ckpt";
pub(super) const OUTER_FILE: &str = "hnet-outer.ckpt";

struct Block {
    qkv: UploadedNvfp4,
    output: UploadedNvfp4,
    norm_scale: UploadedNvfp4,
    norm_bias: UploadedNvfp4,
}

impl Block {
    fn view(&self) -> OuterWeights<'_> {
        OuterWeights {
            qkv: &self.qkv,
            output: &self.output,
            norm_scale: &self.norm_scale,
            norm_bias: &self.norm_bias,
        }
    }
}

pub(super) struct Snapshot {
    encoder: Vec<Block>,
    decoder: Vec<Block>,
    encoder_norm: UploadedNvfp4,
    backbone_norm: UploadedNvfp4,
    decoder_norm: UploadedNvfp4,
    embedding: UploadedNvfp4,
    readout: UploadedNvfp4,
    query: UploadedNvfp4,
    key: UploadedNvfp4,
    residual: UploadedNvfp4,
    residual_bias: UploadedNvfp4,
    represented_padding: DeviceBuffer<u16>,
}

impl Snapshot {
    pub fn load(runtime: &Runtime, path: &Path, layers: usize) -> AppResult<Self> {
        Self::read(runtime, File::open(path)?, layers)
    }

    pub(super) fn read(runtime: &Runtime, input: impl Read, layers: usize) -> AppResult<Self> {
        let mut reader = CheckpointReader::new(BufReader::new(input));
        let count = reader.read_header()? as usize;
        if count != layers * 8 + 10 {
            return Err(format!(
                "H-Net outer checkpoint has {count} tensors for {layers} layers per stack"
            )
            .into());
        }
        let stream = &runtime.stream;
        let width = OUTER_WIDTH;
        let added = added_width()?;
        let mut take = |name: &str, rows: usize, cols: usize| -> AppResult<UploadedNvfp4> {
            let tensor = reader.read_tensor()?;
            let len = rows * cols;
            if tensor.name != name
                || tensor.len != len
                || tensor.bytes.len() != len / 2
                || tensor.scales.len() != csf::encoded_len(rows, cols)
                || tensor.scales.get(..4) != Some(&(cols as u32).to_le_bytes())
                || tensor.scales.get(4..8) != Some(&(rows as u32).to_le_bytes())
                || !tensor.global_scale.is_finite()
                || tensor.global_scale < 0.0
            {
                return Err(format!(
                    "invalid H-Net checkpoint tensor {name} ({rows} by {cols}), found {}",
                    tensor.name
                )
                .into());
            }
            UploadedNvfp4::from_host(
                stream,
                &tensor.bytes,
                &tensor.scales,
                tensor.global_scale,
                len,
            )
        };
        let mut stack = |name| -> AppResult<Vec<Block>> {
            (0..layers)
                .map(|index| {
                    let prefix = format!("hnet.{name}.{index}");
                    Ok(Block {
                        qkv: take(&format!("{prefix}.qkv"), packed_qkv_width(), width)?,
                        output: take(&format!("{prefix}.output"), width, width)?,
                        norm_scale: take(&format!("{prefix}.norm_scale"), 1, width)?,
                        norm_bias: take(&format!("{prefix}.norm_bias"), 1, width)?,
                    })
                })
                .collect()
        };
        let encoder = stack("encoder")?;
        let decoder = stack("decoder")?;
        let encoder_norm = take("hnet.encoder.rms_scale", 1, width)?;
        let backbone_norm = take("hnet.backbone.rms_scale", 1, GPT2_N_EMBD)?;
        let decoder_norm = take("hnet.decoder.rms_scale", 1, width)?;
        let embedding = take("hnet.embedding", (BYTE_BOS + 1) as usize, width)?;
        let readout = take("hnet.byte_readout", BYTE_LOGIT_STRIDE as usize, width)?;
        let query = take("hnet.router.query", width, width)?;
        let key = take("hnet.router.key", width, width)?;
        let residual = take("hnet.skip", width, width)?;
        let residual_bias = take("hnet.skip.bias", 1, width)?;
        let padding = take("hnet.width_padding", 1, added)?;
        let mut represented_padding = DeviceBuffer::zeroed(stream, added)?;
        let mut scratch = DeviceBuffer::zeroed(stream, added)?;
        represent_padding(runtime, &padding, &mut scratch, &mut represented_padding)?;
        stream.synchronize()?;
        Ok(Self {
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
            represented_padding,
        })
    }

    pub fn weights(&self) -> WeightViews<'_> {
        WeightViews {
            encoder: self.encoder.iter().map(Block::view).collect(),
            decoder: self.decoder.iter().map(Block::view).collect(),
            encoder_norm: &self.encoder_norm,
            backbone_norm: &self.backbone_norm,
            decoder_norm: &self.decoder_norm,
            stage: StageWeights {
                query: &self.query,
                key: &self.key,
                residual: &self.residual,
                residual_bias: &self.residual_bias,
            },
            padding: &self.represented_padding,
            readout: &self.readout,
            embedding: &self.embedding,
        }
    }
}
