use std::io::{self, Read};

use rust_kernels_cuda::hnet_byte_head::{BYTE_BOS, BYTE_EOS, BYTE_PADDING_TARGET};

pub(super) const MAGIC: &[u8] = b"NVFP4_REFERENCE_BYTES_V1\n";

/// Stream framed raw bytes without materializing a full document or corpus.
/// Each batch row belongs to one document, including rows cut at the context limit.
pub(super) struct ByteDocuments<R> {
    input: R,
    records: u32,
    current: Option<Document>,
    scratch: Vec<u8>,
    exhausted: bool,
}

struct Document {
    bytes: u32,
    eos: bool,
    previous: u32,
}

pub(super) struct ByteBatch {
    pub inputs: Vec<u32>,
    pub targets: Vec<u32>,
    pub lengths: Vec<u32>,
    pub stride: u32,
    pub scored_targets: u32,
    pub raw_bytes: u32,
}

impl<R: Read> ByteDocuments<R> {
    pub fn new(mut input: R) -> io::Result<Self> {
        let mut magic = [0; MAGIC.len()];
        input.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "incorrect byte document header",
            ));
        }
        let mut count = [0; 4];
        input.read_exact(&mut count)?;
        Ok(Self {
            input,
            records: u32::from_le_bytes(count),
            current: None,
            scratch: Vec::new(),
            exhausted: false,
        })
    }

    pub fn next_batch(&mut self, batch: u32, stride: u32) -> io::Result<Option<ByteBatch>> {
        let rows = batch
            .checked_mul(stride)
            .filter(|&rows| rows > 0)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "incorrect byte batch shape")
            })? as usize;
        let mut result = ByteBatch {
            inputs: vec![BYTE_BOS; rows],
            targets: vec![BYTE_PADDING_TARGET; rows],
            lengths: vec![0; batch as usize],
            stride,
            scored_targets: 0,
            raw_bytes: 0,
        };
        let stride = stride as usize;
        for (sequence, length) in result.lengths.iter_mut().enumerate() {
            if !self.advance_document()? {
                break;
            }
            let doc = self
                .current
                .as_mut()
                .expect("advanced to a nonempty document window");
            let first = sequence * stride;
            let byte_count = (doc.bytes as usize).min(stride);
            self.scratch.resize(byte_count, 0);
            self.input.read_exact(&mut self.scratch)?;
            for (position, &byte) in self.scratch.iter().enumerate() {
                result.inputs[first + position] = doc.previous;
                result.targets[first + position] = u32::from(byte);
                doc.previous = u32::from(byte);
            }
            doc.bytes -= byte_count as u32;
            result.raw_bytes += byte_count as u32;
            let include_eos = doc.bytes == 0 && doc.eos && byte_count < stride;
            if include_eos {
                result.inputs[first + byte_count] = doc.previous;
                result.targets[first + byte_count] = BYTE_EOS;
                doc.eos = false;
            }
            *length = byte_count as u32 + u32::from(include_eos);
            result.scored_targets += *length;
            if doc.bytes == 0 && !doc.eos {
                self.current = None;
            }
        }
        Ok((result.scored_targets > 0).then_some(result))
    }

    fn advance_document(&mut self) -> io::Result<bool> {
        if self.current.is_some() {
            return Ok(true);
        }
        while self.records > 0 {
            let mut count = [0; 4];
            let mut closed = [0];
            self.input.read_exact(&mut count)?;
            self.input.read_exact(&mut closed)?;
            self.records -= 1;
            if closed[0] > 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "incorrect byte document ending flag",
                ));
            }
            let bytes = u32::from_le_bytes(count);
            let eos = closed[0] == 1;
            if bytes > 0 || eos {
                self.current = Some(Document {
                    bytes,
                    eos,
                    previous: BYTE_BOS,
                });
                return Ok(true);
            }
        }
        if !self.exhausted {
            let mut trailing = [0];
            if self.input.read(&mut trailing)? != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bytes after final framed document",
                ));
            }
            self.exhausted = true;
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hnet_byte_batches_score_every_byte_and_keep_document_boundaries() -> io::Result<()> {
        let mut source = MAGIC.to_vec();
        source.extend(3_u32.to_le_bytes());
        for (data, eos) in [
            (&[0xc3, 0xa9, 0][..], true),
            (&[][..], true),
            (&[0xff, 0x0a][..], false),
        ] {
            source.extend((data.len() as u32).to_le_bytes());
            source.push(u8::from(eos));
            source.extend(data);
        }
        let mut docs = ByteDocuments::new(source.as_slice())?;
        let first = docs.next_batch(2, 3)?.unwrap();
        assert_eq!(first.inputs, [BYTE_BOS, 0xc3, 0xa9, 0, BYTE_BOS, BYTE_BOS]);
        assert_eq!(
            first.targets,
            [
                0xc3,
                0xa9,
                0,
                BYTE_EOS,
                BYTE_PADDING_TARGET,
                BYTE_PADDING_TARGET
            ]
        );
        assert_eq!(first.lengths, [3, 1]);
        assert_eq!((first.raw_bytes, first.scored_targets), (3, 4));
        let second = docs.next_batch(2, 3)?.unwrap();
        assert_eq!(
            second.inputs,
            [BYTE_BOS, BYTE_BOS, BYTE_BOS, BYTE_BOS, 0xff, BYTE_BOS]
        );
        assert_eq!(
            second.targets,
            [
                BYTE_EOS,
                BYTE_PADDING_TARGET,
                BYTE_PADDING_TARGET,
                0xff,
                0x0a,
                BYTE_PADDING_TARGET
            ]
        );
        assert_eq!(second.lengths, [1, 2]);
        assert_eq!((second.raw_bytes, second.scored_targets), (2, 3));
        assert!(docs.next_batch(2, 3)?.is_none());
        Ok(())
    }
}
