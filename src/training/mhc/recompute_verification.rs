//! Test-only host fingerprints of actual CUDA operands, no host model replay.
//! Each transfer is hashed then released, rather than retaining33 full tapes.
use super::{
    BOUNDARIES, Mhc, PAD, WIDTH,
    state::{ProjectionTape, Tape},
};
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use sha2::{Digest, Sha256};

pub(in crate::training) struct RecomputeVerification {
    expected: Vec<Option<[u8; 32]>>,
    pub reconstructed: Vec<bool>,
}
impl RecomputeVerification {
    pub fn reset(&mut self) {
        self.expected.fill(None);
        self.reconstructed.fill(false);
    }
    pub fn observe(
        &mut self,
        stream: &CudaStream,
        index: usize,
        reconstructed: bool,
        tape: &Tape,
        projection: &ProjectionTape,
        n: usize,
    ) -> Result<(), DriverError> {
        let mut hash = Sha256::new();
        fn floats(
            hash: &mut Sha256,
            stream: &CudaStream,
            b: &DeviceBuffer<f32>,
            n: usize,
        ) -> Result<(), DriverError> {
            let values = b.to_host_vec(stream)?;
            assert!(n <= values.len());
            // f32 has no padding; compare the actual represented device bits.
            let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), n * 4) };
            hash.update(bytes);
            Ok(())
        }
        floats(&mut hash, stream, &tape.input, n * WIDTH)?;
        let q = projection.input_q.rowwise();
        hash.update(&q.bytes.to_host_vec(stream)?[..n * WIDTH / 2]);
        let scale_bytes = rust_kernels_cuda::nvfp4::csf::rowwise_encoded_len(n, WIDTH);
        hash.update(&q.scales.to_host_vec(stream)?[..scale_bytes]);
        floats(&mut hash, stream, q.global_scales, n)?;
        floats(&mut hash, stream, &projection.inverse, n)?;
        floats(&mut hash, stream, &projection.raw, n * PAD)?;
        floats(&mut hash, stream, &projection.coefficients, n * 24)?;
        let digest: [u8; 32] = hash.finalize().into();
        if reconstructed {
            assert_eq!(
                Some(digest),
                self.expected[index],
                "mHC recomputed input/quantized/inverse/raw/coefficient tape differs at boundary{index}"
            );
            self.reconstructed[index] = true;
        } else {
            self.expected[index] = Some(digest);
        }
        Ok(())
    }
}
impl Mhc {
    pub fn verify_recomputation(&mut self) {
        self.recompute_verification = Some(RecomputeVerification {
            expected: vec![None; BOUNDARIES + 1],
            reconstructed: vec![false; BOUNDARIES + 1],
        });
    }
}
