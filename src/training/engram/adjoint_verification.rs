//! Test-only observation of real CUDA adjoints; no CPU projection or training.
use crate::AppResult;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};

pub(in crate::training) struct MemoryAdjointVerification {
    projectors: Vec<usize>,
    sum: Vec<f32>,
}

impl MemoryAdjointVerification {
    pub fn new() -> Self {
        Self {
            projectors: Vec::new(),
            sum: Vec::new(),
        }
    }
    pub fn observe(
        &mut self,
        stream: &CudaStream,
        projector: usize,
        contribution: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<(), DriverError> {
        let values = contribution.to_host_vec(stream)?;
        assert!(values[..len].iter().all(|x| x.is_finite()));
        if self.sum.is_empty() {
            self.sum.extend_from_slice(&values[..len]);
        } else {
            assert_eq!(self.sum.len(), len);
            for (sum, &value) in self.sum.iter_mut().zip(&values[..len]) {
                *sum += value;
            }
        }
        self.projectors.push(projector);
        Ok(())
    }
    pub fn verify(
        &self,
        stream: &CudaStream,
        accumulated: &DeviceBuffer<f32>,
        table: &DeviceBuffer<f32>,
        ids: &DeviceBuffer<u32>,
        n: usize,
        config: &super::Config,
    ) -> AppResult {
        assert_eq!(
            self.projectors,
            config.projector_indices(),
            "all independent K adjoints and exactly one shared V adjoint must contribute"
        );
        let actual = accumulated.to_host_vec(stream)?;
        assert!(actual.len() >= self.sum.len());
        for (i, (&a, &b)) in actual.iter().zip(&self.sum).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "native dense memory accumulation at {i}"
            );
        }
        let ids = ids.to_host_vec(stream)?;
        let table = table.to_host_vec(stream)?;
        let heads = config.primes.len();
        let head_dim = config.head_dim;
        assert_eq!(self.sum.len(), n * heads * head_dim);
        let mut expected = vec![0f64; table.len()];
        let mut absolute = vec![0f32; table.len()];
        let mut count = vec![0usize; table.len() / head_dim];
        for i in 0..n * heads {
            let row = ids[i] as usize;
            assert!(row < count.len());
            count[row] += 1;
            for j in 0..head_dim {
                let target = row * head_dim + j;
                let value = self.sum[i * head_dim + j];
                expected[target] += f64::from(value);
                absolute[target] += value.abs();
            }
        }
        let mut maximum_error = 0f64;
        for (i, &actual) in table.iter().enumerate() {
            assert!(actual.is_finite());
            if absolute[i] == 0. {
                assert_eq!(
                    actual, 0.,
                    "untouched/padded table gradient must be exactly zero at {i}"
                );
                continue;
            }
            // A gamma_k summation bound covers CUDA's unordered FP32 atomic
            // additions. Its inputs are the observed native dense adjoints.
            let ku = count[i / head_dim] as f64 * f64::from(f32::EPSILON);
            let bound = 1.01 * ku / (1. - ku) * f64::from(absolute[i]) + 1e-12;
            let error = (f64::from(actual) - expected[i]).abs();
            assert!(
                error <= bound,
                "shared table scatter at {i}: actual={actual} expected={} error={error} bound={bound}",
                expected[i]
            );
            maximum_error = maximum_error.max(error);
        }
        println!(
            "four_stream_shared_native_adjoint_verified projectors={:?} dense_sum=bit_exact memory_elements={} table_elements={} collided_rows={} maximum_collision_count={} maximum_scatter_absolute_error={maximum_error} updates=0",
            self.projectors,
            self.sum.len(),
            table.len(),
            count.iter().filter(|&&x| x > 1).count(),
            count.iter().max().copied().unwrap_or(0)
        );
        Ok(())
    }
}
