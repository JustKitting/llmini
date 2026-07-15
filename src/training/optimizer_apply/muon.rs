use cuda_core::{CudaStream, DriverError};

use crate::training::runtime::Runtime;

use super::super::OptimizerTrace;
use super::super::optimizer_muon::{MuonPointerTables, MuonTmaArgs, apply_muon_tma};
use super::super::optimizer_tc_scratch::MuonScratchBuffers;
use super::timed_ms;

pub(super) fn update_muon_groups(
    _stream: &CudaStream,
    runtime: &Runtime,
    tables: &MuonPointerTables,
    scratch: &mut MuonScratchBuffers,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    trace.muon_ms += timed_ms(|| {
        apply_muon_tma(MuonTmaArgs {
            runtime,
            table: &tables.all,
            scratch,
            slot_count: tables.slot_count,
            step,
            average_coefficient,
            grad_scale,
        })
    })?;
    Ok(())
}
