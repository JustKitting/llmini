use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::{HostPtrs, MuonGroupTable};

pub(super) fn upload_table(
    stream: &CudaStream,
    rows: &[HostPtrs],
) -> Result<MuonGroupTable, DriverError> {
    let values: Vec<_> = rows.iter().copied().map(HostPtrs::descriptor).collect();
    let symexp_lin_values: Vec<_> = rows
        .iter()
        .copied()
        .map(HostPtrs::symexp_lin_descriptor)
        .collect();
    Ok(MuonGroupTable {
        slots: DeviceBuffer::from_host(stream, &values)?,
        host_slots: values,
        symexp_lin_slots: DeviceBuffer::from_host(stream, &symexp_lin_values)?,
        host_symexp_lin_slots: symexp_lin_values,
    })
}
