use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::{HostPtrs, MuonGroupTable};

pub(super) fn upload_table(
    stream: &CudaStream,
    rows: &[HostPtrs],
) -> Result<MuonGroupTable, DriverError> {
    let values: Vec<_> = rows.iter().copied().map(HostPtrs::descriptor).collect();
    Ok(MuonGroupTable {
        slots: DeviceBuffer::from_host(stream, &values)?,
        host_slots: values,
    })
}
