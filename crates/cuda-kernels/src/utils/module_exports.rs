//! Discover optional graph entries without triggering a missing-symbol API error.
use cuda_core::{CudaModule, DriverError, IntoResult, sys};
use std::ffi::CStr;

pub(crate) fn contains(module: &CudaModule, kernel: &[u8]) -> Result<bool, DriverError> {
    module.context().bind_to_thread()?;
    let mut count = 0;
    // The module owns these handles/names throughout the query. Its context is
    // bound on this thread, and no raw pointer or handle escapes the borrow.
    unsafe {
        let handle = module.cu_module();
        sys::cuModuleGetFunctionCount(&mut count, handle).result()?;
        if count == 0 {
            return Ok(false);
        }
        let mut functions = vec![std::ptr::null_mut(); count as usize];
        sys::cuModuleEnumerateFunctions(functions.as_mut_ptr(), count, handle).result()?;
        for function in functions {
            let mut name = std::ptr::null();
            sys::cuFuncGetName(&mut name, function).result()?;
            if name.is_null() {
                return Err(DriverError(1));
            }
            if CStr::from_ptr(name).to_bytes() == kernel {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
