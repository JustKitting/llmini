use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::MaybeUninit;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory};

use super::cute::{TmaSwizzle, U4SmemLayout};

const TMA_DESCRIPTOR_WORDS: usize = 16;
const TMA_DESCRIPTOR_COUNT: usize = 4;
const TMA_DESCRIPTOR_CACHE_CHUNK_ENTRIES: usize = 64;
type TmaDescriptorWords = [u64; TMA_DESCRIPTOR_WORDS];
type PackedTmaDescriptors = [TmaDescriptorWords; TMA_DESCRIPTOR_COUNT];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct TmaNvfp4DeviceScaleDescriptorKey {
    addresses: [u64; TMA_DESCRIPTOR_COUNT],
    shape: [u32; 3],
}

impl TmaNvfp4DeviceScaleDescriptorKey {
    pub(super) const fn new(
        addresses: [u64; TMA_DESCRIPTOR_COUNT],
        token_count: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Self {
        Self {
            addresses,
            shape: [token_count, input_dim, output_dim],
        }
    }
}

struct CachedTmaDescriptors {
    device_base: u64,
    // The async H2D source must remain valid until the copy completes. Keeping
    // the immutable host descriptor for the cache lifetime makes that explicit.
    _host: Box<PackedTmaDescriptors>,
}

pub struct TmaNvfp4DeviceScaleDescriptors {
    indices: HashMap<TmaNvfp4DeviceScaleDescriptorKey, usize>,
    entries: Vec<CachedTmaDescriptors>,
    chunks: Vec<DeviceBuffer<PackedTmaDescriptors>>,
    active: Option<usize>,
}

impl TmaNvfp4DeviceScaleDescriptors {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            indices: HashMap::new(),
            entries: Vec::new(),
            chunks: vec![allocate_descriptor_chunk(stream)?],
            active: None,
        })
    }

    pub(super) fn activate(&mut self, key: TmaNvfp4DeviceScaleDescriptorKey) -> bool {
        let Some(&index) = self.indices.get(&key) else {
            return false;
        };
        self.active = Some(index);
        true
    }

    pub(super) fn insert(
        &mut self,
        stream: &CudaStream,
        key: TmaNvfp4DeviceScaleDescriptorKey,
        descriptors: PackedTmaDescriptors,
    ) -> Result<(), DriverError> {
        debug_assert!(!self.indices.contains_key(&key));

        let index = self.entries.len();
        let chunk_index = index / TMA_DESCRIPTOR_CACHE_CHUNK_ENTRIES;
        let chunk_slot = index % TMA_DESCRIPTOR_CACHE_CHUNK_ENTRIES;
        if chunk_index == self.chunks.len() {
            self.chunks.push(allocate_descriptor_chunk(stream)?);
        }

        let device_base = self.chunks[chunk_index].cu_deviceptr()
            + (chunk_slot * std::mem::size_of::<PackedTmaDescriptors>()) as u64;
        let host = Box::new(descriptors);
        unsafe {
            memory::memcpy_htod_async(
                device_base,
                host.as_ref() as *const PackedTmaDescriptors,
                std::mem::size_of::<PackedTmaDescriptors>(),
                stream.cu_stream(),
            )?;
        }

        self.entries.push(CachedTmaDescriptors {
            device_base,
            _host: host,
        });
        self.indices.insert(key, index);
        self.active = Some(index);
        Ok(())
    }

    #[inline]
    pub(super) fn a_deviceptr(&self) -> u64 {
        self.descriptor_deviceptr(0)
    }

    #[inline]
    pub(super) fn b_deviceptr(&self) -> u64 {
        self.descriptor_deviceptr(1)
    }

    #[inline]
    pub(super) fn a_scales_deviceptr(&self) -> u64 {
        self.descriptor_deviceptr(2)
    }

    #[inline]
    pub(super) fn b_scales_deviceptr(&self) -> u64 {
        self.descriptor_deviceptr(3)
    }

    #[inline]
    fn descriptor_deviceptr(&self, descriptor: usize) -> u64 {
        let active = self
            .active
            .expect("TMA descriptors must be prepared before launch");
        self.entries[active].device_base
            + (descriptor * std::mem::size_of::<TmaDescriptorWords>()) as u64
    }
}

fn allocate_descriptor_chunk(
    stream: &CudaStream,
) -> Result<DeviceBuffer<PackedTmaDescriptors>, DriverError> {
    let bytes = TMA_DESCRIPTOR_CACHE_CHUNK_ENTRIES * std::mem::size_of::<PackedTmaDescriptors>();
    let ptr = unsafe { memory::malloc_sync(bytes)? };
    Ok(unsafe {
        DeviceBuffer::from_raw_parts(
            ptr,
            TMA_DESCRIPTOR_CACHE_CHUNK_ENTRIES,
            stream.context().clone(),
        )
    })
}

pub fn encode_u4_tiled_layout<L: U4SmemLayout>(
    global_address: *mut c_void,
    global_width_u4: u64,
    global_height: u64,
    row_stride_bytes: u64,
    tile_height: u32,
) -> Result<[u64; 16], DriverError> {
    use cuda_core::sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;

    encode_tiled(
        global_address,
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        global_width_u4,
        global_height,
        row_stride_bytes,
        L::PACKS_PER_ROW * 8,
        tile_height,
        L::TMA_SWIZZLE,
    )
}

pub fn encode_u16_tiled(
    global_address: *mut c_void,
    global_width_u16: u64,
    global_height: u64,
    row_stride_bytes: u64,
    tile_width_u16: u32,
    tile_height: u32,
) -> Result<[u64; 16], DriverError> {
    use cuda_core::sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16;

    encode_tiled(
        global_address,
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16,
        global_width_u16,
        global_height,
        row_stride_bytes,
        tile_width_u16,
        tile_height,
        TmaSwizzle::None,
    )
}

fn encode_tiled(
    global_address: *mut c_void,
    data_type: cuda_core::sys::CUtensorMapDataType_enum,
    global_width: u64,
    global_height: u64,
    row_stride_bytes: u64,
    tile_width: u32,
    tile_height: u32,
    swizzle: TmaSwizzle,
) -> Result<[u64; 16], DriverError> {
    use cuda_core::sys::{
        CUtensorMap, CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let global_dim = [global_width, global_height];
    let global_strides = [row_stride_bytes];
    let box_dim = [tile_width, tile_height];
    let element_strides = [1, 1];
    let cu_swizzle = match swizzle {
        TmaSwizzle::None => CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
        TmaSwizzle::Swizzle64B => CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
        TmaSwizzle::Swizzle128B => CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
    };

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            data_type,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu_swizzle,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(DriverError(result));
    }

    let tensor_map = unsafe { tensor_map.assume_init() };
    Ok(unsafe { std::mem::transmute_copy::<CUtensorMap, [u64; 16]>(&tensor_map) })
}
