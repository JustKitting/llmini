#[path = "pack/chunk.rs"]
mod chunk;
#[path = "pack/hadamard.rs"]
mod hadamard;
#[path = "pack/payload.rs"]
mod payload;

pub(super) use self::chunk::{
    ms_eden_pack_chunk, ms_eden_pack_chunk_no_chunk_amax, ms_eden_pack_chunk_no_chunk_amax_row,
    pack_chunk,
};

macro_rules! guarded_pack_chunk {
    ($chunk:ident, $chunk_count:ident) => {
        let $chunk = $crate::nvfp4_quant::kernels::ms_eden::pack::pack_chunk();
        if $chunk >= $chunk_count {
            return;
        }
    };
}

pub(super) use guarded_pack_chunk;
