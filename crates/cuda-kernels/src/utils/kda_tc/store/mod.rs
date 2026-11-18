mod compact;
mod coords;

pub(crate) use compact::{
    store_compact_half_quad, store_compact_quad, store_compact_quads, store_hidden_output_quad,
    store_vnew_quads,
};
pub(crate) use coords::compact_fragment_coords;
