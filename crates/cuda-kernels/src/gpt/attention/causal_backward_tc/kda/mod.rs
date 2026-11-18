mod chunkwise;
mod elementwise;
mod finish;
mod intra;
mod state;
pub(crate) use finish::exact_kda_raw_finish_body;

pub(super) use chunkwise::{KdaChunkwiseGrads, KdaChunkwiseInputs, chunkwise_kda_backward_body};
pub(super) use elementwise::{
    KDA_FUSED_BACKWARD_BETA_ELEMS, KDA_FUSED_BACKWARD_TILE_ELEMS, add_kda_compact_body,
    chunk_cumsum_g_body, gather_kda_dout_body, make_kda_strict_neg_matrix_body,
    prepare_kda_backward_inputs_body, prepare_transform_kda_backward_cached_body,
};
pub(super) use finish::{FinishKdaGrads, finish_kda_backward_body};
pub(super) use intra::{
    KDA_INTRA_DG_TILE_ELEMS, KDA_INTRA_DKG_KG_TILE_ELEMS, KDA_INTRA_PAIR_STAT_ELEMS,
    KDA_INTRA_PAIR_VALUE_ELEMS, KDA_INTRA_THREADS_PER_BLOCK, KdaIntraGrads, KdaIntraInputs,
    chunk_intra_kda_backward_body, project_kda_intra_tiles, stage_backprojection,
    stage_chain_native, stage_chain_shared, stage_chain_vbeta, store_backprojection,
    store_chain_half,
};
pub(super) use state::{
    NATIVE_STATE_TILE_ELEMS, chunk_kda_dkg_from_vnew_dh_body, chunk_state_dw_dqg_matmul_body,
    chunk_state_project_du_dw_dqg_body, chunk_state_vnew_body,
};
