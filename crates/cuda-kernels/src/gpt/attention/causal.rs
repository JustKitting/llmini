use cuda_core::DeviceCopy;

pub(crate) const CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK: u32 = 128;
pub(crate) const CAUSAL_MAX_WARPS_PER_BLOCK: u32 = CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK / 32;
const CAUSAL_CHUNK_SIZE: u32 = 64;
const CAUSAL_DECAY_SCALE: f32 = 0.01;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CausalAttentionParams {
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub attention_window: u32,
    pub scale: f32,
    pub chunk_size: u32,
    pub decay_scale: f32,
}

unsafe impl DeviceCopy for CausalAttentionParams {}

impl CausalAttentionParams {
    pub(crate) fn new(
        row_count: u32,
        seq_len: u32,
        batch_size: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
    ) -> Self {
        Self {
            row_count,
            seq_len,
            batch_size,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            attention_window: seq_len,
            scale: 1.0 / (head_dim as f32).sqrt(),
            chunk_size: CAUSAL_CHUNK_SIZE,
            decay_scale: CAUSAL_DECAY_SCALE,
        }
    }

    pub(crate) fn with_attention_window(mut self, attention_window: u32) -> Self {
        assert!(
            attention_window > 0 && attention_window <= self.seq_len,
            "attention window must be in 1..=seq_len"
        );
        self.attention_window = attention_window;
        self
    }

    #[inline(always)]
    pub(crate) fn first_visible_key(&self, query: u32) -> u32 {
        let key_count = query + 1;
        if key_count > self.attention_window {
            key_count - self.attention_window
        } else {
            0
        }
    }

    #[inline(always)]
    pub(crate) fn key_is_visible(&self, query: u32, key: u32) -> bool {
        key <= query && key >= self.first_visible_key(query)
    }
}

#[cfg(test)]
mod tests {
    use super::CausalAttentionParams;

    fn params(window: u32) -> CausalAttentionParams {
        CausalAttentionParams::new(16, 16, 1, 64, 192, 1, 64).with_attention_window(window)
    }

    #[test]
    fn window_keeps_only_the_trailing_causal_keys() {
        let params = params(8);
        assert_eq!(params.first_visible_key(3), 0);
        assert_eq!(params.first_visible_key(7), 0);
        assert_eq!(params.first_visible_key(8), 1);
        assert_eq!(params.first_visible_key(15), 8);
        assert!(params.key_is_visible(15, 8));
        assert!(params.key_is_visible(15, 15));
        assert!(!params.key_is_visible(15, 7));
        assert!(!params.key_is_visible(15, 16));
    }
}

#[path = "causal/kernels.rs"]
pub mod kernels;
#[path = "causal/launcher.rs"]
mod launcher;
pub use launcher::CausalAttentionArgs;
