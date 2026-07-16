pub(crate) const THREADS_PER_BLOCK: u32 = 256;
pub(crate) const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / 32;
// Each lane owns two values from one logical 16-value Four-Six group.
pub(crate) const FOUR_SIX_THREADS_PER_GROUP: u32 = 8;
pub(crate) const GROUPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / FOUR_SIX_THREADS_PER_GROUP;
