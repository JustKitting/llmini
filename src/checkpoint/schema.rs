pub(super) fn tensor_count(block_count: usize) -> u32 {
    const TOKEN_EMBEDDING: u32 = 1;
    const XSA_ALPHAS: u32 = 1;
    const FINAL_NORM: u32 = 2;
    const NEXT_LATENT: u32 = 8;
    const BLOCK: u32 = 15;

    TOKEN_EMBEDDING + XSA_ALPHAS + FINAL_NORM + NEXT_LATENT + block_count as u32 * BLOCK
}

pub(super) fn pre_canon_tensor_count(block_count: usize) -> u32 {
    const TOKEN_EMBEDDING: u32 = 1;
    const XSA_ALPHAS: u32 = 1;
    const FINAL_NORM: u32 = 2;
    const NEXT_LATENT: u32 = 8;
    const BLOCK: u32 = 13;

    TOKEN_EMBEDDING + XSA_ALPHAS + FINAL_NORM + NEXT_LATENT + block_count as u32 * BLOCK
}
