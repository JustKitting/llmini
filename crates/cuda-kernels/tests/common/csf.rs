pub fn encode_csf_scales(logical: &[u8], rows: usize, k: usize) -> Vec<u8> {
    rust_kernels_cuda::nvfp4::csf::encode_rowwise_host(logical, rows, k)
}
