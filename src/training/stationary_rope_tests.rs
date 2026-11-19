//! Full positional-component failure cases declared before implementation.
mod model;
mod operator;

#[test]
fn stationary_position_graph_matches_model_build_and_shift_mask() {
    assert_eq!(
        gpt2_nvfp4::GPT2_STATIONARY_ROPE,
        rust_kernels_cuda::position::STATIONARY_ROPE
    );
    for dim in 0..64u32 {
        let expected = if gpt2_nvfp4::GPT2_STATIONARY_ROPE {
            dim >= 32
        } else {
            (16..32).contains(&dim) || dim >= 48
        };
        assert_eq!(
            rust_kernels_cuda::position::key_offset_dim(dim, 64),
            expected
        );
    }
}

// Failure cases declared before repairing the optional-symbol loader:
// untagged legacy PTX must be accepted only by the default graph without any
// failed CUDA API calls; both tagged graphs must retain real device verification.
#[test]
#[ignore = "actual default/stationary/legacy CUDA modules; zero optimizer updates; run under memcheck"]
fn native_position_module_checks_actual_tagged_and_untagged_modules() -> crate::AppResult {
    let index = std::env::var("CUDA_DEVICE_INDEX")
        .unwrap_or("0".into())
        .parse::<usize>()?;
    let ctx = cuda_core::CudaContext::new(index)?;
    let stream = ctx.new_stream()?;
    let profile = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/profiles/20261115");
    for (file, stationary) in [
        ("kda-saturation-corrected.ptx", None),
        ("stationary-rope-default.ptx", Some(false)),
        ("stationary-rope.ptx", Some(true)),
    ] {
        let module = ctx.load_module_from_file(profile.join(file).to_str().unwrap())?;
        let expected = match stationary {
            None => !gpt2_nvfp4::GPT2_STATIONARY_ROPE,
            Some(graph) => graph == gpt2_nvfp4::GPT2_STATIONARY_ROPE,
        };
        let accepted = match rust_kernels_cuda::position::PositionModule::from_module(module) {
            Ok(position) => position.verify_graph(stream.as_ref()).is_ok(),
            Err(_) => false,
        };
        assert_eq!(
            accepted, expected,
            "wrong positional compatibility for {file}"
        );
        stream.synchronize()?;
    }
    println!(
        "native_position_module_compatibility_verified stationary={} actual_modules=3 optimizer_updates=0",
        gpt2_nvfp4::GPT2_STATIONARY_ROPE
    );
    Ok(())
}

#[test]
#[ignore = "native signature rejects the actual frozen full-RoPE PTX; zero model allocation/updates"]
fn native_stationary_rope_rejects_legacy_ptx_signature() -> crate::AppResult {
    assert!(gpt2_nvfp4::GPT2_STATIONARY_ROPE);
    let index = std::env::var("CUDA_DEVICE_INDEX")
        .unwrap_or("0".into())
        .parse::<usize>()?;
    let ctx = cuda_core::CudaContext::new(index)?;
    let legacy = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/profiles/20261115/kda-saturation-corrected.ptx");
    let module = ctx.load_module_from_file(legacy.to_str().unwrap())?;
    let rejected = match rust_kernels_cuda::position::PositionModule::from_module(module) {
        Err(_) => true,
        Ok(position) => position.verify_graph(ctx.new_stream()?.as_ref()).is_err(),
    };
    assert!(
        rejected,
        "full-RoPE PTX was silently accepted by stationary-position host"
    );
    Ok(())
}
