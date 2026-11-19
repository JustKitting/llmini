use super::window;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn bytes(bytes: &[u8]) -> Self {
        Self::with_suffix(bytes, ".bin")
    }
    fn with_suffix(bytes: &[u8], suffix: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "gradient-window-{}-{}-{}{suffix}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).unwrap();
        Self(path)
    }
    fn tokens(ids: &[u16]) -> Self {
        Self::bytes(&ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>())
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn selected_shard_window_matches_the_native_overlapping_batch() {
    let fixture = Fixture::tokens(&(10..24).collect::<Vec<u16>>());
    let selected = window::load(&fixture.0, 2, 2, 3).unwrap();
    assert_eq!(selected.tokens, [12, 13, 14, 15, 15, 16, 17, 18]);
    assert_eq!(selected.metadata["token_offset"], 2);
    assert_eq!(selected.metadata["batch_size"], 2);
    assert_eq!(selected.metadata["seq_len"], 3);
    assert_eq!(selected.metadata["source_tokens"], 7);
}

#[test]
fn invalid_corpus_windows_fail_without_a_cuda_context() {
    let short = Fixture::tokens(&[1, 2, 3, 4, 5, 6]);
    assert!(window::load(&short.0, 0, 2, 3).is_err());
    assert!(window::load(&short.0, usize::MAX, 1, 1).is_err());
    assert!(window::load(&short.0, 0, usize::MAX, 2).is_err());
    assert!(window::load(&short.0, 0, 0, 3).is_err());
    let odd = Fixture::bytes(&[1, 0, 2]);
    assert!(window::load(&odd.0, 0, 1, 1).is_err());
    let invalid = Fixture::tokens(&[1, 2, 3, 4, 5, 6, u16::MAX]);
    assert!(window::load(&invalid.0, 0, 2, 3).is_err());
}

#[test]
fn independent_window_has_its_own_row_targets_and_offsets() {
    let ids = [1_u16, 2, 3, 45, 11, 12, 13, 55, 6, 7, 8, 65, 21, 22, 23, 75];
    let bytes = ids
        .iter()
        .flat_map(|id| id.to_ne_bytes())
        .collect::<Vec<_>>();
    let fixture = Fixture::with_suffix(&bytes, ".windows.bin");
    let first = window::load(&fixture.0, 0, 2, 3).unwrap();
    let second = window::load(&fixture.0, 8, 2, 3).unwrap();
    assert_eq!(first.tokens, ids[..8]);
    assert_eq!(second.tokens, ids[8..]);
    assert_eq!(second.metadata["source_tokens"], 8);
    assert!(window::load(&fixture.0, 1, 2, 3).is_err());
}
