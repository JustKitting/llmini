use std::fs;
use std::path::{Path, PathBuf};

use super::env_nonempty;
use crate::AppResult;

mod info;
mod paths;

pub(super) use crate::fs_utils::ensure_parent;
pub(super) use info::build_run_info;
use paths::default_run_dir;

#[derive(Clone)]
pub(super) struct RunOutput {
    dir: PathBuf,
}

impl RunOutput {
    pub(super) fn new(dataset: &str, label: &str) -> AppResult<Self> {
        let dir = env_nonempty("TRAIN_RUN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| default_run_dir(dataset, label));
        if dir.exists() && fs::read_dir(&dir)?.next().is_some() {
            return Err(format!("run output directory is not empty: {}", dir.display()).into());
        }
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub(super) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn path(&self, file_name: &str) -> PathBuf {
        self.dir.join(file_name)
    }

    pub(super) fn write_info(&self, info: &str) -> AppResult {
        fs::write(self.path("run_info.txt"), info)?;
        Ok(())
    }
}

pub(super) fn save_model_path(run_output: &RunOutput) -> Option<PathBuf> {
    let value = env_nonempty("TRAIN_SAVE_MODEL")?;
    if value == "1" || value.eq_ignore_ascii_case("true") {
        Some(run_output.path("model.ckpt"))
    } else {
        Some(PathBuf::from(value))
    }
}

pub(super) fn write_generated_text(run_output: &RunOutput, text: &str) -> AppResult<PathBuf> {
    let path = run_output.path("generated.txt");
    ensure_parent(&path)?;
    fs::write(&path, text)?;
    Ok(path)
}
