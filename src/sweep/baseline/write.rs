use std::{
    fs,
    io::{self, Write},
    path::Path,
};

use super::record::Record;
use crate::fs_utils::ensure_parent;

const DEFAULT_SEQ_LEN: usize = 2048;

pub(super) fn record(path: &Path, record: &Record) -> io::Result<()> {
    ensure_parent(path)?;
    let mut file = fs::File::create(path)?;
    writeln!(file, "VAL_LOSS={:.6}", record.val_loss)?;
    if let Some(steps) = record.completed_steps {
        writeln!(file, "COMPLETED_STEPS={steps}")?;
    }
    if let Some(elapsed_s) = record.elapsed_s {
        writeln!(file, "TRAIN_ELAPSED_S={elapsed_s:.6}")?;
    }
    if let Some(screen_loss) = record.screen_loss {
        writeln!(file, "SCREEN_LOSS={screen_loss:.6}")?;
    }
    if let Some(steps) = record.screen_completed_steps {
        writeln!(file, "SCREEN_COMPLETED_STEPS={steps}")?;
    }
    if let Some(elapsed_s) = record.screen_elapsed_s {
        writeln!(file, "SCREEN_ELAPSED_S={elapsed_s:.6}")?;
    }
    if let Some(reason) = &record.screen_reason {
        writeln!(file, "SCREEN_REASON={reason}")?;
    }
    writeln!(file, "LOG_PATH={}", record.log_path.display())?;
    writeln!(file, "GPT2_SEQ_LEN={DEFAULT_SEQ_LEN}")?;
    write_env(&mut file, record.candidate.build_env())?;
    write_env(&mut file, record.candidate.run_env())
}

fn write_env(file: &mut fs::File, values: Vec<(&'static str, String)>) -> io::Result<()> {
    for (name, value) in values {
        writeln!(file, "{name}={value}")?;
    }
    Ok(())
}
