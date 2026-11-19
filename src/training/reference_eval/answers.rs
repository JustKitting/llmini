use crate::{AppResult, reference_validation};
use burn::serde::Deserialize;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use llama2_tokenizer::{TOKENIZER_NAME, TrainingTokenizer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

use super::super::{TokenDataLoader, Trainer, data::VALIDATION_WINDOWS};
use super::input::Window;

#[derive(Deserialize)]
#[serde(crate = "burn::serde")]
struct Answer {
    task: String,
    prompt: String,
    answer: String,
}

#[derive(Default)]
struct Score {
    examples: usize,
    bytes: usize,
    targets: usize,
    nats: f64,
}

impl Score {
    fn json(&self) -> Value {
        json!({"examples":self.examples, "answer_bytes":self.bytes, "target_tokens":self.targets,
            "mean_token_ce":self.nats/self.targets as f64,
            "bits_per_byte":self.nats/self.bytes as f64/std::f64::consts::LN_2})
    }
}

fn plan(tokenizer: &TrainingTokenizer, item: &Answer) -> AppResult<Vec<Window>> {
    if item.task.trim().is_empty() || item.prompt.is_empty() || item.answer.is_empty() {
        return Err("task, prompt and answer must be nonempty".into());
    }
    let boundary = item.prompt.len();
    let tokens =
        tokenizer.encode_ordinary_with_offsets(&format!("{}{}", item.prompt, item.answer))?;
    if tokens
        .iter()
        .any(|&(_, (start, end))| start < boundary && boundary < end)
    {
        return Err(
            "a token crosses the prompt/answer boundary; separate them with a newline".into(),
        );
    }
    let start = tokens
        .iter()
        .position(|&(_, (_, end))| end > boundary)
        .ok_or("answer has no scored tokens")?;
    let mut ids = Vec::with_capacity(tokens.len() + 1);
    ids.push(u16::try_from(tokenizer.bos_token())?);
    for (id, _) in tokens {
        ids.push(u16::try_from(id)?);
    }
    let mut windows = Vec::new();
    let targets = ids.len() - 1;
    for scored in (start..targets).step_by(reference_validation::STRIDE) {
        let next = (scored + reference_validation::STRIDE).min(targets);
        let context_start = next.saturating_sub(GPT2_SEQ_LEN);
        let mut padded = vec![0; GPT2_SEQ_LEN + 1];
        let slice = &ids[context_start..=next];
        padded[..slice.len()].copy_from_slice(slice);
        windows.push(Window {
            ids: padded,
            lo: scored - context_start,
            hi: next - context_start,
        });
    }
    Ok(windows)
}

pub(in crate::training) fn run(
    checkpoint: &Path,
    answers: &Path,
    output: Option<&Path>,
    replay_record: Option<&Path>,
) -> AppResult {
    if !checkpoint.is_file() {
        return Err(format!("checkpoint does not exist: {}", checkpoint.display()).into());
    }
    let data = fs::read_to_string(answers)?;
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let mut tasks = BTreeMap::<String, Score>::new();
    let mut windows = Vec::new();
    for (line_index, line) in data.lines().enumerate() {
        let item: Answer = serde_json::from_str(line)
            .map_err(|err| format!("{}:{}: {err}", answers.display(), line_index + 1))?;
        let plans = plan(&tokenizer, &item)
            .map_err(|err| format!("{}:{}: {err}", answers.display(), line_index + 1))?;
        let score = tasks.entry(item.task.clone()).or_default();
        score.examples += 1;
        score.bytes += item.answer.len();
        windows.extend(plans.into_iter().map(|window| (item.task.clone(), window)));
    }
    if windows.is_empty() {
        return Err("the answer file is empty".into());
    }

    let checkpoint_sha = reference_validation::sha256(checkpoint)?;
    let started = Instant::now();
    let mut trainer = Trainer::from_checkpoint(0x47505432, checkpoint)?;
    let (expected_ce, verified_replay) = super::replay_expectation(checkpoint, replay_record)?;
    if replay_record.is_some() {
        let validation = TokenDataLoader::fineweb_validation_tokens()?;
        let actual_ce = trainer.eval_loss_windows(&validation, VALIDATION_WINDOWS)?;
        if expected_ce.is_some_and(|ce| (ce - f64::from(actual_ce)).abs() > 1e-4)
            || !actual_ce.is_finite()
        {
            return Err("checkpoint CE differs from the supplied replay record".into());
        }
    }
    for group in windows.chunks(GPT2_BATCH_SIZE) {
        let flat = group
            .iter()
            .flat_map(|(_, window)| window.ids.iter().copied())
            .collect::<Vec<_>>();
        let batch = trainer.batch_from_windows(&flat, group.len())?;
        let losses = trainer.eval_token_losses(&batch)?;
        for (sequence, (task, window)) in group.iter().enumerate() {
            let index = sequence * GPT2_SEQ_LEN;
            let values = &losses[index + window.lo..index + window.hi];
            if values.iter().any(|loss| !loss.is_finite()) {
                return Err(format!("nonfinite answer loss in task {task}").into());
            }
            let score = tasks
                .get_mut(task)
                .expect("task counted when reading answers");
            score.targets += values.len();
            score.nats += values.iter().map(|&v| f64::from(v)).sum::<f64>();
        }
    }
    let total = tasks.values().fold(Score::default(), |mut total, score| {
        total.examples += score.examples;
        total.bytes += score.bytes;
        total.targets += score.targets;
        total.nats += score.nats;
        total
    });
    let output = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| checkpoint.with_file_name("answer_validation.json"));
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let result = json!({
        "status":"completed", "checkpoint":checkpoint, "checkpoint_sha256":checkpoint_sha,
        "answer_file":answers, "answer_file_sha256":format!("{:x}", Sha256::digest(data.as_bytes())),
        "tokenizer":TOKENIZER_NAME, "context_tokens":GPT2_SEQ_LEN,
        "stride_tokens":reference_validation::STRIDE, "teacher_forced":true,
        "optimizer_updates":trainer.buffers.optimizer_state.step,
        "elapsed_s":started.elapsed().as_secs_f64(), "total":total.json(),
        "tasks":tasks.iter().map(|(name, score)| (name.clone(), score.json())).collect::<BTreeMap<_,_>>(),
        "verified_replay_provenance":verified_replay,
    });
    fs::write(&output, serde_json::to_string_pretty(&result)? + "\n")?;
    println!("answer_eval_result={}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_windows_score_the_suffix_exactly_once() -> AppResult<()> {
        let tokenizer = TrainingTokenizer::from_default_assets()?;
        let item = Answer {
            task: "test".into(),
            prompt: "Question: What language is this?\nAnswer:\n".into(),
            answer: "français. ".repeat(GPT2_SEQ_LEN),
        };
        let joined = format!("{}{}", item.prompt, item.answer);
        let expected = tokenizer
            .encode_ordinary_with_offsets(&joined)?
            .into_iter()
            .filter(|&(_, (start, _))| start >= item.prompt.len())
            .map(|(id, _)| id as u16)
            .collect::<Vec<_>>();
        let windows = plan(&tokenizer, &item)?;
        assert!(windows.len() > 1);
        let scored = windows
            .iter()
            .flat_map(|w| w.ids[w.lo + 1..=w.hi].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(scored, expected);
        Ok(())
    }
}
