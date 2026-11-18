use clap::Parser;
use std::error::Error;
use std::path::PathBuf;
#[path = "prepare_fineweb/reference.rs"]
mod reference;
#[path = "../reference_validation.rs"]
#[allow(dead_code)]
mod reference_validation;

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Parser)]
struct Args {
    /// Official GPT-2 FineWeb validation shard; prepare reference documents only.
    #[arg(long, requires_all = ["reference_vocab", "reference_output"])]
    reference_tokens: Option<PathBuf>,
    #[arg(long, requires = "reference_tokens")]
    reference_vocab: Option<PathBuf>,
    #[arg(long, requires = "reference_tokens")]
    reference_output: Option<PathBuf>,
}

fn main() -> AppResult {
    let args = Args::parse();
    match (
        args.reference_tokens,
        args.reference_vocab,
        args.reference_output,
    ) {
        (Some(tokens), Some(vocab), Some(output)) => reference::prepare(&tokens, &vocab, &output),
        _ => synth_prep::fineweb::parse_data(),
    }
}
