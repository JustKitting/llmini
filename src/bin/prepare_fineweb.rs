use clap::{Parser, Subcommand};
use std::error::Error;
use std::num::NonZeroUsize;
use std::path::PathBuf;
#[path = "prepare_fineweb/curated.rs"]
mod curated;
#[path = "prepare_fineweb/interleave.rs"]
mod interleave;
#[path = "prepare_fineweb/reference.rs"]
mod reference;
#[path = "../reference_validation.rs"]
#[allow(dead_code)]
mod reference_validation;

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(args_conflicts_with_subcommands = true)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    /// Official GPT-2 FineWeb validation shard; prepare reference documents only.
    #[arg(long, requires_all = ["reference_vocab", "reference_output"])]
    reference_tokens: Option<PathBuf>,
    #[arg(long, requires = "reference_tokens")]
    reference_vocab: Option<PathBuf>,
    #[arg(long, requires = "reference_tokens")]
    reference_output: Option<PathBuf>,
    /// Read a local Parquet file and write a screened, train-only token stream.
    #[arg(long, conflicts_with = "reference_tokens",
        requires_all = ["curated_output", "curated_tokens", "exclude_documents"])]
    curated_parquet: Option<PathBuf>,
    #[arg(long, requires = "curated_parquet")]
    curated_output: Option<PathBuf>,
    /// Number of tokens to write, including any final next-token target.
    #[arg(long, requires = "curated_parquet")]
    curated_tokens: Option<NonZeroUsize>,
    #[arg(long, requires = "curated_parquet")]
    exclude_documents: Option<PathBuf>,
    /// Reserved prompt and answer JSONL. Repeat for multiple files.
    #[arg(long, requires = "curated_parquet")]
    exclude_answers: Vec<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Disperse curated rows using the matched B8 training mix.
    Interleave {
        /// Training manifest containing the original contiguous source blocks.
        #[arg(long)]
        source: PathBuf,
        /// 1-based index of a curated shard. Repeat for each curated shard.
        #[arg(long, required = true)]
        curated_shard: Vec<NonZeroUsize>,
        /// New directory for the mixed windows and training.json.
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() -> AppResult {
    let args = Args::parse();
    if let Some(Command::Interleave {
        source,
        curated_shard,
        output,
    }) = args.command
    {
        return interleave::prepare(&source, &output, &curated_shard);
    }
    if let (Some(parquet), Some(output), Some(tokens), Some(documents)) = (
        args.curated_parquet,
        args.curated_output,
        args.curated_tokens,
        args.exclude_documents,
    ) {
        return curated::prepare(
            &parquet,
            &output,
            tokens.get(),
            &documents,
            &args.exclude_answers,
        );
    }
    match (
        args.reference_tokens,
        args.reference_vocab,
        args.reference_output,
    ) {
        (Some(tokens), Some(vocab), Some(output)) => reference::prepare(&tokens, &vocab, &output),
        _ => synth_prep::fineweb::parse_data(),
    }
}
