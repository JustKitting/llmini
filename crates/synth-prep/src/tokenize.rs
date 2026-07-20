use llama2_tokenizer::TrainingTokenizer;

use super::AppResult;
use super::shards::ShardWriter;

pub fn tokenize_doc(
    text: &str,
    tokenizer: &TrainingTokenizer,
    writer: &mut ShardWriter,
) -> AppResult<usize> {
    let ids = tokenizer.encode_document(text)?;
    let token_count = ids.len();

    for id in ids {
        let token = u16::try_from(id)?;
        writer.push(token)?;
    }

    Ok(token_count)
}
