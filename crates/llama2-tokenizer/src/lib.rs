use std::error::Error;
use std::io;
use std::path::Path;

use tokenizers::Tokenizer;

pub type AppResult<T> = Result<T, Box<dyn Error>>;

include!(concat!(env!("OUT_DIR"), "/training_tokenizer_config.rs"));

// Compatibility name for model code that still refers to the historical
// package constant. This is the padded model/output vocabulary, not
// necessarily the number of token IDs emitted by the tokenizer.
pub const VOCAB_SIZE: usize = MODEL_VOCAB_SIZE;

pub struct TrainingTokenizer {
    tokenizer: Tokenizer,
}

impl TrainingTokenizer {
    pub fn from_default_assets() -> AppResult<Self> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/tokenizers")
            .join(TOKENIZER_ASSET_SUBDIR);
        Self::from_file(root.join("tokenizer.json"))
    }

    pub fn from_file(path: impl AsRef<Path>) -> AppResult<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(tokenizer_error)?;
        if tokenizer.get_vocab_size(false) != TOKENIZER_VOCAB_SIZE {
            return Err(format!(
                "{TOKENIZER_NAME} tokenizer vocab size is {}, expected {TOKENIZER_VOCAB_SIZE}",
                tokenizer.get_vocab_size(false)
            )
            .into());
        }
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> AppResult<Vec<u32>> {
        let mut ids = Vec::with_capacity(DOCUMENT_PREFIX.len() + text.len() / 4);
        ids.extend_from_slice(DOCUMENT_PREFIX);
        ids.extend(self.encode_ordinary(text)?);
        Ok(ids)
    }

    pub fn encode_ordinary(&self, text: &str) -> AppResult<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(tokenizer_error)?
            .get_ids()
            .to_vec())
    }

    pub fn encode_document(&self, text: &str) -> AppResult<Vec<u32>> {
        let mut ids =
            Vec::with_capacity(DOCUMENT_PREFIX.len() + text.len() / 4 + DOCUMENT_SUFFIX.len());
        ids.extend_from_slice(DOCUMENT_PREFIX);
        ids.extend(self.encode_ordinary(text)?);
        ids.extend_from_slice(DOCUMENT_SUFFIX);
        Ok(ids)
    }

    pub fn bos_token(&self) -> u32 {
        BOS_TOKEN
    }

    pub fn eos_token(&self) -> u32 {
        EOS_TOKEN
    }

    pub fn decode(&self, ids: &[u32]) -> AppResult<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(tokenizer_error)
            .map_err(Into::into)
    }
}

pub type Llama2Tokenizer = TrainingTokenizer;

fn tokenizer_error(error: Box<dyn Error + Send + Sync>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_assets_match_selected_tokenizer() -> AppResult<()> {
        let tokenizer = TrainingTokenizer::from_default_assets()?;
        match TOKENIZER_NAME {
            "llama2" => assert_eq!(
                tokenizer.encode_ordinary(
                    "First Citizen:\nBefore we proceed any further, hear me speak."
                )?,
                [
                    3824, 21353, 19642, 29901, 13, 18743, 591, 8469, 738, 4340, 29892, 8293, 592,
                    7726, 29889,
                ]
            ),
            "mistral_v01" => {
                assert_eq!(
                    tokenizer.encode_ordinary("Hello, world!")?,
                    [22557, 28725, 1526, 28808]
                );
            }
            _ => unreachable!(),
        }
        assert_eq!(
            &tokenizer.encode("Tokenizer test")?[..DOCUMENT_PREFIX.len()],
            DOCUMENT_PREFIX
        );
        let document = tokenizer.encode_document("Tokenizer test")?;
        assert!(document.starts_with(DOCUMENT_PREFIX));
        assert!(document.ends_with(DOCUMENT_SUFFIX));
        Ok(())
    }
}
