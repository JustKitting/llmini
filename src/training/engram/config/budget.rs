//! Capacity changes preserve the graph and serialize the actual prime sizes.
use super::{Config, HEAD_DIM, HEADS, LAYERS, is_prime};
use crate::AppResult;

impl Config {
    pub fn multistream_with_table_budget(
        requested: Option<usize>,
    ) -> AppResult<[Self; LAYERS.len()]> {
        let mut configs = std::array::from_fn(Self::reference_multistream);
        let default = configs.iter().map(Self::table_len).sum::<usize>();
        let requested = requested.unwrap_or(default);
        if !(default..=2_000_000_000).contains(&requested) {
            return Err(format!(
                "Engram table budget must be between {default} and 2000000000 lookup values"
            )
            .into());
        }
        if requested == default {
            return Ok(configs);
        }
        let values_per_capacity = LAYERS.len() * 2 * HEADS * HEAD_DIM;
        let mut prime = (requested / values_per_capacity) as u32;
        for config in &mut configs {
            for slot in &mut config.primes {
                loop {
                    prime += 1;
                    if prime > 1_000_000 {
                        return Err("Engram table budget exceeds checkpoint prime limits".into());
                    }
                    if is_prime(prime) {
                        break;
                    }
                }
                *slot = prime;
            }
        }
        let actual = configs.iter().map(Self::table_len).sum::<usize>();
        if actual < requested || actual - requested >= 1_000_000 {
            return Err("Engram table budget cannot satisfy the prime rounding bound".into());
        }
        Ok(configs)
    }

    pub fn multistream_with_ngram_budget(
        requested: Option<usize>,
        four_grams: bool,
    ) -> AppResult<[Self; LAYERS.len()]> {
        let mut configs = Self::multistream_with_table_budget(requested)?;
        if !four_grams {
            return Ok(configs);
        }

        let target_rows = configs.iter().map(Self::table_len).sum::<usize>() / HEAD_DIM;
        let primes = four_gram_primes(target_rows)?;
        for (config, heads) in configs.iter_mut().zip(primes.chunks_exact(3 * HEADS)) {
            config.version = 5;
            config.primes = heads.to_vec();
            let mut rng = gpt2_nvfp4::Gpt2Rng::new(10007 * config.layer as u64 + 4);
            let bound = (i64::MAX as u64 / llama2_tokenizer::MODEL_VOCAB_SIZE as u64) / 2;
            let draw = (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
            config.multipliers.push((draw % bound) * 2 + 1);
        }
        Ok(configs)
    }
}

fn four_gram_primes(target_rows: usize) -> AppResult<Vec<u32>> {
    const MAX_PRIME: u32 = 1_000_000;
    let heads = LAYERS.len() * 3 * HEADS;
    let center = (target_rows / heads) as u32;
    // Find nearby distinct primes on both sides of the mean. Reserve two
    // heads to match the existing table size after prime rounding.
    let per_side = heads / 2 - 1;
    let mut primes = (128..center)
        .rev()
        .filter(|&p| is_prime(p))
        .take(per_side)
        .chain((center..=MAX_PRIME).filter(|&p| is_prime(p)).take(per_side))
        .collect::<Vec<_>>();
    if primes.len() != heads - 2 {
        return Err("could not distribute the four-gram table budget across all heads".into());
    }
    let pair_sum = (target_rows as u32)
        .checked_sub(primes.iter().sum::<u32>())
        .ok_or("could not balance the four-gram prime sizes")?;
    let first = pair_sum.saturating_sub(MAX_PRIME).max(128);
    for low in (first..pair_sum.div_ceil(2).min(MAX_PRIME)).rev() {
        let high = pair_sum - low;
        if is_prime(low) && is_prime(high) && !primes.contains(&low) && !primes.contains(&high) {
            primes.extend([low, high]);
            primes.sort_unstable();
            return Ok(primes);
        }
    }
    Err("could not exactly match the three-gram table budget with distinct primes".into())
}
